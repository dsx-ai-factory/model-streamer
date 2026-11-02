# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# What the entry of an environment and the suites share: the time budget and the caps, the timed phases, the steps that
# a suite runs, and what to say when the script ends early. Sourced (not run) by tests/gpu/setup_and_run.sh, from the
# root of the checkout, after it has set the variables that this reads:
#
#   RESULTS          the results directory         SETUP_TIMING   the file of the timings of the phases (under RESULTS)
#   WORLD            the GPUs (ranks) of the run   GPU_BUDGET_MINUTES (optional)  the budget of the whole run
#
# For a suite (see tests/gpu/README.md, "Adding a test"), these are what a suite.sh may use:
#
#   run_step NAME MINUTES COMMAND...        a step of the run, with a cap: START and END lines, its time recorded, and the
#                                           run fails if it fails (the run goes on: the other steps are worth having)
#   run_unittest NAME MINUTES TEST...       run_step of ordinary unittest modules in one process (gpu_harness.run_unittest)
#   run_torchrun NAME MINUTES SCRIPT [ARG...]   run_step of a script under torchrun with WORLD ranks (one GPU each)
#   phase NAME COMMAND...                   like run_step for a preparation: timed, but a failure is the caller's to handle
#   cap MINUTES                             the seconds for `timeout`: MINUTES, or what is left of the budget if less
#
# `status` is the exit status of the run: 0 until something fails. The functions that phase runs do not have `set -e`
# (bash turns it off for them), so every command in them ends in `|| return`.

START_EPOCH=$(date +%s)
BUDGET_SECONDS=$(( ${GPU_BUDGET_MINUTES:-285} * 60 ))
REPORT_RESERVE_SECONDS=300
status=0
# The command that launches a script on the ranks of the run, for a step that needs something of its own around it:
# run_step NAME MINUTES env -u VAR "${TORCHRUN[@]}" SCRIPT ARGS...
TORCHRUN=(torchrun --standalone --nnodes=1 --nproc_per_node="$WORLD" --max_restarts=0)

# Time. Every step that can hang has a cap (cap MINUTES, below), generous: several times what the step takes on a healthy
# runner. The whole run has a budget, GPU_BUDGET_MINUTES (the workflow's job timeout less 15 minutes), that no step can
# exceed: a cap is cut to what is left of the budget, less 5 minutes for the report. So the script always gets to say
# which cases ran, whatever the caps add up to, and they do not need adjusting to fit the job's timeout.
#
# cap MINUTES: the seconds that a step may take, for timeout: its own cap, or what is left of the budget if that is less
# (but at least a minute, so that a step is never given none).
cap() {
    local want=$(( $1 * 60 )) left
    left=$(( START_EPOCH + BUDGET_SECONDS - $(date +%s) - REPORT_RESERVE_SECONDS ))
    if [ "$left" -lt 60 ]; then left=60; fi
    echo $(( want < left ? want : left ))
}

# A script that stops before its end (a setup step failed, a precondition is not met) says so, instead of leaving the log
# to end on its last step with no report. The entry sets finished=true when it reaches its end.
finished=false
on_exit() {
    local rc=$?
    if [ "$finished" != true ]; then
        echo
        echo "== THE RUN ENDED EARLY (exit $rc): see the last FAILED or ERROR line above. No test was run after it, and none is reported."
        echo "::error title=The GPU test run ended early::The script stopped before it finished (exit $rc): see the last FAILED or ERROR line in the log"
    fi
}
trap on_exit EXIT

# record NAME START END: append the wall time of a phase to the timing file and print it.
record() {
    local secs
    secs=$(awk -v s="$2" -v e="$3" 'BEGIN { printf "%.2f", e - s }')
    printf '{"phase": "%s", "seconds": %s}\n' "$1" "$secs" >> "$SETUP_TIMING"
    echo "[$(date +%T)] END   $1  ${secs}s"
}

# phase NAME COMMAND...: run a command, print START and END lines, and record its wall time.
phase() {
    local name=$1; shift
    local start rc=0
    echo "[$(date +%T)] START $name"
    start=$(date +%s.%N)
    "$@" || rc=$?
    record "$name" "$start" "$(date +%s.%N)"
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%T)] FAILED $name (exit $rc)"
    fi
    return "$rc"
}

# stop_stray_workers: kill what a torchrun that was stopped (a timeout) left behind. The workers of a rank that ignored
# the signal keep their GPU memory and the rendezvous port, and the next step would fail on them and be blamed for it.
# Reads /proc, so that it needs no pkill: a base image is free to leave procps out. What it looks for is a process that runs
# a file of a suite.
stop_stray_workers() {
    local file pid cmd parent ancestors=" $$ "
    # What runs this script is not a stray, whatever its command line says (a unit test of the runner, a wrapper).
    parent=$$
    while [ "$parent" -gt 1 ]; do
        parent=$(sed -E 's/^.*\) [A-Za-z] ([0-9]+) .*$/\1/' "/proc/$parent/stat" 2>/dev/null) || break
        [ -n "$parent" ] || break
        ancestors+="$parent "
    done
    for file in /proc/[0-9]*/cmdline; do
        pid=${file#/proc/}
        pid=${pid%/cmdline}
        cmd=$(tr '\0' ' ' < "$file" 2>/dev/null) || continue
        if [[ "$cmd" == *tests/gpu/suites/* ]] && [[ "$ancestors" != *" $pid "* ]]; then
            echo "killing a worker left behind (pid $pid): $cmd"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

# run_step NAME MINUTES COMMAND...: environment variables for the command go before run_step (VAR=value run_step ...),
# or in front of the command with env (run_step NAME MINUTES env -u VAR command ...).
run_step() {
    local name=$1 minutes=$2 start
    shift 2
    echo "[$(date +%T)] START run.$name"
    start=$(date +%s.%N)
    if ! timeout --signal=TERM --kill-after=30s "$(cap "$minutes")s" "$@"; then
        status=1
        echo "[$(date +%T)] FAILED run.$name"
    fi
    stop_stray_workers
    record "run.$name" "$start" "$(date +%s.%N)"
}

run_unittest() {
    local name=$1 minutes=$2
    shift 2
    run_step "$name" "$minutes" python3 -m gpu_harness.run_unittest --suite "$GPU_SUITE" --run "$name" "$@"
}

run_torchrun() {
    local name=$1 minutes=$2
    shift 2
    run_step "$name" "$minutes" "${TORCHRUN[@]}" "$@"
}
