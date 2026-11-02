# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the suites of tests/gpu/suites/ that GPU_SUITES selects (default: streaming), and report. Sourced after common.sh by
# the entry of an environment (tests/gpu/setup_and_run.sh), from the root of the checkout:
#
#   select_suites || exit 1      the suites to run, from GPU_SUITES (a comma-separated list of names)
#   run_suites                   each suite: its unit tests, its preparation, its tests, its artifacts
#   report_suites                the summary of what ran (summarize.py), and where the time went
#
# A suite is a directory with a suite.sh, which this sources and which defines (suite_run and suite_expect are required of
# every suite by tests/gpu/test_suites.py; the runner itself runs a suite without suite_expect, and would not miss its results):
#
#   suite_description="one line"             what the suite tests
#   suite_unit_tests=(tests/gpu/suites/NAME/test_x.py ...)   CPU-only tests of the suite's own code, run before it
#   suite_prepare()   once, before the tests: fixtures, services to wait for. A failure skips the suite's tests (and the
#                     run fails). Run by `phase`, so bash's `set -e` does not apply: end every command with `|| return`
#   suite_run()       the tests, with run_step, run_unittest and run_torchrun (common.sh); required
#   suite_expect()    prints the result files that the suite must have written, as globs relative to its results directory
#                     (results-<run>-rank*.json): a suite that did not run, or a run that wrote nothing, is then missed
#   suite_collect()   copies what is worth keeping to debug a failure into $GPU_RESULTS (the results directory)
#   suite_timing()    prints files of timings (timing-*.jsonl) that are not in the results directory
#
# While a suite's functions run, these are set: GPU_SUITE (its name), GPU_RESULTS (a directory of its own under the results
# directory), GPU_FIXTURES (a directory of its own for what it generates), and WORLD (the GPUs of the run).

SUITES_DIR=tests/gpu/suites
SUITES=()
EXPECTED_RESULTS=()
EXTRA_TIMING_FILES=()

select_suites() {
    local names name
    IFS=',' read -r -a names <<< "${GPU_SUITES:-streaming}"
    SUITES=()
    for name in "${names[@]}"; do
        name=${name// /}
        [ -n "$name" ] || continue
        if [ ! -f "$SUITES_DIR/$name/suite.sh" ]; then
            echo "ERROR: there is no suite '$name' (a directory $SUITES_DIR/<name> with a suite.sh); the suites are: $(ls "$SUITES_DIR" | tr '\n' ' ')" >&2
            return 1
        fi
        SUITES+=("$name")
    done
    if [ "${#SUITES[@]}" -eq 0 ]; then
        echo "ERROR: GPU_SUITES selects no suite" >&2
        return 1
    fi
}

# What one suite defined must not leak into the next one.
forget_suite() {
    unset -f suite_prepare suite_run suite_expect suite_collect suite_timing
    unset suite_description
    suite_unit_tests=()
}

run_suite() {
    local name=$1 line
    forget_suite
    # shellcheck disable=SC1090
    . "$SUITES_DIR/$name/suite.sh"
    echo "== suite $name: ${suite_description:-}"
    export GPU_SUITE="$name" GPU_RESULTS="$RESULTS/$name" GPU_FIXTURES="$FIXTURES/$name"
    mkdir -p "$GPU_RESULTS" "$GPU_FIXTURES"
    if ! declare -F suite_run >/dev/null; then
        echo "ERROR: the suite $name does not define suite_run" >&2
        status=1
        return 0
    fi

    # The CPU unit tests of the suite's own code. A failure fails the run, but the tests of the suite still run: their
    # results are worth having. Without GPU_RESULTS, which they must not see: it is where the results are written.
    if [ "${#suite_unit_tests[@]}" -gt 0 ]; then
        echo "== self-test of the suite $name"
        phase "$name.unit_tests" env -u GPU_RESULTS timeout --signal=TERM --kill-after=30s "$(cap 15)s" \
            python3 -m unittest "${suite_unit_tests[@]}" || status=1
    fi

    local prepared=true
    if declare -F suite_prepare >/dev/null; then
        echo "== prepare the suite $name"
        if ! phase "$name.prepare" suite_prepare; then
            prepared=false
            status=1
            echo "ERROR: the suite $name could not be prepared: its tests are not run (the run fails at the end)"
        fi
    fi
    if [ "$prepared" = true ]; then
        suite_run
    fi

    if declare -F suite_expect >/dev/null; then
        while IFS= read -r line; do
            [ -n "$line" ] && EXPECTED_RESULTS+=("$name/$line")
        done < <(suite_expect)
    fi
    if declare -F suite_timing >/dev/null; then
        while IFS= read -r line; do
            [ -n "$line" ] && EXTRA_TIMING_FILES+=("$line")
        done < <(suite_timing)
    fi
    if declare -F suite_collect >/dev/null; then
        suite_collect || true
    fi
    return 0
}

run_suites() {
    local name
    for name in "${SUITES[@]}"; do
        run_suite "$name"
    done
}

# Which cases ran. A rank that fails inside a distributed case makes the NCCL watchdog abort the other one, which ends that
# run: the cases after it never start. They must not hide behind the runs that did finish.
report_suites() {
    local files
    echo "== cases that ran"
    timeout --signal=TERM --kill-after=10s 120s python3 tests/gpu/harness/summarize.py "$RESULTS" \
        ${EXPECTED_RESULTS[@]+"${EXPECTED_RESULTS[@]}"} || status=1

    echo "== where the time went"
    mapfile -t files < <(find "$RESULTS" -name 'timing-*.jsonl' | sort)
    python3 -m gpu_harness.phase_timer "${files[@]}" ${EXTRA_TIMING_FILES[@]+"${EXTRA_TIMING_FILES[@]}"} \
        | tee "$RESULTS/timing-summary.txt" || true  # a truncated line must not end the script before the verdict
}
