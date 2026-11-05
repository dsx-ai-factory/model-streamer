#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The entry of the default environment (tests/gpu/compose.yaml): a clean CUDA container (nvidia/cuda, Ubuntu, nothing but
# the system Python) on a host with two GPUs (GPU_WORLD=1 runs on one: a suite leaves out what needs two, and says so). It
# sets the container up, then runs the suites, and says what ran:
#
#   1. set up: system Python and pip, PyTorch and numpy (GPU_TORCH_SPEC, GPU_NUMPY_SPEC), the streamer: the wheel in
#      GPU_WHEEL_DIR (/wheel, mounted: what the workflow built from this commit), the way a user installs it, or a
#      requirement specifier (GPU_STREAMER_SPEC, a released version, for a run on a machine of your own)
#   2. check that the container has the GPUs and NCCL, and (newest-torch leg) that it does not limit the newest torch
#   3. the CPU unit tests of the harness
#   4. the suites that GPU_SUITES selects (default streaming): tests/gpu/suites/<name>/suite.sh, see tests/gpu/README.md
#   5. the summary of the cases that ran, where the time went, and the verdict
# Every step is timed; the table at the end shows where the time went.
#
# An environment of another kind (other containers, another image, a toolchain) has an entry of its own, which sets it up
# and then does what steps 3 to 5 do with tests/gpu/harness/common.sh and suites.sh; tests/gpu/README.md says how.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RESULTS=${GPU_RESULTS:-test-results/gpu}
FIXTURES=${GPU_FIXTURES:-.gpu-fixtures}   # a suite gets a directory of its own in it (GPU_FIXTURES while it runs)
SEED=${GPU_SEED:-$(date +%s)}
WHEEL_DIR=${GPU_WHEEL_DIR:-/wheel}
STREAMER_SPEC=${GPU_STREAMER_SPEC:-}
TORCH_SPEC=${GPU_TORCH_SPEC:-torch}
NUMPY_SPEC=${GPU_NUMPY_SPEC:-numpy}
WORLD=${GPU_WORLD:-2}
case "$WORLD" in
    1|2) ;;
    *) echo "ERROR: GPU_WORLD must be 1 or 2, not '$WORLD'" >&2; exit 2 ;;
esac
export GPU_SEED="$SEED" GPU_WORLD="$WORLD"
export PYTHONPATH="$PWD/tests/gpu/harness${PYTHONPATH:+:$PYTHONPATH}"  # gpu_harness, for the suites and the steps
unset GPU_RESULTS GPU_FIXTURES  # theirs to set, per suite: the base directories are RESULTS and FIXTURES

mkdir -p "$RESULTS" "$FIXTURES"
# What an earlier run left here would be read as results of this one (summarize.py reads every file under it).
bash tests/gpu/harness/clean_results.sh "$RESULTS" $(ls tests/gpu/suites)
# The log is kept by whatever runs this script (the workflow pipes it to tee): a tee started here would be killed with
# the script, and could lose the last lines, which hold the verdict.

SETUP_TIMING="$RESULTS/timing-setup.jsonl"
: > "$SETUP_TIMING"
# shellcheck source=harness/common.sh
. tests/gpu/harness/common.sh
# shellcheck source=harness/suites.sh
. tests/gpu/harness/suites.sh

select_suites || exit 2

echo "== environment"
date -u
uname -r
. /etc/os-release && echo "$PRETTY_NAME"
echo "seed: $SEED"
echo "suites: ${SUITES[*]}   torch: $TORCH_SPEC   numpy: $NUMPY_SPEC"
echo "gpus: $WORLD   streamer: ${STREAMER_SPEC:-the wheel in $WHEEL_DIR}"
if [ "$WORLD" -lt 2 ]; then
    echo "== ONE GPU: what needs two GPUs is not run, and the summary says how many cases"
fi
timeout --signal=TERM --kill-after=10s 120s nvidia-smi
echo "ulimit -n: $(ulimit -n)   ulimit -l: $(ulimit -l)"
free -g | head -2
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true

# The image has no Python. What is installed is what a user needs: Python, pip and the certificates for HTTPS.
# phase runs these functions where `set -e` does not apply, so every command ends in `|| return`.
install_system_packages() {
    export DEBIAN_FRONTEND=noninteractive
    timeout --signal=TERM --kill-after=30s "$(cap 15)s" apt-get update -qq || return
    timeout --signal=TERM --kill-after=30s "$(cap 15)s" \
        apt-get install -y -qq --no-install-recommends python3 python3-pip ca-certificates || return
}
phase install_system_packages install_system_packages
python3 --version

install_python_packages() {
    local wheels wheel
    # Ubuntu 24.04 marks its Python as externally managed (PEP 668) and pip refuses to install into it. The container is
    # thrown away after the run, so the system Python is the environment (older pip ignores the variable).
    export PIP_BREAK_SYSTEM_PACKAGES=1
    # PyTorch and numpy first, at the versions of this run, so that installing the wheel cannot pick others.
    timeout --signal=TERM --kill-after=30s "$(cap 45)s" \
        python3 -m pip install --no-cache-dir "$TORCH_SPEC" "$NUMPY_SPEC" || return
    if [ -n "$STREAMER_SPEC" ]; then
        echo "installing the streamer from the requirement specifier: $STREAMER_SPEC"
        timeout --signal=TERM --kill-after=30s "$(cap 15)s" \
            python3 -m pip install --no-cache-dir "$STREAMER_SPEC" || return
        return 0
    fi
    mapfile -t wheels < <(find "$WHEEL_DIR" -name 'runai_model_streamer-*.whl')
    if [ "${#wheels[@]}" -ne 1 ]; then
        echo "ERROR: expected one runai_model_streamer wheel in $WHEEL_DIR, found ${#wheels[@]}: ${wheels[*]:-}" >&2
        return 1
    fi
    wheel=${wheels[0]}
    echo "installing $wheel"
    timeout --signal=TERM --kill-after=30s "$(cap 15)s" python3 -m pip install --no-cache-dir "$wheel" || return
}
phase install_python_packages install_python_packages
python3 -m pip list 2>/dev/null | grep -i -E "^(torch|numpy|humanize|runai)" || true

timeout --signal=TERM --kill-after=10s "$(cap 10)s" python3 tests/gpu/harness/gpu_check.py

# The newest-torch leg is meant to test the newest PyTorch. pip installs the newest release that this image's Python can
# use, so when torch stops supporting that Python the leg quietly tests an older one. Flag it: the run goes on and
# reports what it can, but it fails, and env-check.txt (shown in the job summary) says the base container has to change.
env_limited=0
if [ "${GPU_EXPECT_NEWEST_TORCH:-false}" = true ]; then
    echo "== is this container limiting the newest torch?"
    # Only the check's own verdict (exit status 3) flags the container. A check that cannot reach PyPI says so and
    # passes; one that crashes or hangs (a cap of its own) is not a verdict, and must not turn into a false flag.
    check_status=0
    # GPU_RESULTS is where the check writes env-check.txt, which the job summary shows (it was unset above: a suite's to set).
    GPU_RESULTS="$RESULTS" timeout --signal=TERM --kill-after=10s 180s python3 tests/gpu/harness/env_check.py newest-torch \
        || check_status=$?
    case "$check_status" in
        0) ;;
        3) env_limited=1 ;;
        *) echo "WARNING: the environment check did not complete (exit $check_status): the newest torch is not verified" ;;
    esac
fi

# The CPU unit tests of the harness (the unit tests of each suite run with the suite). A failure fails the run, but the GPU
# tests still run: their results are worth having. Without GPU_RESULTS, which the tests must not see.
echo "== self-test of the harness"
phase harness_unit_tests timeout --signal=TERM --kill-after=30s "$(cap 15)s" \
    bash tests/gpu/harness/unit_tests.sh harness || status=1

# What every suite runs under: the NCCL watchdog aborts the process on a timeout or an error (TORCH_NCCL_* from PyTorch
# 2.2, NCCL_* before), unbuffered output for a log that is followed, and the streamer's own log on stderr.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export RUNAI_STREAMER_LOG_TO_STDERR=1

run_suites
report_suites

# A container that limits the newest torch fails the leg, but the tests above did run, with the older torch. Say so at
# the end of the log, in the file that the job summary shows, and as an annotation on the run, with the verdict of the
# tests. The two are separate problems: the environment never explains a failing test, so failures are to be looked
# into whatever the environment is, and passing tests are not a test of the newest torch.
if [ "$env_limited" -ne 0 ]; then
    torch_version=$(python3 -c "import importlib.metadata as m; print(m.version('torch'))")
    if [ "$status" -eq 0 ]; then
        message="The tests ran with torch $torch_version, which is not the newest, and all passed. This leg is red only because of its environment: change the base container so that it can run the newest torch (see env-check.txt)."
    else
        message="The tests ran with torch $torch_version, which is not the newest, and some failed (see above). Two separate problems make this leg red: the failing tests, which need to be looked into regardless of the environment, and the environment, which needs a base container that can run the newest torch (see env-check.txt)."
    fi
    echo
    echo "== NOT TESTING THE NEWEST TORCH: $message"
    printf '\nTests of this leg: %s\n' "$message" >> "$RESULTS/env-check.txt"
    echo "::error title=The newest-torch leg is not testing the newest torch::$message"
    status=1
fi

finished=true
exit "$status"
