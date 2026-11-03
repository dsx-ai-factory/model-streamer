#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The CPU unit tests of the GPU tests: no GPU, no container. The one place that lists them.
#
#   tests/gpu/harness/unit_tests.sh [harness|all]
#
#   harness   the harness (tests/gpu/harness/test_*.py), and the tests of run.sh and of the suite contract
#   all       (default) those, and the unit tests that each suite lists in suite_unit_tests
#
# tests_entrypoint.sh runs `harness` before the suites; the runner runs the unit tests of each suite itself, with the
# time caps of a run (harness/suites.sh). Tests that need torch or the streamer are skipped where those are not installed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

scope=${1:-all}
case "$scope" in
    harness|all) ;;
    *) echo "usage: unit_tests.sh [harness|all]" >&2; exit 2 ;;
esac

export PYTHONPATH="$PWD/tests/gpu/harness${PYTHONPATH:+:$PYTHONPATH}"
unset GPU_RESULTS  # the tests make their own

tests=(tests/gpu/harness/test_*.py tests/gpu/test_run.py tests/gpu/test_suites.py)
if [ "$scope" = all ]; then
    for suite_file in tests/gpu/suites/*/suite.sh; do
        # In a subshell: a suite.sh only defines its variables and functions, and must not leak them into the next one.
        while IFS= read -r test_file; do
            [ -z "$test_file" ] || tests+=("$test_file")
        done < <(unset suite_unit_tests; . "$suite_file"; printf '%s\n' ${suite_unit_tests[@]+"${suite_unit_tests[@]}"})
    done
fi

exec "${PYTHON:-python3}" -m unittest "${tests[@]}"
