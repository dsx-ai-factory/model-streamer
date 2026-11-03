#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Remove what an earlier run left in a results directory, before a new run writes to it:
#
#   clean_results.sh RESULTS_DIR [SUITE ...]
#
# summarize.py reads every results file under the directory, so a file of an earlier run (the second rank of a run of two
# GPUs, the smoke results, a run that does not happen this time) would be counted as part of the new run, and a missing
# result would look present. This removes the results and timing files and the summaries, and the directory of each
# SUITE named (what a suite keeps there, its artifacts). It does not remove run.log, which whatever runs the script may
# be writing to. CI always starts from an empty directory; a run on a machine of your own does not.
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: clean_results.sh RESULTS_DIR [SUITE ...]" >&2; exit 2; }
dir=$1
shift
[ -d "$dir" ] || exit 0

find "$dir" -type f \( -name 'results-*.json' -o -name 'timing-*.jsonl' -o -name 'timing-summary.txt' -o -name 'env-check.txt' \) -delete
for suite in "$@"; do
    rm -rf "${dir:?}/$suite"
done
find "$dir" -mindepth 1 -type d -empty -delete
