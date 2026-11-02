#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Remove what an earlier run left in a results directory, before a new run writes to it:
#
#   clean_results.sh RESULTS_DIR
#
# rows_summary.py reads every results file under the directory, so a file of an earlier run (the second rank of a run
# of two GPUs, the smoke results, a strategy that does not run this time) would be counted as part of the new run, and
# a missing result would look present. It does not remove run.log, which whatever runs the script may be writing to.
# CI always starts from an empty directory; a run on a machine of your own does not.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: clean_results.sh RESULTS_DIR" >&2; exit 2; }
dir=$1
[ -d "$dir" ] || exit 0

find "$dir" -type f \( -name 'rows-*.json' -o -name 'timing-*.jsonl' -o -name 'timing-summary.txt' -o -name 'env-check.txt' \) -delete
rm -rf "$dir/self-check" "$dir/smoke" "$dir/manifests"
