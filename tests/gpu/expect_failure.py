#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks that a row of a GPU test run failed as an ordinary row, for the self-check that proves that a failing row fails
the run.

    expect_failure.py RESULTS_DIR STRATEGY ROW_ID ERROR_TEXT

setup_and_run.sh runs the test with a fault injected into a row and not declared (GPU_FAULT), so the row is an ordinary
failing row, and the test has to exit with a failure. A failure exit alone proves little, since a crash exits with one
too, so this checks the results the run left: on every rank the row finished, was not ok, was not a drill, and has an
error that contains ERROR_TEXT. Exits 0 when it is so, 1 when it is not.

Needs only the standard library."""

import json
import os
import sys

RANKS = 2  # the ranks of the test (matrix.WORLD)


def check(results, strategy, row_id, error_text):
    """(ok, message)."""
    for rank in range(RANKS):
        path = os.path.join(results, "rows-%s-rank%d.json" % (strategy, rank))
        if not os.path.exists(path):
            return False, "%s does not exist: the run left no results for rank %d" % (path, rank)
        with open(path) as handle:
            rows = {r["row"]: r for r in json.load(handle)["rows"]}
        row = rows.get(row_id)
        if row is None:
            return False, "rank %d has no row %s" % (rank, row_id)
        if row["ok"] is None:
            return False, "rank %d: row %s never finished" % (rank, row_id)
        if row["ok"]:
            return False, "rank %d: row %s with an injected fault was ok" % (rank, row_id)
        if row.get("drill"):
            return False, "rank %d: row %s was a drill, not an ordinary failing row" % (rank, row_id)
        if not any(error_text in error for error in row.get("errors", [])):
            return False, "rank %d: row %s failed, but with no error that contains %r: %s" % (
                rank, row_id, error_text, row.get("errors", [])[:2])
    return True, "row %s failed as an ordinary row on every rank" % row_id


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        print("usage: expect_failure.py RESULTS_DIR STRATEGY ROW_ID ERROR_TEXT", file=sys.stderr)
        return 2
    ok, message = check(*args)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
