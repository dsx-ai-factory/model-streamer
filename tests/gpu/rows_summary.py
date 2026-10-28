#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which rows of a GPU test run finished, from the results files that test_synthetic_models.py writes.

    rows_summary.py RESULTS_DIR SMOKE STRATEGY [STRATEGY ...]

SMOKE is "true" when the smoke test ran. Prints one line per results file (rows-<strategy>-rank<N>.json, anywhere under
RESULTS_DIR): how many of the expected rows finished, how many of those failed (a row that failed on purpose in a drill
is not counted), and the row in which the run ended if one was left started. Exits 1 when a file is incomplete, or when
a strategy, or the smoke test, has no results file at all.

A rank that fails inside a distributed row makes the NCCL watchdog abort the other one, which ends that strategy's run:
the rows after it never start, and must not hide behind the strategies that did finish.

Needs only the standard library."""

import glob
import json
import os
import sys


def summarize(results, smoke, strategies):
    """(lines, incomplete)."""
    lines = []
    incomplete = False
    for path in sorted(glob.glob(os.path.join(results, "**", "rows-*-rank*.json"), recursive=True)):
        with open(path) as handle:
            data = json.load(handle)
        rows = data["rows"]
        finished = [r for r in rows if r["ok"] is not None]
        unexpected = [r["row"] for r in finished if not r["ok"] and not r.get("drill")]
        line = "%s: %d of %d rows finished, %d failed" % (os.path.relpath(path, results), len(finished),
                                                           data["expected_rows"], len(unexpected))
        started = [r["row"] for r in rows if r["ok"] is None]
        if started:
            line += "; the run ended inside row " + started[0]
        if len(finished) < data["expected_rows"]:
            incomplete = True
            line += "  <-- INCOMPLETE"
        lines.append(line)
    for strategy in strategies:
        if not glob.glob(os.path.join(results, "rows-%s-rank*.json" % strategy)):
            incomplete = True
            lines.append("%s: no results at all  <-- INCOMPLETE" % strategy)
    if smoke and not glob.glob(os.path.join(results, "smoke", "rows-*-rank*.json")):
        incomplete = True
        lines.append("smoke: no results at all  <-- INCOMPLETE")
    return lines, incomplete


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or args[1] not in ("true", "false"):
        print("usage: rows_summary.py RESULTS_DIR true|false STRATEGY ...", file=sys.stderr)
        return 2
    lines, incomplete = summarize(args[0], args[1] == "true", args[2:])
    print("\n".join(lines))
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
