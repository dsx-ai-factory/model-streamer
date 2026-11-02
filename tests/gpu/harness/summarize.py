#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which cases of a GPU test run finished, from the results files that the suites write (gpu_harness/results.py).

    summarize.py RESULTS_DIR [EXPECTED_GLOB ...]

Prints one line per results file (results-<run>-rank<N>.json, anywhere under RESULTS_DIR): how many of the expected cases
finished, how many of those failed (a case with a fault injected on purpose, by a self-check, is not counted), and the
case in which the run ended if one was left started. Exits 1 when a file is incomplete, when an expected result is not
there at all, or when a run has the results file of one rank and not of the other (a rank that died before its first
write). Each EXPECTED_GLOB is a path, relative to RESULTS_DIR, that must match at least one file: it is how a suite says
what it was to write (a suite that does not run at all writes nothing, and would not otherwise be missed).

A rank that fails inside a distributed case makes the NCCL watchdog abort the other one, which ends that run: the cases after
it never start, and must not hide behind the runs that did finish.

A run that left cases out on purpose (on one GPU, the distributed ones) says so in its results, and the line is printed once at
the end: not a failure, but it must not be taken for a run of every case.

Needs only the standard library."""

import glob
import json
import os
import re
import sys

DEFAULT_WORLD = 2  # the ranks of a run whose results do not say; a results file says how many ranks its run had ("world")
_FILE = re.compile(r"^results-(?P<run>.+)-rank(?P<rank>\d+)\.json$")


def summarize(results, expected_globs=()):
    """(lines, incomplete)."""
    lines = []
    incomplete = False
    worlds = {}  # (directory, run) -> the ranks that the run was launched with, which each have a results file
    notes = []
    paths = sorted(glob.glob(os.path.join(results, "**", "results-*-rank*.json"), recursive=True))
    for path in paths:
        with open(path) as handle:
            data = json.load(handle)
        found = _FILE.match(os.path.basename(path))
        worlds[(os.path.dirname(path), found.group("run"))] = data.get("world", DEFAULT_WORLD)
        if data.get("skipped_cases") and data.get("skipped_note") and data["skipped_note"] not in notes:
            notes.append(data["skipped_note"])
        cases = data["cases"]
        finished = [c for c in cases if c["ok"] is not None]
        # A case with a fault injected on purpose (a self-check) has errors by design.
        unexpected = [c["case"] for c in finished if not c["ok"] and not c.get("fault")]
        line = "%s: %d of %d cases finished, %d failed" % (os.path.relpath(path, results), len(finished),
                                                            data["expected_cases"], len(unexpected))
        started = [c["case"] for c in cases if c["ok"] is None]
        if started:
            line += "; the run ended inside case " + started[0]
        if len(finished) < data["expected_cases"]:
            incomplete = True
            line += "  <-- INCOMPLETE"
        lines.append(line)
    # A run has one results file per rank. A rank that died before its first write leaves the file of the other rank, and
    # the run would look present.
    ranks_of = {}
    for path in paths:
        found = _FILE.match(os.path.basename(path))
        ranks_of.setdefault((os.path.dirname(path), found.group("run")), set()).add(int(found.group("rank")))
    for (directory, run), ranks in sorted(ranks_of.items()):
        missing = sorted(set(range(worlds[(directory, run)])) - ranks)
        if missing:
            incomplete = True
            lines.append("%s: no results file of rank %s  <-- INCOMPLETE" % (
                os.path.relpath(os.path.join(directory, "results-%s-rank*.json" % run), results),
                ", ".join(str(rank) for rank in missing)))
    for expected in expected_globs:
        if not glob.glob(os.path.join(results, expected)):
            incomplete = True
            lines.append("%s: no results at all  <-- INCOMPLETE" % expected)
    lines.extend(notes)
    return lines, incomplete


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: summarize.py RESULTS_DIR [EXPECTED_GLOB ...]", file=sys.stderr)
        return 2
    lines, incomplete = summarize(args[0], args[1:])
    print("\n".join(lines))
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
