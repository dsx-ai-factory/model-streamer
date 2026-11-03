# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The results files of a run. Every suite writes them, and harness/summarize.py reads them, so a run that was cut short,
or that did not run at all, is seen whatever kind of test it was.

One file per rank of a run: results-<run>-rank<N>.json, under the results directory of the suite:

    {"suite": "streaming", "run": "sync_buffered",
     "world": 2,                        ranks of the run: the summary expects a file for each
     "expected_cases": 104,             how many cases the run is to finish
     "skipped_cases": 0, "skipped_note": "",   cases left out on purpose, and a line that says so
     "cases": [{"case": "mixed.solo.owned.cuda.tight", "ok": true, ...}, ...]}

A case is written with "ok": null from the moment it starts, so that when the process is killed in the middle of it the
file names that case. Any other keys of a case (seconds, errors, ...) are the suite's. No torch here.
"""

import json
import os
import re

FILE_PATTERN = re.compile(r"^results-(?P<run>.+)-rank(?P<rank>\d+)\.json$")


def results_filename(run, rank):
    """The file of a run for a rank. A run name may hold anything: it is made safe for a file name."""
    return "results-%s-rank%d.json" % ((run.replace(",", "+").replace("/", "_") or "default"), rank)


def timing_filename(run, rank):
    """The timing records of a run for a rank (see phase_timer.py), next to its results file."""
    return results_filename(run, rank).replace("results-", "timing-", 1).replace(".json", ".jsonl")


def write_results(directory, suite, run, rank, cases, expected_cases, world=1, skipped_cases=0, skipped_note=""):
    """Write (the whole file again) the cases of this run so far."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, results_filename(run, rank)), "w") as handle:
        json.dump({"suite": suite, "run": run, "world": world, "expected_cases": expected_cases,
                   "skipped_cases": skipped_cases, "skipped_note": skipped_note, "cases": cases}, handle, indent=1)
