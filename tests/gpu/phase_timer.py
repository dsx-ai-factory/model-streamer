# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock timing of the phases of a GPU test run.

The point is to see where the time of a run goes (generating fixtures, writing them, verifying them,
loading, comparing, ...) so the slow parts can be found and shortened. It is not a benchmark of the
streamer.
"""

import json
import time
from contextlib import contextmanager


class PhaseTimer:
    """Collects one record per timed phase. Records are plain dicts so they can cross processes."""

    def __init__(self):
        self.records = []

    @contextmanager
    def phase(self, name, nbytes=None, **tags):
        start = time.perf_counter()
        try:
            yield
        finally:
            record = {"phase": name, "seconds": time.perf_counter() - start}
            if nbytes is not None:
                record["bytes"] = int(nbytes)
            record.update(tags)
            self.records.append(record)

    def extend(self, records):
        self.records.extend(records)


def _group_key(record):
    return record["phase"]


def report(records, top=None):
    """A table of phases, the slowest first, with each phase's share of the summed time.

    Phases that run in parallel processes add up to more than the wall time, so the share is of the
    summed time and not of the wall time."""
    groups = {}
    for record in records:
        entry = groups.setdefault(_group_key(record), {"seconds": 0.0, "count": 0, "bytes": 0})
        entry["seconds"] += record["seconds"]
        entry["count"] += 1
        entry["bytes"] += record.get("bytes", 0)
    total = sum(entry["seconds"] for entry in groups.values()) or 1.0
    rows = sorted(groups.items(), key=lambda item: item[1]["seconds"], reverse=True)
    if top is not None:
        rows = rows[:top]
    lines = ["%-44s %9s %6s %6s %10s" % ("phase", "seconds", "share", "count", "GiB/s")]
    for name, entry in rows:
        rate = ""
        if entry["bytes"] and entry["seconds"] > 0:
            rate = "%.2f" % (entry["bytes"] / 2**30 / entry["seconds"])
        lines.append("%-44s %9.2f %5.1f%% %6d %10s" % (
            name, entry["seconds"], 100.0 * entry["seconds"] / total, entry["count"], rate))
    return "\n".join(lines)


def dump_jsonl(records, path):
    with open(path, "w") as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    import os
    import sys

    loaded = []
    for name in sys.argv[1:]:
        if os.path.exists(name):
            with open(name) as handle:
                loaded.extend(json.loads(line) for line in handle if line.strip())
    print(report(loaded))
