# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure what this machine can physically do, one module per stage, into a machine profile."""
import json
import os
import platform
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone

GIB = 2**30

from . import disk, gpu, host, memory, s3  # noqa: E402 - the stages take GIB from here

SCHEMA = 1
CEILINGS = ["disk_direct_read_parallel", "disk_direct_read_fresh_dest", "disk_direct_read_single",
            "disk_buffered_read_cold", "host_first_touch_1t", "host_first_touch_nt",
            "host_memcpy_1t", "host_memcpy_nt", "h2d_pinned", "h2d_pageable",
            "s3_get_single", "s3_get_parallel"]


def median_of(values):
    return round(statistics.median(values), 2) if values else None


class Meter:
    """Collects every stage's runs (GiB/s) and the notes about what could not be measured."""

    def __init__(self, repeats):
        self.repeats = repeats
        self.raw = {}
        self.notes = []

    def record(self, name, values, suffix=""):
        self.raw[name] = [round(v, 2) for v in values]
        print(f"  {name:32s} {median_of(values):8.2f} GiB/s   {suffix or 'runs ' + str(self.raw[name])}", flush=True)

    def rate(self, name, fn, nbytes):
        """Time fn, which moves nbytes, `repeats` times."""
        values = []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            fn()
            values.append(nbytes / GIB / (time.perf_counter() - t0))
        self.record(name, values)


def run(a):
    skip = set(filter(None, a.skip.split(",")))
    if not sys.platform.startswith("linux"):
        sys.exit("calibrate.py needs Linux (O_DIRECT, /proc)")
    if "disk" not in skip:
        if not a.path:
            sys.exit("--path is required unless --skip includes disk")
        os.makedirs(a.path, exist_ok=True)
    if a.s3_uri and not shutil.which("s5cmd"):
        sys.exit("--s3-uri needs s5cmd on PATH (pip install s5cmd)")

    meter = Meter(a.repeats)
    budget = memory.memory_budget_gib()
    size_gib = a.size_gib
    while size_gib * 2.5 > budget and size_gib > 1:
        size_gib /= 2
    if size_gib != a.size_gib and "disk" not in skip:
        meter.notes.append(f"disk size reduced from {a.size_gib} to {size_gib} GiB to fit {budget:.0f} GiB of memory")
    mem_gib = min(a.mem_gib, budget / 3)
    file_size = max(1, int(size_gib * GIB / a.files) // disk.BLOCK) * disk.BLOCK

    print(f"host: {platform.node()}  cpus={len(os.sched_getaffinity(0))}  memory budget={budget:.0f} GiB")
    if "host" not in skip:
        memory.measure(meter, mem_gib, a.threads)
    if "disk" not in skip:
        disk.measure(meter, a.path, a.files, file_size, a.threads)
    if "gpu" not in skip:
        gpu.measure(meter)
    s3_params, s3_best = s3.measure(meter, a) if a.s3_uri else (None, None)

    info = host.host_info(a.path or ".")
    if "disk" in skip:
        info.pop("storage", None)
    if s3_params:
        info["s3"] = s3_params
    profile = {
        "schema": SCHEMA, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": info,
        "params": {"repeats": a.repeats, "files": a.files, "file_gib": round(file_size / GIB, 3),
                   "mem_gib": round(mem_gib, 2), "threads": a.threads,
                   "data": "random 64 MiB block per file, repeated", "s3_best": s3_best},
        "ceilings_gibps": {n: median_of(meter.raw.get(n, [])) for n in CEILINGS},
        "raw_gibps": meter.raw, "notes": meter.notes,
    }
    with open(a.out, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"wrote {a.out}")
    for n in meter.notes:
        print("note:", n)
