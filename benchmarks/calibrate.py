#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure what this machine can physically do, and write it to a JSON machine profile.

Linux only. Standard library only; torch is used for the host-to-GPU numbers when it is importable.

    python3 calibrate.py --path /mnt/models --out machine_profile.json

--path is a directory on the storage under test. Temporary files are created there and removed.
Each ceiling is the median of --repeats runs, in GiB/s (2**30 bytes per second): disk reads, host memory
first-touch and copy, host to GPU copy, and with --s3-uri (needs s5cmd) S3 reads. The stages are
described in README.md next to this script.
"""
import argparse
import os

from streamer_bench import calibration


def main():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", help="directory on the storage under test (needed unless disk is skipped)")
    ap.add_argument("--out", default="machine_profile.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--size-gib", type=float, default=16, help="bytes read per disk measurement")
    ap.add_argument("--files", type=int, default=8, help="files read in parallel")
    ap.add_argument("--mem-gib", type=float, default=4, help="buffer size for the memory measurements")
    ap.add_argument("--threads", type=int, default=min(32, cpus))
    ap.add_argument("--skip", default="", help="comma list of: disk,host,gpu")
    ap.add_argument("--s3-uri", help="s3://bucket/prefix whose objects are read for the S3 ceilings")
    ap.add_argument("--s3-concurrency", default="4,8,16,32", help="parts in flight per object, swept")
    ap.add_argument("--s3-part-mib", default="8,32", help="part sizes in MiB, swept")
    calibration.run(ap.parse_args())


if __name__ == "__main__":
    main()
