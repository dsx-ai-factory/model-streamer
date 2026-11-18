# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The load benchmark behind bench.py. See ../README.md."""
import os

GIB = 2**30
DIST = "streamer_dist"
TESTS = {
    "streamer_cpu": {"device": "cpu"},
    "streamer_cpu_notowned": {"device": "cpu"},
    "streamer_cuda": {"device": "cuda"},
    "safeopen_page_in": {"device": "cpu"},
    "safeopen_to_cuda": {"device": "cuda"},
    DIST: {"device": "cuda"},
}
DEFAULT_PROFILE = "machine_profile.json"
VALIDATE = 20  # tensors compared byte for byte per run (across the ranks, for the distributed test)
VALIDATE_MAX_MIB = 128  # the largest tensor picked for that
TIMEOUT_S = 3600  # allowed per run

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(BENCH_DIR, "bench.py")  # what a measured process is started with
CALIBRATE = os.path.join(BENCH_DIR, "calibrate.py")
