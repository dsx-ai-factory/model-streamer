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
    DIST: {"device": "cuda", "dist": True},
    # a whole vLLM model load: needs vllm installed and the model's config.json
    "vllm_default": {"device": "cuda", "load_format": "auto"},
    "vllm_streamer": {"device": "cuda", "load_format": "runai_streamer"},
    "vllm_instanttensor": {"device": "cuda", "load_format": "instanttensor"},  # needs instanttensor too
    # the same with tensor parallelism over --nproc ranks, one GPU each
    "vllm_dist_default": {"device": "cuda", "load_format": "auto", "dist": True},
    "vllm_dist_streamer": {"device": "cuda", "load_format": "runai_streamer", "dist": True},
    "vllm_dist_instanttensor": {"device": "cuda", "load_format": "instanttensor", "dist": True},
}
VLLM = tuple(t for t in TESTS if t.startswith("vllm_"))


def is_dist(test):
    """A test that runs on --nproc ranks under torchrun."""
    return bool(TESTS[test].get("dist"))
DEFAULT_PROFILE = "machine_profile.json"
VALIDATE = 20  # tensors compared byte for byte per run (across the ranks, for the distributed test)
VALIDATE_MAX_MIB = 128  # the largest tensor picked for that
TIMEOUT_S = 3600  # allowed per run

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(BENCH_DIR, "bench.py")  # what a measured process is started with
CALIBRATE = os.path.join(BENCH_DIR, "calibrate.py")
