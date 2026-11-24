# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The load benchmark behind bench.py. See ../README.md."""
import os

GIB = 2**30
# The loaders that are compared, in the order they are reported, with the name each is shown under.
LOADERS = {"safetensors": "Safetensors", "safetensors_prefetch": "Safetensors + prefetch",
           "instanttensor": "InstantTensor", "streamer": "Model Streamer"}

# Each test is one loader bringing the whole model to one destination: host memory ("cpu"), one GPU ("cuda"), or
# every rank's GPU on --nproc ranks ("dist"). How each one loads is the `load` function in worker.py.
TESTS = {
    "safetensors_cpu": {"loader": "safetensors", "device": "cpu"},
    "safetensors_prefetch_cpu": {"loader": "safetensors_prefetch", "device": "cpu"},
    "streamer_cpu": {"loader": "streamer", "device": "cpu"},
    "safetensors_cuda": {"loader": "safetensors", "device": "cuda"},
    "safetensors_prefetch_cuda": {"loader": "safetensors_prefetch", "device": "cuda"},
    "instanttensor_cuda": {"loader": "instanttensor", "device": "cuda"},
    "streamer_cuda": {"loader": "streamer", "device": "cuda"},
    "safetensors_dist": {"loader": "safetensors", "device": "cuda", "dist": True},
    "safetensors_prefetch_dist": {"loader": "safetensors_prefetch", "device": "cuda", "dist": True},
    "instanttensor_dist": {"loader": "instanttensor", "device": "cuda", "dist": True},
    "streamer_dist": {"loader": "streamer", "device": "cuda", "dist": True},
}


def loader_of(test):
    return TESTS[test]["loader"]


def group_of(test):
    """Where a test's tensors end up: "cpu", "cuda" or "dist". Tests of one group are compared with each other."""
    return "dist" if TESTS[test].get("dist") else TESTS[test]["device"]


def is_dist(test):
    """A test that runs on --nproc ranks under torchrun."""
    return bool(TESTS[test].get("dist"))


DEFAULT_PROFILE = "machine_profile.json"
VALIDATE = 20  # tensors compared byte for byte per run (across the ranks, for the distributed test)
VALIDATE_MAX_MIB = 128  # the largest tensor picked for that
TIMEOUT_S = 3600  # allowed per run
SCRATCH = "bench_scratch"  # where an s3:// model is downloaded to, unless --scratch-dir says otherwise
S3_CALIBRATE_GIB = 64  # how much of an s3:// model an automatic calibration reads
# The safetensors prefetch reads the files into the page cache ahead of the load, as vLLM does with
# --safetensors-load-strategy=prefetch. These are vLLM's defaults.
PREFETCH_THREADS = 8
PREFETCH_BLOCK = 16 << 20
GPU_IDLE_GIB = 1.0  # a GPU with more than this in use before the run is taken to be used by something else
SKIPPED = 3  # exit status of a run that measured nothing because the model does not fit where it was to go

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(BENCH_DIR, "bench.py")  # what a measured process is started with
CALIBRATE = os.path.join(BENCH_DIR, "calibrate.py")
