# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The load benchmark behind bench.py. See ../README.md."""
import os

GIB = 2**30
# The loaders that are compared, in the order they are reported, with the name each is shown under.
LOADERS = {"safetensors": "Safetensors", "instanttensor": "InstantTensor", "streamer": "Model Streamer"}

# Each test is one loader bringing the whole model to one destination: host memory ("cpu"), one GPU ("cuda"), or
# every rank's GPU on --nproc ranks ("dist"). "about" is its setup in a sentence or two, shown in the report.
TESTS = {
    "safetensors_cpu": {
        "loader": "safetensors", "device": "cpu",
        "about": "safetensors.safe_open maps each file and every tensor is read into host memory by touching "
                 "each of its pages. One process; no GPU is used."},
    "streamer_cpu": {
        "loader": "streamer", "device": "cpu",
        "about": 'The streamer reads every file into host memory (device="cpu"). One process; no GPU is used.'},
    "safetensors_cuda": {
        "loader": "safetensors", "device": "cuda",
        "about": 'safetensors.safe_open maps each file and .to("cuda") copies one tensor at a time to one GPU. '
                 "One process."},
    "instanttensor_cuda": {
        "loader": "instanttensor", "device": "cuda",
        "about": "instanttensor.safe_open reads all files and delivers the tensors on one GPU. One process."},
    "streamer_cuda": {
        "loader": "streamer", "device": "cuda",
        "about": 'The streamer reads every file and delivers the tensors on one GPU (device="cuda"). One process.'},
    "safetensors_dist": {
        "loader": "safetensors", "device": "cuda", "dist": True,
        "about": 'Every rank, one GPU each under torchrun, loads all files itself: safetensors.safe_open and '
                 '.to("cuda") per tensor.'},
    "instanttensor_dist": {
        "loader": "instanttensor", "device": "cuda", "dist": True,
        "about": "InstantTensor's distributed load under torchrun, one GPU per rank: instanttensor.safe_open with "
                 "the ranks' process group, so every rank receives all tensors."},
    "streamer_dist": {
        "loader": "streamer", "device": "cuda", "dist": True,
        "about": "The streamer's distributed load (is_distributed=True) under torchrun, one GPU per rank: each rank "
                 "reads its share of the files and the ranks pass tensors to each other, so every rank receives "
                 "all of them."},
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

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(BENCH_DIR, "bench.py")  # what a measured process is started with
CALIBRATE = os.path.join(BENCH_DIR, "calibrate.py")
