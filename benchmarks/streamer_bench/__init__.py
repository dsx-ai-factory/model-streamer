# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The load benchmark behind bench.py. See ../README.md."""
import os

GIB = 2**30
# Each test: where the tensors end up, how vLLM is asked to load (vLLM tests), whether it runs on
# --nproc ranks, and "about": its setup in a sentence or two, shown next to the test in the report.
TESTS = {
    "streamer_cpu": {
        "device": "cpu",
        "about": 'The streamer reads every file into host memory (device="cpu"). One process; no GPU is used.'},
    "streamer_cuda": {
        "device": "cuda",
        "about": 'The streamer reads every file and delivers the tensors on one GPU (device="cuda"). One process.'},
    "safeopen_to_cuda": {
        "device": "cuda",
        "about": 'The baseline without the streamer: safetensors.safe_open maps each file and .to("cuda") copies '
                 "one tensor at a time to one GPU. One process."},
    "streamer_dist": {
        "device": "cuda", "dist": True,
        "about": "The streamer's distributed load (is_distributed=True) under torchrun, one GPU per rank: each rank "
                 "reads its share of the files and the ranks pass tensors to each other, so every rank receives "
                 "all of them."},
    # a whole vLLM model load: needs vllm installed and the model's config.json
    "vllm_default": {
        "device": "cuda", "load_format": "auto",
        "about": "vLLM loads the model on one GPU with its default safetensors loader."},
    "vllm_streamer": {
        "device": "cuda", "load_format": "runai_streamer",
        "about": "vLLM loads the model on one GPU as `vllm serve --load-format runai_streamer` does: the streamer "
                 "reads into host memory and vLLM copies each tensor to the GPU."},
    "vllm_streamer_gpu": {
        "device": "cuda", "load_format": "runai_streamer", "extra": {"distributed": True},
        "about": 'vLLM loads the model on one GPU with the streamer and the loader option {"distributed": true}: '
                 "the streamer delivers the tensors on the GPU itself, with no copy by vLLM."},
    "vllm_instanttensor": {  # needs instanttensor too
        "device": "cuda", "load_format": "instanttensor",
        "about": "vLLM loads the model on one GPU with InstantTensor (--load-format instanttensor)."},
    "vllm_dist_default": {
        "device": "cuda", "load_format": "auto", "dist": True,
        "about": "vLLM with tensor parallelism over the ranks, one GPU each, and its default safetensors loader."},
    "vllm_dist_streamer": {
        "device": "cuda", "load_format": "runai_streamer", "dist": True, "extra": {"distributed": True},
        "about": "vLLM with tensor parallelism over the ranks, one GPU each, and the streamer's distributed load "
                 'straight to the GPUs (--load-format runai_streamer with {"distributed": true}).'},
    "vllm_dist_instanttensor": {
        "device": "cuda", "load_format": "instanttensor", "dist": True,
        "about": "vLLM with tensor parallelism over the ranks, one GPU each, and InstantTensor."},
}
DIST = "streamer_dist"
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
