#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks that the container has what the GPU tests need, and says which versions they run with.

    gpu_check.py

GPU_WORLD (1 or 2) is the number of GPUs that the run uses. Fails when the streamer was not imported from the installed
wheel, when fewer GPUs than that are visible, or when PyTorch has no NCCL.
"""
import os

import torch
import torch.distributed as dist
import runai_model_streamer
from runai_model_streamer.libstreamer import STREAMER_LIBRARY


def main():
    nccl = torch.cuda.nccl.version()
    nccl = ".".join(map(str, nccl)) if isinstance(nccl, tuple) else str(nccl)
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; NCCL {nccl}", flush=True)
    print(f"streamer package: {runai_model_streamer.__file__}", flush=True)
    print(f"streamer library: {STREAMER_LIBRARY}", flush=True)
    if "site-packages" not in runai_model_streamer.__file__ and "dist-packages" not in runai_model_streamer.__file__:
        raise RuntimeError("the streamer was not imported from the installed wheel")
    world = int(os.environ["GPU_WORLD"])
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        raise RuntimeError(f"GPU tests require {world} visible CUDA GPUs")
    if not dist.is_nccl_available():
        raise RuntimeError("GPU tests require an NCCL-enabled PyTorch build")
    for index in range(world):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)


if __name__ == "__main__":
    main()
