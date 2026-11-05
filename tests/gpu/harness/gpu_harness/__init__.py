# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the suites of the real-GPU tests share. Nothing in here is a test of the streamer.

    world           how many GPUs (ranks) a run uses: gpu_world()
    session         Session: the ranks of a run launched by torchrun, one GPU each, and what they write
    results         the results files that every suite writes and the summary reads
    run_unittest    runs ordinary unittest modules in one process and writes their results files
    object_store    a client of the object storage that the environment provides (boto3, from the environment)
    safetensors_file  a small safetensors writer, for tests that need a file and not a model
    tensors         the bytes of a tensor, and their CRC32
    phase_timer     wall-clock timing of the phases of a run

A suite (tests/gpu/suites/<name>/) imports from here; this imports nothing from a suite. See tests/gpu/README.md.
The modules are importable with tests/gpu/harness on sys.path, which the runner sets (PYTHONPATH).
"""
