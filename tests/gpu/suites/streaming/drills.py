# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the self-checks of the GPU test (the SelfChecks case of test_synthetic_models.py) look for. No torch, so it is
unit tested on the CPU (test_drills.py).

A fault is injected on purpose, and the check that is meant to catch it has to report it, as itself:

    corrupt  one bit of the first tensor of the row is flipped after it arrived: the CRC32 check must report that
             tensor, with an error of this form (RowRun.check)
    leak     a 32 MiB GPU tensor is kept alive after the row: the GPU memory check must report it

A rank that fails inside a distributed row is not something to inject: the NCCL watchdog aborts the other rank, so there
is nothing left to report it.
"""

MIB = 1 << 20
LEAK_DRILL_BYTES = 32 * MIB


def is_crc_error_of(error, tensor):
    """Whether an error of a row is the CRC32 error of this tensor: it starts with the name and the details in parentheses,
    and says crc32. (An error of another tensor, or another kind of error of this one, is not.)"""
    return tensor is not None and error.startswith(tensor + " (") and "crc32" in error


def is_leak_error(error):
    """Whether an error of a row is the GPU memory error."""
    return error.startswith("GPU memory leak")
