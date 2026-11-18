# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host to GPU 0 copy, from pinned and from pageable memory. Needs torch; skipped with a note without it."""
import time

from .. import GIB


def h2d(pinned, gib=1, reps=8):
    """GiB/s of copying a `gib` buffer to the GPU `reps` times, after one warm-up copy."""
    import torch
    n = gib * GIB
    src = torch.empty(n, dtype=torch.uint8)
    src.fill_(1)
    if pinned:
        src = src.pin_memory()
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst.copy_(src, non_blocking=pinned)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dst.copy_(src, non_blocking=pinned)
    torch.cuda.synchronize()
    return gib * reps / (time.perf_counter() - t0)


def measure(meter):
    """h2d_pinned and h2d_pageable."""
    try:
        import torch
    except ImportError:
        meter.notes.append("torch not importable: h2d ceilings are null")
        return
    if not torch.cuda.is_available():
        meter.notes.append("no CUDA device: h2d ceilings are null")
        return
    print("host to GPU 0")
    for name, pinned in (("h2d_pinned", True), ("h2d_pageable", False)):
        meter.record(name, [h2d(pinned) for _ in range(meter.repeats)])
