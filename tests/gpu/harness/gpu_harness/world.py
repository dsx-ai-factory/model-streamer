# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How many GPUs, and so ranks, a run uses. No torch: usable by anything, and unit tested on the CPU."""

import os

DEFAULT_WORLD = 2  # the GPUs of the runner: what CI runs, and what a run uses unless GPU_WORLD says one


def gpu_world(text=None):
    """The ranks of a run, from GPU_WORLD: DEFAULT_WORLD (two GPUs, the default and what CI runs) or 1 (one GPU). A suite
    that needs two ranks leaves those cases out on one, and says so (see Session.skip)."""
    if text is None:
        text = os.environ.get("GPU_WORLD")
    if text in (None, ""):
        return DEFAULT_WORLD
    if text not in ("1", str(DEFAULT_WORLD)):
        raise ValueError("GPU_WORLD must be 1 or %d, not %r" % (DEFAULT_WORLD, text))
    return int(text)
