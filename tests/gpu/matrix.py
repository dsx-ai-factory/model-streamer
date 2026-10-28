# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rows of the real-GPU test, and the memory limit of each. No torch and no GPU, so it is unit tested on
the CPU (test_matrix.py).

A row is one load of one model:

    model     one of the models of gen_models.py
    dist      streamed with is_distributed, over the two ranks, or not
    policy    the partition policy of a distributed row: spans, chunks or files
    owned     the owned argument of stream_files
    device    "cuda": stream to the rank's GPU; "cpu_to": stream to host memory, then copy each tensor with .to()
    limit     the host memory limit, which decides how the ring of read buffers is used (see memory_limit)

Three sets of rows, chosen with GPU_MATRIX:

    core    the rows that matter most, one option at a time on the richest model: a quick run
    full    every valid combination
    smoke   a few loads of the small real model (real_model.py), with the streamer's default settings
"""

import itertools
import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_models import MODEL_NAMES  # noqa: E402
from real_model import MODEL_NAME as REAL_MODEL  # noqa: E402

MIB = 1 << 20
WORLD = 2  # ranks of the test

# The streamer's default number of read buffers (RUNAI_STREAMER_RING_BUFFERS).
RING_BUFFERS = 18

Row = namedtuple("Row", "id model dist policy owned device limit")

# How the host ring of read buffers is used. N buffers of B bytes each, where B = max(largest range,
# budget // 18) and N = min(18, budget // B), with budget = min(limit, size of the model):
#   tight    the ring has two or three buffers and the model passes through it many times: a buffer is read
#            while another is consumed, and every buffer is reused
#   deep     18 buffers, and the model still passes through them more than once
#   default  40 GB: a model that small fits in the ring whole and no buffer is ever reused
LIMITS = ("tight", "deep", "default")

# A deep ring needs 18 buffers of the largest tensor to be smaller than the model, or nothing is reused. That
# holds for these two models; giant_tensor (a 2 GiB tensor in a 4 GiB model) cannot have a deep ring that recycles, and
# many_files (a few tensors of up to 10 MiB among small files, 5 to 11 MB in all) can only for some seeds.
DEEP_MODELS = ("mixed", "many_tensors")

POLICIES = ((False, None), (True, "spans"), (True, "chunks"), (True, "files"))


def memory_limit(manifest, distributed, level):
    """The value of RUNAI_STREAMER_MEMORY_LIMIT for a row, or None to leave it unset.

    A distributed load divides the limit, a node total, among the ranks, so a rank gets limit / WORLD.
    tight: about a sixteenth of the model, but never below twice the largest range, times the ranks. A range
    must fit in one buffer, and twice the largest range leaves room for two buffers, so a buffer can be read
    while the previous one is consumed. (Where the whole model is smaller than two largest ranges, as in
    giant_tensor, the ring still has one buffer.) The largest range is a tensor, or a file's header: the
    streamer reads each header as one range, and many_tensors' is 10 MB.
    deep: 18 largest tensors per rank, so the ring has all its 18 buffers."""
    if level == "default":
        return None
    ranks = WORLD if distributed else 1
    total = manifest["total_data_bytes"]
    largest_tensor = manifest["largest_tensor_bytes"]
    largest_range = max([largest_tensor] + [f["header_len"] for f in manifest["files"]])
    tight = max(total // 16, 2 * largest_range * ranks, MIB)
    if level == "tight":
        return tight
    if level == "deep":
        if RING_BUFFERS * largest_tensor >= total:
            raise ValueError("a ring of %d buffers of %d bytes does not recycle in a model of %d bytes" % (
                RING_BUFFERS, largest_tensor, total))
        return max(tight, RING_BUFFERS * largest_tensor * ranks)
    raise ValueError("unknown limit %r" % (level,))


def is_valid(model, dist, policy, owned, device, limit):
    if dist and device != "cuda":
        return False  # distributed streaming needs a CUDA destination
    if policy == "files" and model == "many_tensors":
        return False  # one file: there is nothing to give each rank
    if limit == "deep" and model not in DEEP_MODELS:
        return False
    return True


def row_id(model, dist, policy, owned, device, limit):
    return "%s.%s.%s.%s.%s" % (model, "dist." + policy if dist else "solo", "owned" if owned else "borrowed",
                               device, limit)


def full_rows():
    rows = []
    for model in MODEL_NAMES:
        for (dist, policy), owned, device, limit in itertools.product(POLICIES, (True, False),
                                                                        ("cuda", "cpu_to"), LIMITS):
            if is_valid(model, dist, policy, owned, device, limit):
                rows.append(Row(row_id(model, dist, policy, owned, device, limit), model, dist, policy, owned,
                                device, limit))
    return rows


# One option at a time on the richest and cheapest model (mixed), and one solo and one distributed load of every
# other model (many_files also with the files policy).
CORE_IDS = (
    "mixed.solo.owned.cuda.tight",
    "mixed.solo.borrowed.cuda.tight",
    "mixed.dist.spans.owned.cuda.tight",
    "mixed.dist.chunks.owned.cuda.tight",
    "mixed.solo.owned.cpu_to.tight",
    "mixed.solo.owned.cuda.deep",
    "mixed.solo.owned.cuda.default",
    "many_tensors.solo.owned.cuda.tight",
    "many_tensors.dist.spans.owned.cuda.tight",
    "giant_tensor.solo.owned.cuda.tight",
    "giant_tensor.dist.spans.owned.cuda.tight",
    "many_files.solo.owned.cuda.tight",
    "many_files.dist.files.owned.cuda.tight",
    "many_files.dist.spans.owned.cuda.tight",
)


def core_rows():
    by_id = {row.id: row for row in full_rows()}
    return [by_id[wanted] for wanted in CORE_IDS]


# The small real model, loaded the way a user loads it: the default memory limit. One solo load straight to the GPU,
# one distributed over the two ranks, and one to host memory and then .to() (what most callers do).
SMOKE_ROWS = (
    (False, None, True, "cuda"),
    (True, "spans", True, "cuda"),
    (False, None, False, "cpu_to"),
)


def smoke_rows():
    return [Row(row_id(REAL_MODEL, dist, policy, owned, device, "default"), REAL_MODEL, dist, policy, owned, device,
                "default") for dist, policy, owned, device in SMOKE_ROWS]


def rows_for(mode):
    if mode == "core":
        return core_rows()
    if mode == "full":
        return full_rows()
    if mode == "smoke":
        return smoke_rows()
    raise ValueError("GPU_MATRIX must be core, full or smoke, not %r" % (mode,))


def row_prefixes(text):
    """The prefixes in a comma-separated list. Spaces around a prefix are dropped, and an empty entry (a trailing comma)
    is ignored: an empty prefix would match every row, and the filter would silently do nothing."""
    return tuple(part.strip() for part in (text or "").split(",") if part.strip())


def no_rows_outcome(filter_text, allow_empty, matrix_name):
    """(exit status, message) of a run that selected no row. Without a filter that is a mistake of the matrix. With one,
    it is a leg that has nothing to run only where allow_empty says so (the oldest-torch leg has just the core rows);
    anywhere else it is a filter that matches nothing, which is a typo, and a green leg that ran nothing would hide it."""
    prefixes = row_prefixes(filter_text)
    if not prefixes:
        return 1, "no rows selected"
    message = "no row of the %s matrix starts with %s" % (matrix_name, list(prefixes))
    if allow_empty:
        return 0, message + ": nothing to run on this leg"
    return 1, message + ": a typo in the filter? Only a leg that has just some of the rows may select none"


def filter_rows(rows, text):
    """The rows whose id starts with one of the prefixes in text; all of them when there is no prefix."""
    prefixes = row_prefixes(text)
    return [row for row in rows if row.id.startswith(prefixes)] if prefixes else list(rows)
