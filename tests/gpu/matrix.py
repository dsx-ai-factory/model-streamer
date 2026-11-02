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
    limit     the host memory limit: tight, roomy or default (see memory_limit)

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

Row = namedtuple("Row", "id model dist policy owned device limit")

# The host memory limit, RUNAI_STREAMER_MEMORY_LIMIT, which bounds what the streamer holds of the model at a time. The
# levels are defined by what a user can see of it (the documented contract: a limit is a node total, divided between
# the ranks of the node, and no rank gets less than the largest tensor), not by how the streamer arranges its buffers:
#   tight    the largest tensor of the model for each rank, which is the least that the streamer gives a rank (a solo
#            load with less is refused, and the streamer raises the share of a rank to that): the load has to get
#            through the whole model with room for one tensor at a time
#   roomy    a quarter of the model, but not less than twice the tight limit: several tensors at a time, still far
#            less than the model, so memory has to be reused
#   default  40 GB: a model that small fits whole and nothing is reused
# A limit below the size of the model cannot hold the model: whatever the streamer does, it has to reuse memory.
LIMITS = ("tight", "roomy", "default")

# The models that a roomy limit makes sense for. giant_tensor is the model of a tensor of 2 GiB in 4.5 GB: twice its
# largest tensor is nearly the whole model, and the row would only repeat the tight one at the cost of minutes.
ROOMY_MODELS = ("mixed", "many_tensors", "many_files")

POLICIES = ((False, None), (True, "spans"), (True, "chunks"), (True, "files"))


def memory_limit(manifest, distributed, level):
    """The value of RUNAI_STREAMER_MEMORY_LIMIT for a row, or None to leave it unset.

    A distributed load divides the limit, a node total, among the ranks, so a rank gets limit / WORLD, and no rank
    gets less than the largest tensor: the limits are given for the ranks of the load.
    tight: the largest tensor of the model, for each rank. (A file's header is read as one range too, so it must not be
    larger than the largest tensor, which gen_models.py sees to: a model that breaks that is refused here.)
    roomy: a quarter of the model, but not less than twice the tight limit. Refused where that is not below the model,
    because a limit that the model fits in reuses nothing."""
    if level == "default":
        return None
    ranks = WORLD if distributed else 1
    total = manifest["total_data_bytes"]
    largest_tensor = manifest["largest_tensor_bytes"]
    largest_header = max(f["header_len"] for f in manifest["files"])
    if largest_header > largest_tensor:
        raise ValueError("the largest header of the model, %d bytes, is larger than its largest tensor, %d bytes: the "
                         "streamer reads a header as one range, so the limit would have to be sized by it" % (
                             largest_header, largest_tensor))
    tight = max(largest_tensor * ranks, MIB)
    if level == "tight":
        return tight
    if level == "roomy":
        roomy = max(total // 4, 2 * tight)
        if roomy >= total:
            raise ValueError("a roomy limit of %d bytes is not below the model, %d bytes: nothing would be reused" % (
                roomy, total))
        return roomy
    raise ValueError("unknown limit %r" % (level,))


def is_valid(model, dist, policy, owned, device, limit):
    if dist and device != "cuda":
        return False  # distributed streaming needs a CUDA destination
    if policy == "files" and model == "many_tensors":
        return False  # one file: there is nothing to give each rank
    if limit == "roomy" and model not in ROOMY_MODELS:
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
    "mixed.solo.owned.cuda.roomy",
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


def gpu_world(text=None):
    """The ranks of a run, from GPU_WORLD: WORLD (two GPUs, every row, the default and what CI runs) or 1 (one GPU: the
    rows that load without a second rank, see rows_for)."""
    if text is None:
        text = os.environ.get("GPU_WORLD")
    if text in (None, ""):
        return WORLD
    if text not in ("1", str(WORLD)):
        raise ValueError("GPU_WORLD must be 1 or %d, not %r" % (WORLD, text))
    return int(text)


def rows_for(mode, world=WORLD):
    """The rows of a mode. With one rank (world 1) the distributed rows are left out: they stream over NCCL between
    two ranks on two GPUs, and two ranks cannot share one GPU. See skipped_rows."""
    if mode == "core":
        rows = core_rows()
    elif mode == "full":
        rows = full_rows()
    elif mode == "smoke":
        rows = smoke_rows()
    else:
        raise ValueError("GPU_MATRIX must be core, full or smoke, not %r" % (mode,))
    return rows if world >= WORLD else [row for row in rows if not row.dist]


def skipped_rows(mode, world=WORLD):
    """The rows of a mode that a run of this many ranks leaves out."""
    return [row for row in rows_for(mode) if row not in rows_for(mode, world)]


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


def unmatched_prefixes(rows, text):
    """The prefixes in text that no row of rows starts with. Another prefix may still select rows, so the run does not
    look empty, and a typo in one prefix would go unseen."""
    return [prefix for prefix in row_prefixes(text) if not any(row.id.startswith(prefix) for row in rows)]
