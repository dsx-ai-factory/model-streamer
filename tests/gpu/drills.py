# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault injection, to show that the checks of the GPU test catch what they are for. No torch, so it is unit tested on
the CPU (test_drills.py).

GPU_DRILL=corrupt:ROW,leak:ROW makes the test break that row on purpose; the run then passes only if the break is
reported, and fails if it goes unnoticed.

    corrupt  flips one bit of the first tensor of the row after it arrived: the CRC32 check must report that tensor
    leak     keeps a 32 MiB GPU tensor alive after the row: the GPU memory check must report it

A rank that fails inside a distributed row is not a drill: the NCCL watchdog aborts the other rank, so there is nothing
left to report it.
"""

MIB = 1 << 20
LEAK_DRILL_BYTES = 32 * MIB
KINDS = ("corrupt", "leak")


def parse_drills(text):
    """{row id: kind} from "KIND:ROW_ID,KIND:ROW_ID"."""
    drills = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        kind, _, row_id = item.partition(":")
        if kind not in KINDS or not row_id:
            raise RuntimeError("GPU_DRILL: %r is not KIND:ROW_ID with KIND one of %s" % (item, sorted(KINDS)))
        drills[row_id] = kind
    return drills


def drill_caught(kind, errors, corrupted_tensor=None, leak_bytes=0):
    """Whether the errors of a drilled row report the fault that was injected, and not some other problem of the row.

    corrupt: an error of the CRC32 check that names the tensor that was corrupted (an error for another tensor is a
    real failure, not this one).
    leak: the GPU memory error, with at least as many bytes still allocated as the drill leaked (leak_bytes is the
    measured excess over the baseline), so that a drill that did nothing cannot be satisfied by a small real leak."""
    if kind == "corrupt":
        return corrupted_tensor is not None and any(
            error.startswith(corrupted_tensor + " (") and "crc32" in error for error in errors)
    if kind == "leak":
        return leak_bytes >= LEAK_DRILL_BYTES and any(error.startswith("GPU memory leak") for error in errors)
    raise ValueError("unknown drill %r" % (kind,))


def _is_the_report(kind, error, corrupted_tensor):
    if kind == "corrupt":
        return corrupted_tensor is not None and error.startswith(corrupted_tensor + " (") and "crc32" in error
    if kind == "leak":
        return error.startswith("GPU memory leak")
    raise ValueError("unknown drill %r" % (kind,))


def unexpected_errors(kind, errors, corrupted_tensor=None, error_count=None):
    """The errors of a drilled row that are not the report of its fault: the row has another problem, which a drill
    must not hide. error_count is how many errors the row had in all; the errors beyond those recorded cannot be told
    apart, and count as unexpected."""
    unexpected = [error for error in errors if not _is_the_report(kind, error, corrupted_tensor)]
    if error_count is not None and error_count > len(errors):
        unexpected.append("... and %d more error(s) that were not recorded" % (error_count - len(errors)))
    return unexpected
