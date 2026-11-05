# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Dict, Iterator, List, NamedTuple, Tuple

from runai_model_streamer.distributed_streamer.partition import partition_for_rank
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    FilesRequestsIterator,
)


class PredictedRange(NamedTuple):
    """One range as a peer will lay it out, in the caller's own numbering."""

    file_id: int
    range_index: int
    size: int
    offset: int        # from the start of the slot


def predicted_slots(
    file_stream_requests: List[FileChunks],
    group_size: int,
    rank: int,
    buffer_size: int,
) -> Iterator[List[PredictedRange]]:
    """What `rank` will hold in each of its slots, in the order it fills them.

    This is what replaces the metadata broadcast. Every term is a pure function of the file list, the
    group size and the slot size, so a rank computes a PEER's layout by passing the peer's rank -
    nothing is exchanged.

    THE PRODUCTION PACKER RUNS HERE, not a copy of it. FilesRequestsIterator is the object the ring
    itself packs with, and it allocates nothing, so a prediction cannot drift from what the ring
    will do. Reimplementing the rule would be the one way to make this design silently wrong.

    Offsets are a plain running cursor because a device ring places ranges with _place(aligned=False),
    which is a plain running cursor from the slot base. No address enters the calculation, which is
    what makes the answer the same on every rank."""
    span = partition_for_rank(file_stream_requests, group_size, rank)

    files = [file_chunks for file_chunks, _ in span.partition]
    # Range index within a partitioned file -> that range's place in the CALLER's numbering. The
    # partition renumbers, and the consumer asked in its own terms.
    originals: Dict[int, Dict[int, Tuple[int, int, int]]] = {
        file_chunks.id: mapping for file_chunks, mapping in span.partition
    }

    packer = FilesRequestsIterator(buffer_size, files)

    while True:
        request = packer.next_request()
        if request is None:
            return

        offset = 0
        slot: List[PredictedRange] = []
        for file_index, file_chunks in enumerate(request.files):
            for range_index, size in enumerate(file_chunks.sizes):
                original_file, original_range, _ = originals[file_chunks.id][
                    request.global_range_index(file_index, range_index)
                ]
                slot.append(PredictedRange(original_file, original_range, size, offset))
                offset += size

        yield slot
