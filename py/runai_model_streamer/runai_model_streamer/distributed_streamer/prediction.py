# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple

from runai_model_streamer.distributed_streamer.distribution_source import (
    SlotRange,
    advance,
)
from runai_model_streamer.distributed_streamer.partition import partition_for_rank
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    FilesRequestsIterator,
)

__all__ = ["predicted_slots"]


def predicted_slots(
    file_stream_requests: List[FileChunks],
    group_size: int,
    rank: int,
    buffer_size: int,
    max_ranges_per_slot: Optional[int] = None,
    span_partition: Optional[List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]] = None,
) -> Iterator[List[SlotRange]]:
    """What `rank` will hold in each of its slots, in the order it fills them.

    Every term is a pure function of the file list, the group size and the slot shape, so a rank
    computes a PEER's layout by passing the peer's rank. Nothing is exchanged.

    span_partition is this rank's share when the caller already has it. Working it out here reads
    every tensor of the whole model, and a caller that predicts for each rank in turn would pay that
    pass once per rank for the same answer.
    """
    if span_partition is None:
        span_partition = partition_for_rank(file_stream_requests, group_size, rank).partition

    files = [file_chunks for file_chunks, _ in span_partition]

    # The partition renumbers the ranges it hands out, and the consumer asked in its own terms.
    originals: Dict[int, Dict[int, Tuple[int, int, int]]] = {
        file_chunks.id: mapping for file_chunks, mapping in span_partition
    }

    # THE RING'S OWN PACKER, not a copy of the rule it follows. A ring groups ranges into slots with
    # this object, and it allocates nothing, so a prediction built on it cannot drift from the
    # grouping a peer will really use.
    packer = FilesRequestsIterator(buffer_size, files, 1, max_ranges_per_slot)

    while True:
        request = packer.next_request()
        if request is None:
            return

        cursor = 0
        slot: List[SlotRange] = []
        for file_index, file_chunks in enumerate(request.files):
            for range_index, size in enumerate(file_chunks.sizes):
                original_file, original_range, _ = originals[file_chunks.id][
                    request.global_range_index(file_index, range_index)
                ]

                # The layout rule, shared with the source that will send these slots. A second copy
                # of it here would disagree in silence, and a receiver would read the wrong bytes.
                at, cursor = advance(cursor, size)
                slot.append(SlotRange(original_file, original_range, at, size))

        yield slot
