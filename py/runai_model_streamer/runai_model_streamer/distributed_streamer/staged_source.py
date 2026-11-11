# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A distribution source fed by copies out of a host ring.

Used where the C++ layer cannot read into the sending device: a gloo group on a CPU device, or any
device type the library does not serve. A host ring is read into, and each range is copied from it
into the send buffer.

The host ring's own layout never leaves this rank, so it keeps the congruent placement a direct read
needs. Only the send buffer follows the interface's layout rule.
"""

from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple

from runai_model_streamer.distributed_streamer.distribution_source import (
    Filled,
    SlotRange,
    next_offset,
)


class StagedDistributionSource:
    """Copies ranges from a host ring into one send buffer.

    ONE buffer, so one slot is out at a time. The interface allows a deeper source and this is not
    one: next_slot() refuses while a slot is out rather than overwriting it.
    """

    DEPTH = 1

    def __init__(
        self,
        ranges: Iterator[Tuple[int, int, Any]],
        buffer: Any,
        originals: Dict[int, Dict[int, Tuple[int, int, int]]],
        max_ranges_per_slot: int,
    ) -> None:
        """ranges yields (file_id, range_index, memory) in this rank's own numbering.

        originals converts that numbering to the caller's, as rank_dicts_map does.
        """
        self._ranges = ranges
        self._buffer = buffer
        self._originals = originals
        self._max_ranges_per_slot = max_ranges_per_slot

        if max_ranges_per_slot < 1:
            raise ValueError(
                f"a slot must hold at least one range, got {max_ranges_per_slot}"
            )

        self._leftover: Optional[Tuple[int, int, Any]] = None
        self._exhausted = False
        self._out = False

    def next_slot(self) -> Optional[Filled]:
        if self._out:
            raise RuntimeError(
                "next_slot() while a slot is out; this source has one buffer, so the slot must be "
                "released first"
            )
        if self._exhausted and self._leftover is None:
            return None

        placed: List[SlotRange] = []
        cursor = 0
        capacity = self._buffer.numel()

        while len(placed) < self._max_ranges_per_slot:
            item = self._leftover if self._leftover is not None else next(self._ranges, None)
            self._leftover = None

            if item is None:
                self._exhausted = True
                break

            file_id, range_index, memory = item
            size = memory.numel()
            offset = next_offset(cursor)

            if offset + size > capacity:
                self._leftover = item
                break

            self._buffer[offset : offset + size].copy_(memory.squeeze())

            caller_file, caller_range, _ = self._originals[file_id][range_index]
            placed.append(SlotRange(caller_file, caller_range, offset, size))
            cursor = offset + size

        if not placed:
            if self._leftover is not None:
                # None would read as end of stream, and the loop would broadcast empty batches until
                # it failed somewhere else. The caller sizes the buffer to the largest range.
                file_id, range_index, memory = self._leftover
                raise ValueError(
                    f"range {range_index} of file {file_id} is {memory.numel()} bytes and does not "
                    f"fit in a {capacity}-byte send buffer"
                )
            return None

        self._out = True
        return Filled(memory=self._buffer[:cursor], ranges=placed)

    def release(self, filled: Filled, done: Any = None) -> None:
        """Take the buffer back.

        `done` is ignored. The copies into this buffer run on the caller's stream, and so does
        anything the caller enqueued against the slot, so the next fill is already ordered after it.
        """
        self._out = False
