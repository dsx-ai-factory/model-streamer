# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A distribution source whose ring IS the send buffer.

Used where the C++ layer can write into the sending device. The library reads straight into the ring,
and the slot is broadcast as it lies - there is no copy between reading and sending.

This source packs nothing. The ring placed the ranges when it built the request, so all this does is
describe them: the offsets come from the ring, and the ids are converted to the caller's numbering.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from runai_model_streamer.distributed_streamer.distribution_source import (
    RANGE_OFFSET_ALIGNMENT,
    Filled,
    SlotRange,
)


class DirectDistributionSource:
    """Hands over whole ring slots, in the order they were submitted.

    ONE SLOT AT A TIME. get_slots parks at a yield holding the slot, so there is never a second one
    to give out; next_slot() refuses rather than overwriting. The overlap still happens, in the ring
    behind it: releasing a slot lets the library refill it while the collective for the previous one
    is still running.

    RELEASE DOES THE WORK. Resuming the generator is what hands the event to the ring, so release()
    resumes it there and then and keeps whatever comes back for the next next_slot(). Deferring that
    resume would strand the LAST slot's event: a loop that stops counting when the stream is done
    never calls next_slot() again, the streamer never learns the event, and its teardown falls back
    to waiting on a stream the collective was not recorded on - freeing the ring while a broadcast
    is still enqueued against it.
    """

    DEPTH = 1

    def __init__(
        self,
        file_streamer: Any,
        originals: Dict[int, Dict[int, Tuple[int, int, int]]],
    ) -> None:
        """originals converts this rank's numbering to the caller's, as rank_dicts_map does."""
        ring = file_streamer.requests_iterator
        if ring is None:
            raise ValueError("the streamer has no ring; stream_files has not run")

        # CHECKED, not assumed. A ring placed by another rule - a host ring is placed congruent with
        # its file offsets - puts its ranges where a peer computing the layout will not look.
        if ring.range_alignment != RANGE_OFFSET_ALIGNMENT:
            raise ValueError(
                f"this ring places its ranges by {ring.range_alignment} and a sent slot must use "
                f"{RANGE_OFFSET_ALIGNMENT}, or a peer reads it at the wrong offsets"
            )

        # HELD, not re-read. stream_files builds a new ring, and asking the streamer later would
        # subtract a new pool's address from this request's destinations - wrong offsets, no error.
        self._ring = ring
        self._slots = file_streamer.get_slots()
        self._originals = originals

        self._ready: Optional[Filled] = None
        self._out: Optional[Filled] = None
        self._ended = False

    def next_slot(self) -> Optional[Filled]:
        if self._out is not None:
            raise RuntimeError(
                "next_slot() while a slot is out; this source holds one slot, so it must be "
                "released first"
            )
        if self._ended:
            return None

        if self._ready is None:
            # Only the first call gets here: every later one was served by a release.
            self._ready = self._resume(None)
            if self._ready is None:
                return None

        filled, self._ready = self._ready, None
        self._out = filled
        return filled

    def release(self, filled: Filled, done: Any = None) -> None:
        if self._out is None:
            raise RuntimeError("release() with no slot out; a slot is released once")
        if filled is not self._out:
            # Letting it through would resume the ring for a slot the caller is not finished with,
            # and hand the event meant for an older slot to the one being refilled.
            raise RuntimeError("release() of a slot this source did not just hand out")

        self._out = None
        self._ready = self._resume(done)

    def _resume(self, event: Any) -> Optional[Filled]:
        """Give the generator the event and take whatever it offers next."""
        try:
            request, memory = self._slots.send(event)
        except StopIteration:
            self._ended = True
            return None

        return Filled(memory=memory, ranges=self._describe(request))

    def _describe(self, request: Any) -> List[SlotRange]:
        """The ranges in this slot, in the caller's numbering and in layout order."""
        offsets = self._ring.range_offsets_of(request)

        ranges = []
        for file_index, file_chunks in enumerate(request.files):
            for local_index, size in enumerate(file_chunks.sizes):
                flat = request.flat_index(file_index, local_index)
                mine = request.global_range_index(file_index, local_index)
                caller_file, caller_range, _ = self._originals[file_chunks.id][mine]
                ranges.append(SlotRange(caller_file, caller_range, offsets[flat], size))

        return ranges
