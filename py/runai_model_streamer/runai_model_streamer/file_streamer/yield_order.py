# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
from typing import Any, Dict, List, Union

from runai_model_streamer.file_streamer.requests_iterator import FileChunks


def _first_ordinal_of_each_file(files_chunks: List[FileChunks]) -> Dict[int, int]:
    """Where each file's ordinals start: the number of ranges in every earlier file.

    Shared so that the duplicate-id check happens in BOTH modes. Two files with one id would share a
    base, so their ordinals would collide and one response would overwrite the other - a tensor lost
    with nothing raised. Turning ordering off must not turn that check off with it."""
    first_ordinal: Dict[int, int] = {}
    ranges_so_far = 0
    for file_chunks in files_chunks:              # list order is the file order
        if file_chunks.id in first_ordinal:
            raise ValueError(f"file id {file_chunks.id} appears twice in one stream")
        first_ordinal[file_chunks.id] = ranges_so_far
        ranges_so_far += len(file_chunks.sizes)
    return first_ordinal


class YieldOrder:
    """Hands responses back in file order, whatever order they completed in.

    A range's ordinal is the number of ranges in every EARLIER file, plus its own index within its
    file. The order comes from the file's POSITION in the list the caller passed; the id is only how
    a response finds its file again, so the order does not depend on how the caller numbered them.

    One ordinal covers both orderings a caller expects - by file, then by offset within the file -
    and it covers them across submissions too, because next_request packs files in list order, so a
    submission always holds a continuous span of ordinals.

    A parked response is a VIEW into the ring, so waiting costs no copy. It does cost read-ahead: the
    slot behind a parked view cannot be refilled until that view has been handed over.
    """

    def __init__(self, files_chunks: List[FileChunks]) -> None:
        self._first_ordinal_of_file = _first_ordinal_of_each_file(files_chunks)
        self._next = 0

        # Keyed by ordinal, because the question asked is "has THIS one arrived", never "which is
        # smallest" - so nothing has to be kept sorted. It holds only what is parked, which the ring
        # bounds, rather than one entry per range in the stream.
        self._parked: Dict[int, Any] = {}

    def ordinal(self, file_id: int, range_index: int) -> int:
        """`range_index` counts within its own file and across submissions, which is what
        FilesRequest.global_range_index answers."""
        try:
            return self._first_ordinal_of_file[file_id] + range_index
        except KeyError:
            # Not reachable from anything a caller passes: this orderer and the requests are built
            # from the same file list, so a missing id means the streamer mapped a response to the
            # wrong file. Raised rather than left as a KeyError out of arrived() so it reads as the
            # internal fault it is. try/except because this runs on every response and costs nothing
            # until it raises.
            raise ValueError(
                f"internal error: response mapped to file id {file_id}, which is not in this stream"
            ) from None

    def arrived(self, file_id: int, range_index: int, payload: Any) -> List[Any]:
        """What may be handed over now, in order. Empty while this response still has to wait.

        The whole run is returned rather than one payload, so a single late response releases
        everything that was parked behind it."""
        self._parked[self.ordinal(file_id, range_index)] = payload

        ready = []
        while self._next in self._parked:
            ready.append(self._parked.pop(self._next))
            self._next += 1
        return ready

    def discard_parked(self) -> None:
        """Drop what is still parked, so a stream that ended early holds no ring slots.

        A parked payload is a VIEW into a slot, and a view outlives the pool it came from: dropping
        the ring frees nothing a parked view still points at. Only safe once the library has stopped
        writing into those slots, which is why the caller does this after the drain."""
        self._parked.clear()


class UnorderedYield:
    """Hands every response back as it arrives, which is what the streamer did before ordering.

    The same shape as YieldOrder, so the caller needs no condition of its own: which one is in use
    is decided once, where the stream is set up."""

    def __init__(self, files_chunks: List[FileChunks]) -> None:
        # Only for the duplicate-id check, which must not depend on which mode is in use. The map
        # itself is not needed here: nothing is ordered, so nothing needs an ordinal.
        _first_ordinal_of_each_file(files_chunks)

    def arrived(self, file_id: int, range_index: int, payload: Any) -> List[Any]:
        return [payload]

    def discard_parked(self) -> None:
        """Nothing is ever parked here, so there is nothing to drop."""


RUNAI_STREAMER_ORDERED_YIELD_ENV_VAR_NAME = "RUNAI_STREAMER_ORDERED_YIELD"
DEFAULT_ORDERED_YIELD = 1


def yield_order_for(files_chunks: List[FileChunks]) -> Union[YieldOrder, UnorderedYield]:
    """Ordered unless the caller turned it off.

    On by default because safetensors hands tensors over in file order, and a consumer may rely on
    that order or be tuned for it. Read as an int and cast, which is how the C++ layer reads its own
    flags, so 0 is the only value that turns it off."""
    value = os.environ.get(RUNAI_STREAMER_ORDERED_YIELD_ENV_VAR_NAME, str(DEFAULT_ORDERED_YIELD))
    try:
        enabled = int(value) != 0
    except ValueError:
        raise ValueError(
            f"{RUNAI_STREAMER_ORDERED_YIELD_ENV_VAR_NAME} must be an integer (0 turns ordering off), "
            f"got {value!r}"
        ) from None
    return YieldOrder(files_chunks) if enabled else UnorderedYield(files_chunks)
