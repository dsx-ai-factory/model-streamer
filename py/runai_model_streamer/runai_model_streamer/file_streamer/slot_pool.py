# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any, Callable, Deque, List, Optional, Tuple
from collections import deque


class Slot:
    """One ring slot: its memory, and the address the C layer was given for it.

    ONE OBJECT rather than two parallel lists. The pairing is made once, here, so the pool holds
    slots and has nothing that can drift. Two lists would, and a desync would hand the C layer an
    address into memory we no longer hold - which reads as corrupt weights, not as an error.

    The address is PASSED IN rather than derived on demand. It is asked for once per RANGE, and
    numpy builds a fresh ctypes object on every `.ctypes` access, which is an order of magnitude
    dearer than reading an attribute.
    """

    __slots__ = ("memory", "address")

    def __init__(self, memory: Any, address: int) -> None:
        self.memory = memory
        self.address = address


class SlotPool:
    """A fixed number of buffers, taken one at a time and given back once drained.

    WHO OWNS A TAKEN SLOT'S MEMORY is the pool's policy, not its caller's. Under `owned` a released
    slot is dropped and a fresh one takes its place, so memory already handed out is never written to
    again. Without it the same slot comes straight back, and the caller must not touch it after the
    release.

    The index stays in the free list either way, so the DEPTH - and the backpressure that depth
    provides - does not depend on the policy. Only the memory's identity changes.

    The allocator is injected because what a slot is made of belongs to the path using the pool: host
    memory for the CPU ring, device memory for the GPU one.

    A released slot may be PENDING rather than free - see release(). The event is only ever asked
    .query() and .synchronize(), so this module stays free of any device library.
    """

    def __init__(self, depth: int, alloc: Callable[[], Slot], owned: bool) -> None:
        self._alloc = alloc
        self._owned = owned
        # None between a slot being given away and taken again. Private, because that state is
        # bookkeeping and must never reach a caller - see slot().
        self._slots: List[Optional[Slot]] = [alloc() for _ in range(depth)]
        self._free = deque(range(depth))
        # (index, event) for slots whose last reader has not finished. In release order, so the
        # oldest - the one most likely to be done - is swept and waited on first.
        self._pending: Deque[Tuple[int, Any]] = deque()

    def __len__(self) -> int:
        return len(self._slots)

    def has_free(self) -> bool:
        """Whether a slot can be taken now.

        SWEEPS FIRST, so this is not a pure query: a pending slot whose event has completed becomes
        free here. That makes every "can I take one?" the poll point, which is where the answer is
        wanted - nothing else has a reason to ask."""
        self._sweep_pending()
        return len(self._free) > 0

    def _sweep_pending(self) -> None:
        """Free every pending slot whose event has completed. Non-blocking.

        Oldest first, and stops at the first one still running: the events were recorded in release
        order on ONE stream, so a later one cannot complete before an earlier one. That is an
        assumption about the caller, not something enforced here - a caller that recorded them on
        different streams would only see a completed slot freed late, never a pending one freed
        early."""
        while self._pending and self._pending[0][1].query():
            index, _ = self._pending.popleft()
            self._free.append(index)

    def wait_for_free(self) -> None:
        """Block until a pending slot is free. Only for when nothing is in flight and nothing is
        free: the caller would otherwise go back to waiting for a response that can never arrive,
        because it has nothing left to submit.

        Does nothing when a slot is already free, and raises when there is nothing to wait for -
        that state is a caller that lost track of its own slots, not a condition to sleep through."""
        if self.has_free():
            return
        if not self._pending:
            raise RuntimeError(
                "no ring slot is free and none is pending - every slot is held by a live request"
            )
        self._pending[0][1].synchronize()
        self._sweep_pending()

    def take(self) -> int:
        """The next free slot, with memory under it.

        Allocating HERE rather than in release() is what keeps a load from ending on a ring of fresh
        buffers nobody asked for: the last releases are followed by no take at all."""
        index = self._free.popleft()
        if self._slots[index] is None:
            self._slots[index] = self._alloc()
        return index

    def release(self, index: int, ready_event: Optional[Any] = None) -> None:
        """Give a drained slot back.

        Under `owned` dropping our reference here is what lets the slot die once the caller drops its
        last view, and what makes it impossible for us to write to it again.

        WITH ready_event the slot becomes PENDING rather than free, and returns to the free list only
        once that event has completed. A host slot needs none: the copy out of it finishes before
        tensor.to() returns. A device slot does: the clone out of it is only ENQUEUED, and the next
        writer is the C++ layer's own stream, which torch orders against nothing.

        A second release is rejected rather than appended: handing the same index out twice would
        put one slot under two live requests, and the C layer would write both into it."""
        if index in self._free or any(pending == index for pending, _ in self._pending):
            raise ValueError(
                f"ring slot {index} is already free - releasing it twice would hand the same "
                f"memory to two callers"
            )

        # A contradiction rather than a configuration: under `owned` the slot is given away and the
        # next take() allocates a fresh one, so there is no reuse for an event to protect. Waiting on
        # one would delay a slot that no longer exists.
        if self._owned and ready_event is not None:
            raise ValueError(
                f"ring slot {index} was released with an event by an owned pool - owned slots are "
                f"handed away, not reused, so nothing waits to write to them"
            )

        if self._owned:
            self._slots[index] = None

        if ready_event is None:
            self._free.append(index)
        else:
            self._pending.append((index, ready_event))

    def slot(self, index: int) -> Slot:
        """A TAKEN slot.

        A slot is None only between being given away and being taken again, and take() allocates
        before it hands an index back - so every index a live caller holds has one. Raised rather
        than assumed, because the alternative is an error deep inside a view, a long way from
        whatever handed out a stale index."""
        slot = self._slots[index]
        if slot is None:
            raise RuntimeError(
                f"ring slot {index} was given away and has not been taken again - a caller is "
                f"holding an index it does not own"
            )
        return slot
