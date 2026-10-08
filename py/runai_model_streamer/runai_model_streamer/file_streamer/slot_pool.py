from __future__ import annotations
from typing import Any, Callable, List, Optional
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
    """

    def __init__(self, depth: int, alloc: Callable[[], Slot], owned: bool) -> None:
        self._alloc = alloc
        self._owned = owned
        # None between a slot being given away and taken again. Private, because that state is
        # bookkeeping and must never reach a caller - see slot().
        self._slots: List[Optional[Slot]] = [alloc() for _ in range(depth)]
        self._free = deque(range(depth))

    def __len__(self) -> int:
        return len(self._slots)

    def has_free(self) -> bool:
        return len(self._free) > 0

    def take(self) -> int:
        """The next free slot, with memory under it.

        Allocating HERE rather than in release() is what keeps a load from ending on a ring of fresh
        buffers nobody asked for: the last releases are followed by no take at all."""
        index = self._free.popleft()
        if self._slots[index] is None:
            self._slots[index] = self._alloc()
        return index

    def release(self, index: int) -> None:
        """Give a drained slot back.

        Under `owned` dropping our reference here is what lets the slot die once the caller drops its
        last view, and what makes it impossible for us to write to it again.

        A second release is rejected rather than appended: handing the same index out twice would
        put one slot under two live requests, and the C layer would write both into it."""
        if index in self._free:
            raise ValueError(
                f"ring slot {index} is already free - releasing it twice would hand the same "
                f"memory to two callers"
            )

        if self._owned:
            self._slots[index] = None
        self._free.append(index)

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
