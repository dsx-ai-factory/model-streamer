import unittest
import numpy as np
from runai_model_streamer.file_streamer.slot_pool import Slot, SlotPool

SLOT_SIZE = 8


class CountingAllocator:
    """A slot allocator that remembers how often it was asked.

    The count is what makes lazy allocation testable: "no memory is taken at release" cannot be read
    off the slot list, which only says a slot is absent, not whether something was built for it.
    """

    def __init__(self, size: int = SLOT_SIZE) -> None:
        self.size = size
        self.calls = 0

    def __call__(self) -> Slot:
        self.calls += 1
        memory = np.zeros(self.size, dtype=np.uint8)
        return Slot(memory, memory.ctypes.data)


class TestSlotPool(unittest.TestCase):
    """Who owns a taken slot's memory, and what that does to the ring's depth."""

    def test_a_released_slot_is_reused_when_not_owned(self):
        # The baseline owning is measured against: the same memory comes back, which is why a caller
        # may not hold a view past the release.
        pool = SlotPool(2, CountingAllocator(), owned=False)
        index = pool.take()
        address = pool.slot(index).address

        pool.release(index)

        self.assertEqual(pool.slot(index).address, address)

    def test_the_reusable_ring_never_holds_an_empty_slot(self):
        # Giving a slot away is the owned path only. Without it the ring must stay exactly as it was -
        # every slot backed at every point in a cycle, so nothing downstream can meet a None.
        pool = SlotPool(2, CountingAllocator(), owned=False)

        # `is None` per slot, not `assertNotIn`: `None in [ndarray, ...]` falls through to `==`, which
        # numpy answers with an array and Python cannot reduce to a truth value.
        def backed():
            return all(slot is not None for slot in pool._slots)

        for _ in range(4):
            self.assertTrue(backed())
            index = pool.take()
            self.assertTrue(backed())
            pool.release(index)
        self.assertTrue(backed())

    def test_a_given_away_slot_is_refused_rather_than_returned(self):
        """The invariant stated as an error, not as a comment.

        A caller only ever holds an index take() handed it, and that allocates first - so this cannot
        happen through the API. It is asserted because the failure without it is a numpy error deep
        inside a view, a long way from whatever handed out a stale index."""
        pool = SlotPool(2, CountingAllocator(), owned=True)
        index = pool.take()
        pool.release(index)

        with self.assertRaises(RuntimeError) as raised:
            pool.slot(index)
        self.assertIn("given away", str(raised.exception))

    def test_an_owned_slot_is_given_away_not_recycled(self):
        # DEPTH 1, so the slot that comes back is the one that went away. With a deeper pool the free
        # list is FIFO and the next take returns a different index, which would leave this asserting
        # nothing about reuse.
        alloc = CountingAllocator()
        pool = SlotPool(1, alloc, owned=True)
        self.assertEqual(alloc.calls, 1)

        index = pool.take()
        given_away = pool.slot(index).memory
        pool.release(index)

        # Nothing allocated yet - the replacement is built when the slot is next taken, because the
        # last releases of a load are followed by no take at all.
        self.assertEqual(alloc.calls, 1)
        self.assertIsNone(pool._slots[index])

        self.assertEqual(pool.take(), index, "the fixture must reuse the same index")
        self.assertEqual(alloc.calls, 2)

        # A DIFFERENT ALLOCATION, checked by identity rather than by address. Nothing holds the old
        # array here, so malloc is free to hand the same address straight back - and does, which is
        # what made an address comparison pass alone and fail in the full suite.
        self.assertIsNot(pool.slot(index).memory, given_away)

    def test_an_owned_slot_keeps_its_bytes_after_release(self):
        # THE GUARANTEE ITSELF, and the reason an address check alone is not enough: a caller that
        # keeps a yielded tensor must still read its own bytes after the pool has moved on.
        pool = SlotPool(2, CountingAllocator(), owned=True)
        index = pool.take()
        pool.slot(index).memory[:4] = [1, 2, 3, 4]
        held = pool.slot(index).memory[:4]          # what the caller is handed

        pool.release(index)
        pool.take()

        # EVERY slot the pool still owns, not just the one that came back: the free list is FIFO, so
        # which slot is handed out next is incidental.
        for slot in pool._slots:
            if slot is not None:
                slot.memory[:4] = [9, 9, 9, 9]

        self.assertEqual(list(held), [1, 2, 3, 4])

    def test_the_depth_is_unchanged_by_owning(self):
        # Backpressure is the free list, and giving a slot away must not add or lose one - otherwise
        # the reader would run further ahead than the memory limit allows.
        pool = SlotPool(2, CountingAllocator(), owned=True)
        for _ in range(3):
            index = pool.take()
            self.assertTrue(pool.has_free())
            pool.release(index)
            self.assertEqual(len(pool), 2)


if __name__ == "__main__":
    unittest.main()
