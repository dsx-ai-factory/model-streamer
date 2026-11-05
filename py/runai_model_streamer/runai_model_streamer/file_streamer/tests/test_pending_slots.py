# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from runai_model_streamer.file_streamer.slot_pool import Slot, SlotPool


class FakeEvent:
    """What a slot's release event has to be: something that can be asked whether it is done, and
    waited on. The real one is a torch CUDA event; nothing here needs a device."""

    def __init__(self) -> None:
        self.done = False
        self.waits = 0

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.waits += 1
        self.done = True


def pool(depth: int = 2, owned: bool = False) -> SlotPool:
    return SlotPool(depth, lambda: Slot(bytearray(8), 0xF00D), owned=owned)


class TestPendingSlots(unittest.TestCase):
    def test_a_slot_released_with_an_event_is_not_free_until_it_completes(self):
        p = pool(1)
        event = FakeEvent()
        p.release(p.take(), event)
        self.assertFalse(p.has_free(), "the copy out of the slot is still running")
        event.done = True
        self.assertTrue(p.has_free())

    def test_without_an_event_a_slot_is_free_at_once(self):
        p = pool(1)
        p.release(p.take())
        self.assertTrue(p.has_free())

    def test_a_pending_slot_is_never_handed_out(self):
        p = pool(1)
        index = p.take()
        p.release(index, FakeEvent())
        with self.assertRaises(IndexError):
            p.take()          # the free list is empty while the slot is pending

    def test_releasing_a_pending_slot_twice_is_refused(self):
        p = pool(1)
        index = p.take()
        p.release(index, FakeEvent())
        with self.assertRaises(ValueError) as raised:
            p.release(index, FakeEvent())
        self.assertIn("already free", str(raised.exception))

    def test_an_owned_pool_refuses_an_event(self):
        """A contradiction, not a configuration: an owned slot is handed away and the next take()
        allocates a fresh one, so there is no reuse for an event to protect."""
        p = pool(1, owned=True)
        with self.assertRaises(ValueError) as raised:
            p.release(p.take(), FakeEvent())
        self.assertIn("handed away, not reused", str(raised.exception))

    def test_an_owned_pool_still_releases_without_one(self):
        p = pool(1, owned=True)
        p.release(p.take())
        self.assertTrue(p.has_free())

    def test_the_sweep_stops_at_the_first_event_still_running(self):
        """Events are recorded in release order on one stream, so a later one cannot complete before
        an earlier one. Stopping early keeps the sweep O(freed) rather than O(pending)."""
        p = pool(2)
        first, second = FakeEvent(), FakeEvent()
        a, b = p.take(), p.take()
        p.release(a, first)
        p.release(b, second)
        second.done = True            # out of order, which the stream cannot actually produce
        self.assertFalse(p.has_free(), "the sweep must not look past the oldest pending slot")
        first.done = True
        self.assertTrue(p.has_free())

    def test_wait_for_free_blocks_on_the_oldest_event_and_frees_it(self):
        p = pool(1)
        event = FakeEvent()
        p.release(p.take(), event)
        p.wait_for_free()
        self.assertEqual(event.waits, 1)
        self.assertTrue(p.has_free())

    def test_wait_for_free_does_nothing_when_a_slot_is_already_free(self):
        p = pool(1)
        event = FakeEvent()
        p.release(p.take(), event)
        event.done = True
        p.wait_for_free()
        self.assertEqual(event.waits, 0, "nothing to wait for - the sweep already freed it")

    def test_waiting_with_every_slot_held_is_an_error_not_a_sleep(self):
        p = pool(1)
        p.take()                       # taken and never released
        with self.assertRaises(RuntimeError) as raised:
            p.wait_for_free()
        self.assertIn("held by a live request", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
