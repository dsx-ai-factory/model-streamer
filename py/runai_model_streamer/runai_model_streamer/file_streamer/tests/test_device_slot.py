# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import unittest.mock

import torch

from runai_model_streamer.file_streamer.file_streamer import (
    DEVICE_SLOT_ALIGNMENT,
    alloc_device_slot,
    device_slots,
)
from runai_model_streamer.file_streamer.requests_iterator import (
    _alloc_slot,
    FileChunks,
    FilesRequestsIteratorWithBuffer,
    HOST_SLOTS,
    RunaiStreamerMemoryLimitException,
)

# A device ring needs SOME event factory, and these tests never release a slot with an event, so none
# is ever built. Named rather than written inline, so a reader does not have to prove that.
UNUSED_EVENT = object


class TestAllocDeviceSlot(unittest.TestCase):
    """Runs on the host: torch.empty takes "cpu" like any other device, so the arithmetic is checked
    without a GPU. The CUDA case is a separate, skipped test."""

    def test_the_slot_is_exactly_the_size_asked_for(self):
        slot = alloc_device_slot(4096, "cpu")
        self.assertEqual(slot.memory.numel(), 4096)

    def test_the_address_is_aligned_and_is_the_buffers_own(self):
        for size in (1, 4096, 4097, 1 << 20):
            slot = alloc_device_slot(size, "cpu")
            self.assertEqual(slot.address % DEVICE_SLOT_ALIGNMENT, 0, f"size {size} is not aligned")
            self.assertEqual(slot.address, slot.memory.data_ptr(),
                             "the address handed to C must be the buffer's own, not its base's")

    def test_the_slice_keeps_its_base_alive(self):
        """The over-allocated base is dropped on return, so only the view holds the memory."""
        slot = alloc_device_slot(1 << 20, "cpu")
        self.assertIsNotNone(slot.memory.untyped_storage())
        self.assertGreaterEqual(slot.memory.untyped_storage().nbytes(),
                                (1 << 20) + DEVICE_SLOT_ALIGNMENT)

    def test_a_zero_sized_slot_still_has_a_real_address(self):
        """A ring of no ranges is a real case - an empty shard, or stream_files([]). torch answers 0
        from data_ptr() on a zero-element tensor where numpy gives the true pointer, so the address
        is taken from the base; otherwise the library would be handed a null on this path alone."""
        for size in (0, 1):
            self.assertNotEqual(alloc_device_slot(size, "cpu").address, 0, f"size {size}")
        self.assertNotEqual(_alloc_slot(0).address, 0, "the host allocator must agree")

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_a_cuda_slot_lands_on_the_device(self):
        slot = alloc_device_slot(1 << 20, "cuda:0")
        self.assertTrue(slot.memory.is_cuda)
        self.assertEqual(slot.address % DEVICE_SLOT_ALIGNMENT, 0)


class TestSlotMemory(unittest.TestCase):
    """One object carries the allocator and what it implies, so the two cannot drift apart."""

    FILES = [FileChunks.contiguous(0, "a.bin", 0, [1024, 1024])]

    def ring(self, **kwargs):
        return FilesRequestsIteratorWithBuffer(
            buffer_size=4096, num_buffers=2, files_chunks=self.FILES, **kwargs
        )

    def test_a_device_ring_is_made_of_device_memory_and_reserves_no_pads(self):
        ring = self.ring(slots=device_slots("cpu", UNUSED_EVENT))
        # Pads exist for direct reads, which never land on these slots.
        self.assertEqual(ring._slot_size, 4096)
        for slot in ring._pool._slots:
            self.assertIsInstance(slot.memory, torch.Tensor)
            self.assertEqual(slot.memory.numel(), 4096)

    def test_the_host_ring_is_the_default_and_keeps_its_pads(self):
        ring = self.ring()
        self.assertGreater(ring._slot_size, 4096, "the host ring reserves room for direct-read pads")
        self.assertNotIsInstance(ring._pool._slots[0].memory, torch.Tensor)

    def test_one_fact_drives_both_the_pads_and_the_placement(self):
        """direct_io is why pads are reserved AND why ranges are placed congruently, so a ring can
        never reserve pad room it then declines to use, or vice versa."""
        device = self.ring(slots=device_slots("cpu", UNUSED_EVENT))
        host = self.ring()
        self.assertFalse(device._slot_memory.direct_io)
        self.assertTrue(host._slot_memory.direct_io)

    def test_a_device_ring_refuses_a_release_without_an_event(self):
        """The clone out of a device slot is only enqueued, so releasing it as free would let the
        next write land in memory the clone is still reading."""
        ring = self.ring(slots=device_slots("cpu", UNUSED_EVENT))
        request = ring.next_request()
        with self.assertRaises(ValueError) as raised:
            ring.release(request)
        self.assertIn("needs an event", str(raised.exception))

    def test_a_host_ring_releases_without_one(self):
        ring = self.ring()
        ring.release(ring.next_request())
        self.assertTrue(ring.has_free_buffer())

    def test_the_debug_line_names_the_memory_by_device_type(self):
        """Not a fixed "GPU": the label follows the device, so a second vendor reads correctly in the
        ring's log line without anyone editing it. torch.device parses the string without a device."""
        self.assertEqual(device_slots("cuda:0", UNUSED_EVENT).label, "CUDA")
        self.assertEqual(device_slots("cpu", UNUSED_EVENT).label, "CPU")
        self.assertEqual(HOST_SLOTS.label, "CPU")


if __name__ == "__main__":
    unittest.main()


class TestSlotGoesBackWhenRangesDoNotFit(unittest.TestCase):
    """The recoverable failure must leave the ring whole.

    When a layout does not fit, the caller is told so and invited to retry - which is only honest if
    the slot it was about to use went back to the pool. A DEVICE ring is the case that matters:
    release() refuses a slot handed back without an event, so taking that route here would both lose
    the slot and replace the message with an event-ordering error.

    _place is forced to fail rather than fed a layout that cannot fit, because the packer caps a
    request to the slot: the real trigger is a ring built from a different size than the one being
    packed against, and what is under test is the handler, not the packer."""

    FILES = [FileChunks.contiguous(0, "a.bin", 0, [1024, 1024])]

    def ring(self, **kwargs):
        return FilesRequestsIteratorWithBuffer(
            buffer_size=4096, num_buffers=2, files_chunks=self.FILES, **kwargs
        )

    def assert_slot_comes_back(self, ring):
        with unittest.mock.patch.object(ring, "_place", return_value=None):
            with self.assertRaises(RunaiStreamerMemoryLimitException):
                ring.next_request()

        # The NUMBER, not has_free(): one slot lost out of two still leaves the ring usable, so a
        # boolean would pass with the slot leaked.
        self.assertEqual(len(ring._pool._free), 2, "the slot was not handed back")

    def test_a_device_ring_keeps_its_slot_and_says_what_went_wrong(self):
        self.assert_slot_comes_back(self.ring(slots=device_slots("cpu", UNUSED_EVENT)))

    def test_a_host_ring_does_the_same(self):
        self.assert_slot_comes_back(self.ring())
