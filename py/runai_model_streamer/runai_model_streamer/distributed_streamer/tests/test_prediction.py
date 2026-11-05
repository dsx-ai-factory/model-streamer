# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from runai_model_streamer.distributed_streamer.partition import partition_for_rank
from runai_model_streamer.distributed_streamer.prediction import (
    PredictedRange,
    predicted_slots,
)
from runai_model_streamer.file_streamer.requests_iterator import (
    DIRECT_IO_BLOCK,
    FileChunks,
    FilesRequestsIteratorWithBuffer,
    SlotMemory,
    _alloc_slot,
)

# direct_io=False is what a DEVICE ring uses, and it is the case the prediction describes: ranges
# packed one after another with no padding. A host ring aligns each range to its file offset, which
# is a different layout and not one a peer can compute.
TIGHT_SLOTS = SlotMemory(_alloc_slot, direct_io=False, label="TEST",
                         new_event=None, current_stream=None)


def what_the_ring_does(files, originals, buffer_size, num_buffers=2):
    """The ring's own answer: per slot, the ranges it places and where, relative to the slot base.

    Reads the ring rather than the streamer, so the comparison is against the code that decides the
    layout and not against a second description of it.

    `originals` renumbers back to the CALLER's file ids, with the same expression prefill() uses -
    the partition numbers each rank's share from zero, and the consumer asked in its own terms."""
    ring = FilesRequestsIteratorWithBuffer(
        buffer_size, num_buffers, files, DIRECT_IO_BLOCK, owned=False, slots=TIGHT_SLOTS,
    )
    slots = []
    while True:
        request = ring.next_request()
        if request is None:
            return slots

        base = ring._pool.slot(request.buffer_index).address
        slot = []
        for file_index, file_chunks in enumerate(request.files):
            for range_index, size in enumerate(file_chunks.sizes):
                flat = request.flat_index(file_index, range_index)
                original_file, original_range, _ = originals[file_chunks.id][
                    request.global_range_index(file_index, range_index)
                ]
                slot.append((original_file, original_range, size,
                             request.range_dsts[flat] - base))
        slots.append(slot)
        # Freed at once: this walks the layout, it never reads a byte, so there is nothing to wait
        # for and a ring of two buffers would otherwise stop after two requests.
        ring.release(request)


class TestPredictionMatchesTheRing(unittest.TestCase):
    """The prediction has to agree with the ring exactly, because it is about to replace a message
    that carried the same facts. A disagreement is not a slow load - it is a broadcast of the wrong
    size, which hangs or corrupts."""

    def files(self, count, ranges_per_file, size):
        return [FileChunks.contiguous(i, f"f{i}.bin", 0, [size] * ranges_per_file)
                for i in range(count)]

    def assert_agrees(self, files, group_size, buffer_size):
        for rank in range(group_size):
            span = partition_for_rank(files, group_size, rank)
            mine = [file_chunks for file_chunks, _ in span.partition]
            originals = {fc.id: mapping for fc, mapping in span.partition}
            expected = what_the_ring_does(mine, originals, buffer_size)

            predicted = [
                [(r.file_id, r.range_index, r.size, r.offset) for r in slot]
                for slot in predicted_slots(files, group_size, rank, buffer_size)
            ]

            # The SLOT BOUNDARIES as well as the contents: two layouts can hold the same ranges in
            # the same order and still disagree about where one slot ends, which is exactly the
            # disagreement that breaks a broadcast size.
            self.assertEqual([len(s) for s in predicted], [len(s) for s in expected],
                             f"rank {rank}: slot boundaries differ")
            self.assertEqual(predicted, expected, f"rank {rank}: layout differs")
        return expected

    def test_one_rank_one_slot(self):
        files = self.files(count=1, ranges_per_file=4, size=100)
        slots = self.assert_agrees(files, group_size=1, buffer_size=4096)
        self.assertEqual(len(slots), 1, "the setup fitted everything in one slot")

    def test_the_ranges_are_packed_with_no_gaps(self):
        files = self.files(count=2, ranges_per_file=3, size=100)
        for slot in predicted_slots(files, 1, 0, 4096):
            offsets = [r.offset for r in slot]
            self.assertEqual(offsets, [100 * i for i in range(len(slot))],
                             "a device ring packs tight, so an offset is the sum of what precedes it")

    def test_a_slot_that_fills_up_splits_the_same_way(self):
        # 10 ranges of 100 against a 450 byte slot: 4 per slot, then 2. A boundary the packer has to
        # find, not one the sizes make obvious.
        files = self.files(count=1, ranges_per_file=10, size=100)
        slots = self.assert_agrees(files, group_size=1, buffer_size=450)
        self.assertEqual([len(s) for s in slots], [4, 4, 2], "the setup did not exercise a split")

    def test_a_budget_one_byte_from_a_boundary(self):
        """The boundary itself, not just the contents.

        10 ranges of 100 against 399 bytes fits THREE per slot - one more byte would fit four. A
        slot size that is a round multiple hides this: every case with room to spare agrees with a
        packer that is slightly too generous, and a wrong boundary is the failure that matters,
        because it changes the size of the broadcast."""
        files = self.files(count=1, ranges_per_file=10, size=100)
        slots = self.assert_agrees(files, group_size=1, buffer_size=399)
        self.assertEqual([len(s) for s in slots], [3, 3, 3, 1],
                         "the setup is not one byte from a boundary any more")

    def test_every_rank_of_a_group_agrees(self):
        files = self.files(count=6, ranges_per_file=5, size=100)
        self.assert_agrees(files, group_size=4, buffer_size=350)

    def test_uneven_range_sizes(self):
        files = [FileChunks.contiguous(0, "a.bin", 0, [10, 300, 7, 120, 65]),
                 FileChunks.contiguous(1, "b.bin", 0, [500, 1, 44])]
        self.assert_agrees(files, group_size=2, buffer_size=512)

    def test_a_peer_is_predicted_without_touching_its_share(self):
        """The point of the whole design: rank 0 computes rank 1's layout from the file list alone."""
        files = self.files(count=4, ranges_per_file=4, size=100)
        peer = list(predicted_slots(files, 2, 1, 700))

        span = partition_for_rank(files, 2, 1)
        theirs = what_the_ring_does([fc for fc, _ in span.partition],
                                    {fc.id: m for fc, m in span.partition}, 700)

        self.assertEqual([[(r.file_id, r.range_index, r.size, r.offset) for r in s] for s in peer],
                         theirs)
        self.assertGreater(len(peer), 0, "the peer was predicted to read nothing")


if __name__ == "__main__":
    unittest.main()
