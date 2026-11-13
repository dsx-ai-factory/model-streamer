# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A predicted slot is the slot the ring really builds.

This is the property the whole scheme rests on. A rank that predicts a peer's layout instead of
being told it reads the peer's bytes at the offsets computed here, so a prediction that drifts from
the ring does not fail - it returns the wrong weights."""

import os
import unittest
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.partition import partition, partition_for_rank
from runai_model_streamer.distributed_streamer.prediction import predicted_slots
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    FilesRequestsIteratorWithBuffer,
    SlotMemory,
    _alloc_slot,
)


class FakeStream:
    def synchronize(self) -> None:
        pass


class FakeEvent:
    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


SENT_SLOTS = SlotMemory(_alloc_slot, direct_io=False, label="TEST", new_event=FakeEvent,
                        current_stream=lambda device=None: FakeStream())

# Sizes that DIFFER from each other, so a layout that lost one of them still lands somewhere else.
# A ZERO is in there because the rule treats it apart - it takes no room at all - and that is the one
# case a prediction could get right by counting ranges and wrong by walking sizes.
SIZES = [100, 300, 0, 700, 250, 0, 900, 150]
POLICIES = ("spans", "chunks", "files")
BUFFER_SIZE = 2048
GROUP_SIZE = 2
CAP = 256


def files():
    """Two files, ranges laid end to end."""
    out = []
    for file_id in (0, 1):
        offsets, cursor = [], 0
        for size in SIZES:
            offsets.append(cursor)
            cursor += size
        out.append(FileChunks(file_id, f"model-{file_id}.bin", offsets, list(SIZES)))
    return out


def ring_layout(rank, alignment=1, cap=CAP):
    """What the real ring places, as (offset from slot start, size) per slot."""
    span = partition_for_rank(files(), GROUP_SIZE, rank)
    mine = [file_chunks for file_chunks, _ in span.partition]

    ring = FilesRequestsIteratorWithBuffer(
        BUFFER_SIZE, 64, mine, owned=False, slots=SENT_SLOTS, range_alignment=alignment,
        max_ranges_per_slot=cap,
    )

    slots = []
    while True:
        request = ring.next_request()
        if request is None:
            return slots

        # The first range sits at offset 0, so its address IS the slot base. Reading the base from
        # the pool would reach into the ring's own state instead.
        base = request.range_dsts[0]
        sizes = [size for file_chunks in request.files for size in file_chunks.sizes]
        slots.append([(dst - base, size) for dst, size in zip(request.range_dsts, sizes)])


def predicted_layout(rank, cap=CAP):
    return [[(r.offset, r.size) for r in slot]
            for slot in predicted_slots(files(), GROUP_SIZE, rank, BUFFER_SIZE, cap)]


class PredictionMatchesTheRing(unittest.TestCase):
    def test_every_slot_of_every_rank_under_every_policy(self):
        """The policy decides which ranges a rank holds, so each one is a different layout."""
        for policy in POLICIES:
            for rank in range(GROUP_SIZE):
                with self.subTest(policy=policy, rank=rank):
                    with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
                        self.assertEqual(predicted_layout(rank), ring_layout(rank))

    def test_a_cap_that_bites_is_predicted_too(self):
        """The production path always caps, and a cap moves the SLOT BOUNDARIES. An offset rule that
        agreed range by range would still predict the wrong slots without this."""
        for policy in POLICIES:
            for rank in range(GROUP_SIZE):
                with self.subTest(policy=policy, rank=rank):
                    with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
                        self.assertEqual(predicted_layout(rank, cap=2),
                                         ring_layout(rank, cap=2))

    def test_the_cap_really_bites_at_two(self):
        """Pins that the test above compares something. Without this, a cap no slot reaches would
        make it a copy of the test before it."""
        with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": "spans"}):
            capped = predicted_layout(0, cap=2)
            uncapped = predicted_layout(0, cap=CAP)
        self.assertTrue(all(len(slot) <= 2 for slot in capped))
        self.assertGreater(len(capped), len(uncapped))

    def test_the_fixture_holds_a_zero_sized_range(self):
        """Pins the case the layout rule treats apart."""
        self.assertIn(0, SIZES)
        with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": "spans"}):
            sizes = [size for slot in predicted_layout(0) for _, size in slot]
        self.assertIn(0, sizes)

    def test_the_layout_is_a_running_sum_and_not_padded(self):
        """Pins the numbers, so the test above cannot pass by comparing two copies of one bug."""
        with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": "files"}):
            first = predicted_layout(0)[0]
        self.assertEqual([offset for offset, _ in first[:4]], [0, 100, 400, 400])
        self.assertEqual([size for _, size in first[:4]], [100, 300, 0, 700])

    def test_a_padded_ring_does_not_match(self):
        """The mutation. With a pad the ring spaces its ranges out, and the prediction must then
        disagree - otherwise the test above proves nothing about the rule."""
        self.assertNotEqual(predicted_layout(0), ring_layout(0, alignment=512))


class PredictionSpeaksTheCallersNumbering(unittest.TestCase):
    def test_ids_are_the_callers_not_the_partitions(self):
        span = partition_for_rank(files(), GROUP_SIZE, 0)
        wanted = set()
        for file_chunks, mapping in span.partition:
            for original_file, original_range, _ in mapping.values():
                wanted.add((original_file, original_range))

        got = {(r.file_id, r.range_index)
               for slot in predicted_slots(files(), GROUP_SIZE, 0, BUFFER_SIZE)
               for r in slot}
        self.assertEqual(got, wanted)


if __name__ == "__main__":
    unittest.main()


class ASharedPartitionPredictsTheSame(unittest.TestCase):
    """A caller may work the shares out once for the whole group and pass each rank its own.

    The two ways of getting a share number their FileChunks differently - one counts across the
    group, the other from zero within the rank - so this asserts the numbering never reaches the
    answer. It is what lets the group's shares be worked out in one pass over the model."""

    def spans_and_whole(self, rank, policy):
        with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
            shares = partition(files(), GROUP_SIZE)
            alone = list(predicted_slots(files(), GROUP_SIZE, rank, BUFFER_SIZE, CAP))
            shared = list(predicted_slots(files(), GROUP_SIZE, rank, BUFFER_SIZE, CAP,
                                          shares[rank]))
        return alone, shared

    def test_every_rank_under_every_policy(self):
        for policy in ("spans", "chunks", "files"):
            for rank in range(GROUP_SIZE):
                with self.subTest(policy=policy, rank=rank):
                    alone, shared = self.spans_and_whole(rank, policy)
                    self.assertEqual(alone, shared)

    def test_the_shares_are_not_all_empty(self):
        """Pins that the comparison above has something to compare."""
        alone, _ = self.spans_and_whole(0, "spans")
        self.assertGreater(sum(len(slot) for slot in alone), 0)
