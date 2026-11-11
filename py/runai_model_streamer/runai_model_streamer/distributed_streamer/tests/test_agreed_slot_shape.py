# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    agreed_slot_shape,
)
from runai_model_streamer.distributed_streamer.partition import partition_for_rank
from runai_model_streamer.file_streamer.requests_iterator import (
    device_ring_budget,
    device_ring_sizing,
    FileChunks,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
)

FREE = 1 << 30


def files(sizes_per_file):
    return [FileChunks.contiguous(index, f"f{index}.bin", 0, sizes)
            for index, sizes in enumerate(sizes_per_file)]


class TestAgreedSlotShape(unittest.TestCase):
    """The shape every rank of a group sends from.

    What it has to give is one answer from inputs every rank already shares - no reduction of its
    own, because a second reduction is a second place to disagree."""

    def share_of(self, whole, rank, ranks=2):
        return [file_chunks for file_chunks, _ in partition_for_rank(whole, ranks, rank).partition]

    def test_two_ranks_agree_only_when_both_inputs_are_the_agreed_ones(self):
        """The property the whole thing rests on, with the ranks made different.

        Each rank has its OWN free memory and its OWN share. Feeding those gives two shapes, so a
        ring sized that way could not be broadcast between them. Feeding the group minimum and the
        whole list - which both ranks already hold - gives one shape, with no reduction of its own.
        Both inputs matter, so both are varied here."""
        whole = files([[1 << 20, 1 << 20], [8, 8, 8, 8]])
        own_free = {0: 1 << 28, 1: 1 << 26}      # the ranks do not have the same room
        group_min = min(own_free.values())

        apart = [agreed_slot_shape(own_free[rank], self.share_of(whole, rank)) for rank in (0, 1)]
        self.assertNotEqual(apart[0], apart[1],
                            "this fixture gives both ranks the same shape, so it proves nothing")

        together = [agreed_slot_shape(group_min, whole) for rank in (0, 1)]
        self.assertEqual(together[0], together[1])
        self.assertNotIn(together[0], apart,
                         "the agreed shape is neither rank's own answer, so it had to be agreed")

    def test_only_the_shared_free_memory_agrees(self):
        """Half of the previous test, isolated: the same list, each rank's own free memory."""
        whole = files([[1 << 20] * 64])

        self.assertNotEqual(agreed_slot_shape(1 << 28, whole),
                            agreed_slot_shape(1 << 26, whole))

    def test_order_changes_the_shape(self):
        """RECORDED, not desired. The sizing packs ranges in list order to score a depth, so the
        same files in another order can answer differently - which makes "the same list" a
        requirement on ORDER, not only on contents."""
        big, small = [1 << 20] * 8, [8] * 24
        grouped = files([big + small])
        interleaved = files([[size for pair in zip(big, small[:8]) for size in pair] + small[8:]])

        self.assertNotEqual(agreed_slot_shape(128 << 20, grouped),
                            agreed_slot_shape(128 << 20, interleaved))

    def test_a_group_that_agreed_nothing_has_no_shape(self):
        """None reaches here from a group whose device torch cannot measure, or one not on a device
        backend: the branch that reduces a figure was never taken."""
        self.assertIsNone(agreed_slot_shape(None, files([[1 << 20]])))

    def test_an_empty_request_asks_for_nothing(self):
        """RECORDED. A caller with no ranges gets a zero-sized shape rather than None, and a ring
        built from it would hold nothing - the caller is expected not to stream an empty list."""
        self.assertEqual(agreed_slot_shape(FREE, []), (0, 1))

    def test_the_largest_range_in_the_whole_list_fits(self):
        """A receiver allocates this size too, so it must hold the largest range ANY rank sends -
        including one this rank has no share of."""
        whole = files([[8], [1 << 20]])

        buffer_size, _ = agreed_slot_shape(FREE, whole)
        self.assertGreaterEqual(buffer_size, 1 << 20)

    def test_a_share_would_have_given_a_smaller_answer(self):
        """Why the whole list is passed: the rank holding only small ranges would otherwise agree a
        buffer too small for what its peer broadcasts."""
        whole = files([[8, 8], [1 << 20]])
        small_share = files([[8, 8]])

        self.assertLess(agreed_slot_shape(FREE, small_share)[0],
                        agreed_slot_shape(FREE, whole)[0])

    def test_no_room_answers_none(self):
        self.assertIsNone(agreed_slot_shape(1, files([[1 << 20]])))

    def test_the_depth_is_the_one_asked_for(self):
        """Passed as an ARGUMENT, with the environment set to something else: the sizing underneath
        reads that variable itself, so a test that only set it would pass whether or not this
        function handed the value on."""
        with patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "9"}):
            self.assertEqual(agreed_slot_shape(FREE, files([[1000] * 16]), ring_buffers=4)[1], 4)

    def test_free_memory_changes_the_shape(self):
        """It is sized from the agreed figure, so a different one must land somewhere - otherwise
        passing the group's minimum rather than this rank's own would make no difference."""
        big = agreed_slot_shape(1 << 34, files([[1 << 20] * 64]))
        small = agreed_slot_shape(1 << 26, files([[1 << 20] * 64]))

        self.assertNotEqual(big, small)
        self.assertLess(small[0] * small[1], big[0] * big[1])


if __name__ == "__main__":
    unittest.main()


class TheReceiveBufferIsReservedOutOfTheBudget(unittest.TestCase):
    """A rank that sends its slots also receives its peers', into a buffer of the same size.

    The ring is allocated by stream_files and the receive buffer later, in get_chunks. If the ring
    took the whole budget there would be nothing left for it, and the rank would raise an allocation
    error after its peers had already entered the collective."""

    def whole(self):
        """MORE DATA THAN BUDGET, which is what makes the reservation bite at all.

        A stream that fits the budget whole is spanned by a ring smaller than the budget, so there
        is already room to receive into and nothing is reserved. Sizing the fixture to fit would
        make every assertion below vacuous."""
        return files([[1 << 20] * 4096])

    def test_the_fixture_really_does_fill_the_budget(self):
        """Asserted, not assumed: it is what makes the rest of this class mean anything."""
        buffer_size, num_buffers = device_ring_sizing(FREE, self.whole())
        budget = device_ring_budget(FREE, self.whole())

        self.assertGreater((num_buffers + 1) * buffer_size, budget,
                           "an unreserved ring must leave no room for a receive buffer here")

    def test_the_ring_leaves_room_for_one_more_slot(self):
        """The property, stated as arithmetic rather than as a depth: ring plus receive fits."""
        shape = agreed_slot_shape(FREE, self.whole())
        self.assertIsNotNone(shape)

        buffer_size, num_buffers = shape
        budget = device_ring_budget(FREE, self.whole())
        self.assertLessEqual((num_buffers + 1) * buffer_size, budget)

    def test_it_costs_depth_and_never_slot_size(self):
        """The slot is the broadcast unit. Shrinking it would break every peer; one buffer fewer
        costs this rank some overlap and nobody else anything."""
        unreserved = device_ring_sizing(FREE, self.whole())
        reserved = agreed_slot_shape(FREE, self.whole())

        self.assertEqual(reserved[0], unreserved[0], "the slot size is untouched")
        self.assertEqual(reserved[1], unreserved[1] - 1, "exactly one buffer pays for it")

    def test_a_budget_with_room_for_one_slot_only_falls_back(self):
        """Room for a ring but not for a ring AND somewhere to receive into. Every rank reaches that
        from the same numbers, so the whole group falls back together."""
        one_big = files([[1 << 20]])
        tight = device_ring_floor_free(one_big)

        shape = agreed_slot_shape(tight, one_big)
        if shape is not None:
            buffer_size, num_buffers = shape
            self.assertLessEqual((num_buffers + 1) * buffer_size,
                                 device_ring_budget(tight, one_big))

    def test_an_empty_file_list_is_unchanged(self):
        """Nothing to size, and the reservation must not turn that into a fallback."""
        self.assertEqual(agreed_slot_shape(FREE, []), (0, 1))


def device_ring_floor_free(files_chunks):
    """The free memory at which the budget is exactly the floor."""
    from runai_model_streamer.file_streamer.requests_iterator import (
        device_ring_floor,
        _device_memory_fraction,
    )
    return int(device_ring_floor(files_chunks) / _device_memory_fraction())
