# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The slot size is a contract between ranks, and it is checked rather than assumed.

Every rank computes the slot size alone and then reads its peers' bytes at offsets derived from it.
Some of the inputs are read PER RANK - the ring depth setting, the device memory limit, its
fraction - so two ranks given different environments compute different sizes. Nothing downstream
notices: the load finishes and the weights are wrong."""

import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.distributed_streamer import distributed_streamer as module
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
)

GROUP = object()
MINE = 1 << 20


def streamer(slot_size=MINE, sends_directly=True, group=GROUP):
    s = _distributedStreamer.__new__(_distributedStreamer)
    s.distribution_group = group
    s.device = "cpu"
    s.original_group_rank = 0
    s.slot_shape = (slot_size, 2) if slot_size else None
    s.sends_directly = sends_directly
    return s


def peers(*sizes):
    """A fake MAX all_reduce over this rank and `sizes`.

    The real one leaves [largest, -smallest] in the tensor, which is how one collective answers
    both ends. Reproduced here rather than mocked away, so the test fails if that encoding changes.
    """
    def fake(tensor, op=None, group=None):
        everyone = [int(tensor[0].item())] + list(sizes)
        tensor[0] = max(everyone)
        tensor[1] = max(-size for size in everyone)
    return fake


class TheGroupMustAgreeOnTheSlotSize(unittest.TestCase):
    def test_a_group_that_agrees_passes(self):
        with patch.object(module.dist, "all_reduce", side_effect=peers(MINE, MINE)):
            streamer().check_the_group_agreed()

    def test_a_peer_that_packed_larger_is_refused(self):
        with patch.object(module.dist, "all_reduce", side_effect=peers(MINE, 4 * MINE)):
            with self.assertRaisesRegex(RuntimeError, "did not agree on the slot size"):
                streamer().check_the_group_agreed()

    def test_a_peer_that_packed_smaller_is_refused(self):
        """Both ends, from the one collective: this rank is the largest here, not the smallest."""
        with patch.object(module.dist, "all_reduce", side_effect=peers(MINE // 4)):
            with self.assertRaisesRegex(RuntimeError, "did not agree on the slot size"):
                streamer().check_the_group_agreed()

    def test_the_message_names_the_two_sizes(self):
        """A rank cannot see its peers' environments, so the number is the only lead it has."""
        with patch.object(module.dist, "all_reduce", side_effect=peers(4 * MINE)):
            with self.assertRaises(RuntimeError) as caught:
                streamer().check_the_group_agreed()
        self.assertIn(str(MINE), str(caught.exception))
        self.assertIn(str(4 * MINE), str(caught.exception))

    def test_it_names_the_settings_that_move_the_size(self):
        """The ones read per rank. Without them the message says what is wrong and not where.

        RUNAI_STREAMER_RING_BUFFERS is NOT one of them: it governs the host ring. Naming a variable
        that cannot cause the failure sends whoever reads this to the wrong place."""
        with patch.object(module.dist, "all_reduce", side_effect=peers(4 * MINE)):
            with self.assertRaises(RuntimeError) as caught:
                streamer().check_the_group_agreed()
        message = str(caught.exception)
        self.assertIn("RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB", message)
        self.assertIn("RUNAI_STREAMER_DEVICE_RING_SLOTS", message)
        self.assertIn("RUNAI_STREAMER_DEVICE_MEMORY_LIMIT", message)
        self.assertIn("RUNAI_STREAMER_DEVICE_MEMORY_FRACTION", message)
        self.assertNotIn("RUNAI_STREAMER_RING_BUFFERS", message)


class AGroupThatSplitBetweenThePathsIsCaughtToo(unittest.TestCase):
    """One rank sending ring slots while another stages posts a different number of collectives per
    round, so the two would wait for each other until the group times out."""

    def test_staging_against_sending_is_refused(self):
        with patch.object(module.dist, "all_reduce", side_effect=peers(0)):
            with self.assertRaisesRegex(RuntimeError, "did not agree on the slot size"):
                streamer().check_the_group_agreed()

    def test_a_group_that_all_stages_agrees(self):
        """Zero everywhere is agreement, not a failure - the whole group takes the host path."""
        with patch.object(module.dist, "all_reduce", side_effect=peers(0, 0)):
            streamer(slot_size=None, sends_directly=False).check_the_group_agreed()

    def test_a_rank_that_stages_sends_zero_whatever_shape_it_holds(self):
        """sends_directly decides, not the shape: a rank can hold a shape and still stage, and the
        peers must see the path it will actually take."""
        sent = []
        with patch.object(module.dist, "all_reduce",
                          side_effect=lambda t, op=None, group=None: sent.append(int(t[0].item()))):
            streamer(slot_size=MINE, sends_directly=False).check_the_group_agreed()
        self.assertEqual(sent, [0])


class WithoutAGroupThereIsNothingToAgreeWith(unittest.TestCase):
    def test_no_group_posts_no_collective(self):
        """An undistributed stream reaches here too, and a collective on no group would raise."""
        with patch.object(module.dist, "all_reduce") as reduced:
            streamer(group=None).check_the_group_agreed()
        reduced.assert_not_called()


if __name__ == "__main__":
    unittest.main()
