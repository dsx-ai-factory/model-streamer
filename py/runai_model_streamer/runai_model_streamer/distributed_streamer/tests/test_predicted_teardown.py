# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every way out of the predicted loop, driven through get_chunks.

The loop itself is tested elsewhere by calling stream_predicted directly. What is covered HERE is
what wraps it: the cell that holds the slot in flight, the clone on the way out, and the three exits
- the stream ending, the stream raising, and a caller walking away.

A slot that is not given back strands the ring for the rest of the load, and a slot given back too
early is refilled while the collective is still reading it. Neither raises."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from runai_model_streamer.distributed_streamer import distributed_streamer as module
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
)
from runai_model_streamer.distributed_streamer.distribution_source import Filled, SlotRange

SIZE = 100
ROWS = [SlotRange(7, 3, 0, SIZE), SlotRange(2, 0, SIZE, SIZE)]
TOTAL = 2 * SIZE


class FakeSource:
    """One slot per turn, and a record of how each was given back."""

    def __init__(self, turns, raise_on=None):
        self.left = turns
        self.raise_on = raise_on
        self.released = []
        self.handed_out = 0

    def next_slot(self):
        if self.raise_on is not None and self.handed_out == self.raise_on:
            raise RuntimeError("the ring gave up")
        if self.left <= 0:
            return None
        self.left -= 1
        self.handed_out += 1
        return Filled(torch.arange(TOTAL, dtype=torch.uint8), ROWS)

    def release(self, filled, event):
        self.released.append((filled, event))


def streamer(turns, owned=True, raise_on=None):
    """A rank of one with `turns` slots to send, wired for the predicted path."""
    s = _distributedStreamer.__new__(_distributedStreamer)
    s.file_streamer = MagicMock()
    s.device = "cpu"
    s.max_chunk = 1 << 20
    # One metadata row per range of the request - two here, as ROWS has two.
    s.metadata_rows = len(ROWS)
    s.slot_shape = (TOTAL, 2)
    s.receive_buffers = 2
    s.sends_directly = True
    s.reading_from_storage = True
    s.rank_dicts_map = {}
    s.total_chunks_to_read = 0
    s.owned = owned
    s.is_error = False
    s.group_size = 1
    s.rank = 0
    s.original_group_rank = 0
    s.local_group_global_ranks = [0]
    s.distribution_group = object()
    s.predicted = [[ROWS] * turns]
    s.source = FakeSource(turns, raise_on=raise_on)
    return s


class Driven(unittest.TestCase):
    def drive(self, s):
        """get_chunks with the device, the staging and the collective faked out."""
        support = MagicMock()
        support.new_event.side_effect = lambda: MagicMock()
        support.current_stream.return_value = MagicMock()
        staging = (None, torch.zeros(TOTAL + 64, dtype=torch.uint8),
                   torch.zeros(4, 4), torch.zeros(4, 4))

        def post(tensor, src=None, group=None, async_op=False):
            return MagicMock()

        return patch.multiple(
            module,
            allocate_dist_staging=MagicMock(return_value=staging),
            DirectDistributionSource=MagicMock(return_value=s.source),
            device_support=MagicMock(return_value=support),
        ), patch.object(module.dist, "broadcast", side_effect=post), \
            patch.object(_distributedStreamer, "slot_is_free", lambda self, st=None: "EVENT")


class TheStreamRunsOut(Driven):
    def test_every_slot_is_given_back_and_nothing_is_left_held(self):
        s = streamer(turns=2)
        a, b, c = self.drive(s)
        with a, b, c:
            out = list(s.get_chunks())

        self.assertEqual([(f, i) for f, i, _ in out], [(7, 3), (2, 0)] * 2)
        self.assertEqual(len(s.source.released), 2, "a slot was not given back")
        self.assertEqual([event for _, event in s.source.released], ["EVENT", "EVENT"])

    def test_owned_hands_out_a_copy_and_not_the_slot(self):
        """The slot is refilled behind the caller, so an owned tensor must not be a view of it."""
        s = streamer(turns=1, owned=True)
        a, b, c = self.drive(s)
        with a, b, c:
            tensors = [t for _, _, t in s.get_chunks()]

        for tensor in tensors:
            self.assertEqual(tensor.data_ptr() % 1, 0)
            self.assertEqual(tensor.storage_offset(), 0, "a clone starts its own storage")

    def test_unowned_hands_out_the_slot_itself(self):
        """Pins that the test above is measuring the clone: the same stream without it yields views
        into the slot, which start part way into it."""
        s = streamer(turns=1, owned=False)
        a, b, c = self.drive(s)
        with a, b, c:
            offsets = [t.storage_offset() for _, _, t in s.get_chunks()]

        self.assertEqual(offsets, [0, SIZE], "these are views, not copies")


class TheStreamRaises(Driven):
    def test_the_slot_in_flight_is_given_back(self):
        """A FINALLY, because the loop left by an exception and the ring is otherwise stranded for
        the rest of the load."""
        s = streamer(turns=3, raise_on=1)
        a, b, c = self.drive(s)
        with a, b, c:
            with self.assertRaisesRegex(RuntimeError, "the ring gave up"):
                list(s.get_chunks())

        self.assertEqual(len(s.source.released), 1, "the held slot was not given back")
        self.assertTrue(s.is_error)

    def test_the_error_reaches_the_caller(self):
        """Tidying up must not swallow what went wrong."""
        s = streamer(turns=3, raise_on=1)
        a, b, c = self.drive(s)
        with a, b, c:
            with self.assertRaises(RuntimeError):
                list(s.get_chunks())


class TheCallerWalksAway(Driven):
    """close() raises GeneratorExit at the yield. The slot is deliberately NOT released there:
    releasing resumes the library's own generator, which reads a whole further slot before it
    returns, and a generator nobody kept is closed by the collector - so that read would happen at
    an arbitrary moment."""

    def test_closing_part_way_releases_nothing_and_does_not_raise(self):
        s = streamer(turns=3)
        a, b, c = self.drive(s)
        with a, b, c:
            chunks = s.get_chunks()
            next(chunks)
            chunks.close()

        self.assertEqual(s.source.released, [], "the slot was given back on the abandoned path")

    def test_a_slot_was_actually_in_flight_when_it_closed(self):
        """Pins that the test above closed mid-stream rather than after the loop had tidied up."""
        s = streamer(turns=3)
        a, b, c = self.drive(s)
        with a, b, c:
            chunks = s.get_chunks()
            next(chunks)
            self.assertEqual(s.source.handed_out, 1)
            chunks.close()


if __name__ == "__main__":
    unittest.main()
