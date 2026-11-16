# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The predicted path posts one transfer per sender and never waits for metadata.

Under gloo the direct path is unreachable, so the loop is driven here instead. What matters is how
many collectives a round costs, where the sizes come from, and that a receive buffer is never posted
into while the clones out of it are still pending."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from runai_model_streamer.distributed_streamer.distribution_source import (
    Filled,
    SlotRange,
)
from runai_model_streamer.distributed_streamer import distributed_streamer as module
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
)


GROUP = object()
SIZE = 100
# Packed one after another, as a sent ring lays them out.
ROWS = [SlotRange(7, 3, 0, SIZE), SlotRange(2, 0, SIZE, SIZE)]
TOTAL = 2 * SIZE


class FakeEvent:
    def __init__(self):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True


class FakeStream:
    def __init__(self):
        self.waited_on = []

    def wait_event(self, event):
        self.waited_on.append(event)


class FakeSource:
    """Hands out one slot at a time and remembers how they were given back."""

    def __init__(self, fills):
        self.fills = list(fills)
        self.released = []

    def next_slot(self):
        return self.fills.pop(0) if self.fills else None

    def release(self, filled, event):
        self.released.append((filled, event))


def streamer(rank, group_size, predicted, receive_buffers=2):
    s = _distributedStreamer.__new__(_distributedStreamer)
    s.receive_buffers = receive_buffers
    s.group_size = group_size
    s.original_group_rank = rank
    s.local_group_global_ranks = list(range(group_size))
    s.distribution_group = GROUP
    s.rank = rank
    s.predicted = predicted
    s.device = "cpu"
    return s


class Harness(unittest.TestCase):
    def drive(self, s, source, received=None):
        """Run the pipelined loop and collect the posts and what came out."""
        posts = []
        received = torch.zeros(TOTAL + 64, dtype=torch.uint8) if received is None else received
        stream = FakeStream()
        support = MagicMock()
        support.new_event.side_effect = FakeEvent
        support.current_stream.return_value = stream

        def post(tensor, src=None, group=None, async_op=False):
            posts.append((tensor, src))
            tensor.copy_(torch.arange(len(tensor), dtype=torch.uint8))
            return MagicMock()

        with patch.object(module.dist, "broadcast", side_effect=post), \
             patch.object(module, "device_support", return_value=support), \
             patch.object(_distributedStreamer, "slot_is_free", lambda self, st=None: "EVENT_A"):
            out = list(s.stream_predicted(source, received, None, [None]))
        return posts, out, stream


class Sender(Harness):
    def test_one_post_per_turn_and_no_metadata(self):
        s = streamer(0, 1, [[ROWS]])
        source = FakeSource([Filled(torch.arange(TOTAL, dtype=torch.uint8), ROWS)])
        posts, out, _ = self.drive(s, source)
        self.assertEqual(len(posts), 1)
        self.assertEqual([(f, i) for f, i, _ in out], [(7, 3), (2, 0)])

    def test_the_slot_is_given_back_with_its_event(self):
        s = streamer(0, 1, [[ROWS]])
        source = FakeSource([Filled(torch.arange(TOTAL, dtype=torch.uint8), ROWS)])
        self.drive(s, source)
        self.assertEqual(len(source.released), 1)
        self.assertEqual(source.released[0][1], "EVENT_A")

    def test_a_slot_no_rank_predicted_is_refused(self):
        """A slot nobody expects is never broadcast, so without this the weights in it are lost and
        the load still reports success."""
        s = streamer(0, 1, [[ROWS]])
        source = FakeSource([Filled(torch.arange(TOTAL, dtype=torch.uint8), ROWS),
                             Filled(torch.arange(TOTAL, dtype=torch.uint8), ROWS)])
        with self.assertRaisesRegex(RuntimeError, "no rank predicted"):
            self.drive(s, source)

    def test_the_source_is_asked_once_past_the_plan(self):
        """Pins that the test above compares something: a source holding exactly what was predicted
        is still asked, and still passes."""
        s = streamer(0, 1, [[ROWS]])
        source = FakeSource([Filled(torch.arange(TOTAL, dtype=torch.uint8), ROWS)])
        self.drive(s, source)
        self.assertEqual(source.fills, [])

    def test_a_fill_that_is_not_what_was_predicted_raises(self):
        s = streamer(0, 1, [[ROWS]])
        renamed = [SlotRange(99, 1, 0, SIZE), ROWS[1]]
        source = FakeSource([Filled(torch.arange(TOTAL, dtype=torch.uint8), renamed)])
        with self.assertRaises(RuntimeError) as caught:
            self.drive(s, source)
        self.assertIn("wrong names", str(caught.exception))


class ASenderWithSeveralTurns(Harness):
    """A group of one: every turn is this rank's, so one send follows the one before it with no
    receive between them. That is the ONLY shape where the posted turn is also a send, and it is
    the shape the mid-loop consume exists for."""

    def plan(self, turns):
        return [Filled(torch.arange(TOTAL, dtype=torch.uint8), ROWS) for _ in range(turns)]

    def test_each_slot_is_given_back_with_the_event_of_its_own_broadcast(self):
        """A slot goes back only once its broadcast has been consumed, so the event goes with it.
        Released without one, the ring refills a slot the collective is still reading."""
        s = streamer(0, 1, [[ROWS, ROWS]])
        source = FakeSource(self.plan(2))
        posts, out, _ = self.drive(s, source)

        self.assertEqual(len(posts), 2)
        self.assertEqual([event for _, event in source.released], ["EVENT_A", "EVENT_A"])
        self.assertEqual([(f, i) for f, i, _ in out], [(7, 3), (2, 0)] * 2)

    def test_one_slot_is_out_at_a_time(self):
        """Pins why the consume comes first: asking for a slot gives the previous one back, so the
        release must already have happened."""
        s = streamer(0, 1, [[ROWS, ROWS, ROWS]])
        source = FakeSource(self.plan(3))
        self.drive(s, source)
        self.assertEqual(len(source.released), 3)


class Receiver(Harness):
    """Rank 1 of two with nothing to send, so every turn that carries rows is a receive.

    Rank 1 and not rank 2: a rank outside its own group is not a configuration that can exist, and
    a setup value has to mean what it says."""

    def test_it_posts_one_transfer_sized_from_the_prediction(self):
        s = streamer(1, 2, [[ROWS], []])
        posts, out, _ = self.drive(s, FakeSource([]))
        self.assertEqual(len(posts), 1)
        tensor, src = posts[0]
        self.assertEqual(src, 0)
        self.assertEqual(len(tensor), TOTAL)
        self.assertEqual([(f, i) for f, i, _ in out], [(7, 3), (2, 0)])

    def test_a_turn_a_rank_has_no_slot_for_is_skipped(self):
        s = streamer(1, 2, [[], []])
        posts, out, _ = self.drive(s, FakeSource([]))
        self.assertEqual(posts, [])
        self.assertEqual(out, [])

    def test_a_receive_buffer_is_not_reposted_before_its_clones_finish(self):
        """Two buffers alternate, so the third turn returns to the first and must wait on it."""
        s = streamer(1, 2, [[ROWS, ROWS, ROWS], []])
        posts, _, stream = self.drive(s, FakeSource([]))
        self.assertEqual(len(posts), 3)
        self.assertEqual(len(stream.waited_on), 1)

    def test_one_receive_buffer_waits_on_every_turn_but_the_first(self):
        """A budget with no room for a second buffer. The same transfers are posted, and the same
        bytes come out - what is given up is the overlap, which shows as a wait per turn."""
        s = streamer(1, 2, [[ROWS, ROWS, ROWS], []], receive_buffers=1)
        posts, out, stream = self.drive(s, FakeSource([]))
        self.assertEqual(len(posts), 3)
        self.assertEqual(len(stream.waited_on), 2)
        self.assertEqual([(f, i) for f, i, _ in out], [(7, 3), (2, 0)] * 3)

    def test_the_second_buffer_is_made_the_same_way_as_the_first(self):
        """Both must begin on the same remainder - an offset means the same thing in a sender's
        buffer and a receiver's copy only then. empty_like of an already sliced tensor promises
        nothing about where it starts."""
        s = streamer(1, 2, [[ROWS, ROWS], []])
        received = torch.zeros(TOTAL + 64, dtype=torch.uint8)
        with patch.object(module, "aligned_device_buffer",
                          return_value=torch.zeros(TOTAL + 64, dtype=torch.uint8)) as made:
            self.drive(s, FakeSource([]), received=received)
        made.assert_called_once_with(received.numel(), "cpu")

    def test_a_rank_that_cannot_afford_a_second_buffer_keeps_loading(self):
        """The count is local and changes no collective, so a rank short of memory gives up the
        overlap rather than failing the whole group. It then behaves as a one buffer rank."""
        s = streamer(1, 2, [[ROWS, ROWS, ROWS], []])
        with patch.object(module, "aligned_device_buffer",
                          side_effect=torch.cuda.OutOfMemoryError("no room")):
            posts, out, stream = self.drive(s, FakeSource([]))
        self.assertEqual(len(posts), 3, "the same transfers are still posted")
        self.assertEqual(len(stream.waited_on), 2, "and it waits per turn, as one buffer does")
        self.assertEqual([(f, i) for f, i, _ in out], [(7, 3), (2, 0)] * 3)

    def test_two_buffers_are_only_used_when_the_group_agreed_them(self):
        """Pins that the pair above differ by the buffer count and nothing else."""
        s = streamer(1, 2, [[ROWS, ROWS, ROWS], []])
        _, two, _ = self.drive(s, FakeSource([]))
        s = streamer(1, 2, [[ROWS, ROWS, ROWS], []], receive_buffers=1)
        _, one, _ = self.drive(s, FakeSource([]))
        self.assertEqual([(f, i) for f, i, _ in two], [(f, i) for f, i, _ in one])


if __name__ == "__main__":
    unittest.main()
