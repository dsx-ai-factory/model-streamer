# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A send this rank posted is joined before the loop lets go of its slot.

The sends are posted with async_op=True, so NCCL runs them on its own stream: posting makes NCCL
wait on this stream, never the other way round. Only work.wait() puts the dependency back. Without
it the release event completes while the send is still reading the slot, the ring refills it, and
the peers receive bytes the reader has already overwritten - silently.

The window is narrow and real: a turn posts its own send and THEN consumes the turn before it,
which yields to the caller. Anything the caller raises there leaves our send outstanding."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
)
from runai_model_streamer.distributed_streamer.distribution_source import (
    Filled,
    SlotRange,
)


MODULE = "runai_model_streamer.distributed_streamer.distributed_streamer"
SIZE = 64


class FakeWork:
    """What dist.broadcast(async_op=True) hands back."""

    def __init__(self, log, index):
        self.log = log
        self.index = index
        self.waited = False

    def wait(self):
        self.waited = True
        self.log.append(("wait", self.index))


class FakeEvent:
    def record(self, stream=None):
        self.stream = stream


class FakeSupport:
    def new_event(self):
        return FakeEvent()

    def current_stream(self, device):
        return "the stream"


class TwoSlots:
    """A source with two slots, which records how each came back."""

    def __init__(self, log):
        self.log = log
        self.memory = torch.arange(SIZE, dtype=torch.uint8)
        self.left = 2
        self.released = []

    def next_slot(self):
        if self.left <= 0:
            return None
        self.left -= 1
        return Filled(memory=self.memory, ranges=[SlotRange(0, 0, 0, SIZE)])

    def release(self, filled, done=None):
        self.log.append(("release", done))
        self.released.append((filled, done))


class OutstandingSendIsJoined(unittest.TestCase):
    def streamer(self):
        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.file_streamer = MagicMock()
        streamer.device = torch.device("cpu")
        streamer.owned = False
        streamer.group_size = 2
        streamer.rank = 1
        streamer.original_group_rank = 1
        streamer.local_group_global_ranks = [0, 1]
        streamer.distribution_group = object()
        # TWO, because with one the loop consumes the turn before it BEFORE posting the next, and
        # the window this test is about never opens.
        streamer.receive_buffers = 2
        streamer.is_error = False
        # A RECEIVE FROM THE PEER, THEN OUR OWN SEND, which is the only order that leaves a send
        # of ours posted while the loop yields. Two sends in a row are consumed early instead.
        rows = [SlotRange(0, 0, 0, SIZE)]
        streamer.predicted = [[rows], [rows]]
        return streamer

    def run_until_the_caller_raises(self):
        """Drive the loop to the yield inside the second turn's consume, then raise there."""
        log = []
        source = TwoSlots(log)
        streamer = self.streamer()
        works = []

        def broadcast(buffer, rank, group=None, async_op=False):
            work = FakeWork(log, len(works))
            works.append(work)
            return work

        outstanding = [None]
        with patch(f"{MODULE}.device_support", return_value=FakeSupport()), \
             patch(f"{MODULE}.dist.broadcast", side_effect=broadcast), \
             patch(f"{MODULE}.aligned_device_buffer",
                   side_effect=lambda n, device: torch.zeros(n, dtype=torch.uint8)), \
             patch.object(_distributedStreamer, "check_fill_matches_prediction",
                          lambda *a, **k: None):
            chunks = streamer.stream_predicted(source, torch.zeros(SIZE, dtype=torch.uint8),
                                               "the stream", outstanding)
            next(chunks)                      # the peer's rows, yielded after our send is posted
            with self.assertRaises(RuntimeError):
                chunks.throw(RuntimeError("the caller's clone failed"))

        return log, works

    def test_the_outstanding_send_is_waited_for(self):
        log, works = self.run_until_the_caller_raises()

        self.assertEqual(len(works), 2, "the peer's transfer and our own send were posted")
        self.assertTrue(works[0].waited, "the peer's transfer is joined by consume()")
        self.assertTrue(works[1].waited,
                        "and OUR SEND, still outstanding, is joined on the way out")

    def test_it_is_joined_before_the_slot_goes_back(self):
        """Order, not just the fact. A wait after the release is no use: the ring may already
        have refilled the slot by then."""
        log, _ = self.run_until_the_caller_raises()

        waits = [i for i, (what, _) in enumerate(log) if what == "wait"]
        releases = [i for i, (what, _) in enumerate(log) if what == "release"]
        self.assertTrue(waits, "nothing was ever joined")
        if releases:
            self.assertLess(max(waits), min(releases),
                            "every join must come before the slot goes back")


if __name__ == "__main__":
    unittest.main()
