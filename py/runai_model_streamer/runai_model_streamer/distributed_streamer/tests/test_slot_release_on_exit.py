# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The slot goes back however the loop ends.

A caller that stops part way through raises GeneratorExit at the yield. It derives from
BaseException, so neither `except` in get_chunks catches it, and before this the slot in flight was
simply dropped. For a source the library refills that is not a leak but a hang: release is what
hands the event over, and without it the teardown waits on a stream the collective was never
recorded on."""

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


class FakeSupport:
    """A vendor row whose current stream the test can move."""

    def __init__(self):
        self.stream = "the stream it started on"

    def new_event(self):
        class Event:
            def record(self, stream=None):
                self.stream = stream

        return Event()

    def current_stream(self, device):
        return self.stream


class Recording:
    """A source with two slots to give, which remembers how each came back."""

    def __init__(self, slots=2, refuse=False):
        self.memory = torch.arange(1, 1 + 64, dtype=torch.uint8)
        self.left = slots
        self.refuse = refuse
        self.released = []

    def next_slot(self):
        if self.left <= 0:
            return None
        self.left -= 1
        return Filled(memory=self.memory, ranges=[SlotRange(0, 0, 0, 64)])

    def release(self, filled, done=None):
        if self.refuse:
            raise RuntimeError("this source will not take it back")
        self.released.append((filled, done))


class Refilled(Recording):
    """A source the library refills: it insists on an event, as a device ring's pool does."""

    def release(self, filled, done=None):
        if done is None:
            raise ValueError("needs an event")
        super().release(filled, done)


class SlotReleaseOnExit(unittest.TestCase):
    def streamer(self):
        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.file_streamer = MagicMock()
        streamer.device = torch.device("cpu")
        streamer.max_chunk = 64
        streamer.rank_dicts_map = {}
        streamer.owned = False
        streamer.group_size = 1
        streamer.rank = 0
        streamer.original_group_rank = 0
        streamer.local_group_global_ranks = [0]
        streamer.distribution_group = object()
        streamer.reading_from_storage = True
        streamer.sends_directly = False
        streamer.slot_shape = None
        streamer.total_chunks_to_read = 2
        streamer.is_error = False
        return streamer

    def run_until_first_yield(self, source):
        """Start the loop, take one tensor, then walk away from it."""
        streamer = self.streamer()
        with patch(f"{MODULE}.StagedDistributionSource", return_value=source), \
             patch(f"{MODULE}.dist.broadcast"):
            chunks = streamer.get_chunks()
            next(chunks)
            chunks.close()

    def test_an_abandoned_stream_does_not_resume_the_ring(self):
        """A caller that walked away gets no further read.

        Releasing resumes the library's own generator, which reads a whole further slot before it
        returns - and an abandoned generator is closed by the collector, so that read would happen
        at an arbitrary moment. The streamer's teardown waits on the stream instead, which covers
        the collective because both run on it."""
        source = Recording()

        self.run_until_first_yield(source)

        self.assertEqual(source.released, [], "nothing is pulled through on the way out")

    def test_an_abandoned_stream_does_not_resume_a_refilling_source_either(self):
        """Same answer for the source that would actually do the read."""
        source = Refilled()
        streamer = self.streamer()
        streamer.sends_directly = True
        streamer.slot_shape = (64, 2)

        with patch(f"{MODULE}.DirectDistributionSource", return_value=source), \
             patch(f"{MODULE}.device_support", return_value=FakeSupport()), \
             patch(f"{MODULE}.dist.broadcast"):
            chunks = streamer.get_chunks()
            next(chunks)
            chunks.close()

        self.assertEqual(source.released, [])

    def test_a_failed_round_still_gives_the_slot_back(self):
        """What the finally is for. The loop raised rather than being closed, so there is no
        collector to worry about, and the slot has no other path back."""
        source = Refilled()
        streamer = self.streamer()
        streamer.sends_directly = True
        streamer.slot_shape = (64, 2)

        with patch(f"{MODULE}.DirectDistributionSource", return_value=source), \
             patch(f"{MODULE}.device_support", return_value=FakeSupport()), \
             patch(f"{MODULE}.dist.broadcast", side_effect=RuntimeError("the collective failed")):
            with self.assertRaises(RuntimeError):
                list(streamer.get_chunks())

        self.assertEqual(len(source.released), 1, "the slot comes back")
        self.assertIsNotNone(source.released[0][1], "with the event this source demands")

    def test_a_source_that_refuses_does_not_replace_the_reason_for_stopping(self):
        """The finally runs while the generator is closing. Raising there would turn a tidy-up
        failure into the thing the caller sees."""
        source = Recording(refuse=True)

        self.run_until_first_yield(source)      # must not raise

    def test_a_caller_that_moved_streams_still_gives_the_slot_back(self):
        """The two changes must not cancel each other.

        The stream guard raises at the release inside the loop, which leaves the slot held. Asking
        the guard a second time in the finally would raise again and strand the ring - by the very
        code written to save it."""
        source = Refilled()
        streamer = self.streamer()
        streamer.sends_directly = True
        streamer.slot_shape = (64, 2)

        support = FakeSupport()
        with patch(f"{MODULE}.DirectDistributionSource", return_value=source), \
             patch(f"{MODULE}.device_support", return_value=support), \
             patch(f"{MODULE}.dist.broadcast"):
            chunks = streamer.get_chunks()
            next(chunks)
            support.stream = "a stream the caller moved to"
            with self.assertRaises(ValueError):
                list(chunks)

        self.assertEqual(len(source.released), 1, "the slot is given back anyway")
        self.assertIsNotNone(source.released[0][1], "and with an event, which this source demands")

    def test_a_slot_already_given_back_is_not_released_twice(self):
        """Releasing a slot the source has taken back would hand it an event for the slot that
        replaced it."""
        source = Recording(slots=1)
        streamer = self.streamer()
        streamer.total_chunks_to_read = 1

        with patch(f"{MODULE}.StagedDistributionSource", return_value=source), \
             patch(f"{MODULE}.dist.broadcast"):
            list(streamer.get_chunks())

        self.assertEqual(len(source.released), 1, "once, by the loop and not again by the finally")


if __name__ == "__main__":
    unittest.main()
