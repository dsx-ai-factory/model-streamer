# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The event that says the broadcast and the clones are done with a slot.

A source that refills its slots will not take them back without one. Nothing here needs a GPU: what
is worth pinning is WHICH stream the event is recorded on, and that is a plain call."""

import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
)


class FakeEvent:
    def __init__(self):
        self.recorded_on = "never recorded"

    def record(self, stream=None):
        self.recorded_on = stream


class FakeSupport:
    """Stands in for a vendor row. current_stream answers a different stream per device, which is
    the whole point of passing the device in."""

    def __init__(self):
        self.events = []
        self.asked_for = []

    def new_event(self):
        event = FakeEvent()
        self.events.append(event)
        return event

    def current_stream(self, device):
        self.asked_for.append(device)
        return f"stream of {device}"


class SlotIsFree(unittest.TestCase):
    def streamer(self, device):
        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.device = device
        return streamer

    def test_a_device_without_events_gets_none(self):
        """The host under gloo. A staged source is ordered by its own stream and needs nothing."""
        with patch(
            "runai_model_streamer.distributed_streamer.distributed_streamer.device_support",
            return_value=None,
        ):
            self.assertIsNone(self.streamer("cpu").slot_is_free())

    def test_the_event_is_recorded_on_this_streamer_s_device(self):
        """NOT on whichever device is current.

        They differ once a process serves more than one, and an event on the wrong device completes
        at the wrong time - it would free the slot while the clones are still reading it."""
        support = FakeSupport()

        with patch(
            "runai_model_streamer.distributed_streamer.distributed_streamer.device_support",
            return_value=support,
        ):
            done = self.streamer("cuda:3").slot_is_free()

        self.assertIs(done, support.events[0])
        self.assertEqual(support.asked_for, ["cuda:3"])
        self.assertEqual(done.recorded_on, "stream of cuda:3")

    def test_a_device_with_no_ordinal_is_never_what_is_recorded_on(self):
        """stream_files resolves the ordinal, so this method must never see a bare "cuda".

        A bare one means "whichever device is current", which can answer differently later - the
        staging buffers and this event would then sit on different devices. The resolution is
        asserted where it happens; here the point is that an index-less device would be passed
        straight through, so nothing downstream repairs it."""
        support = FakeSupport()
        streamer = self.streamer(torch.device("cuda"))

        with patch(
            "runai_model_streamer.distributed_streamer.distributed_streamer.device_support",
            return_value=support,
        ):
            streamer.slot_is_free()

        self.assertIsNone(support.asked_for[0].index,
                          "an unresolved device reaches the stream lookup unchanged")

    def test_each_round_gets_its_own_event(self):
        """One event marks one slot. Re-recording a shared event would move the mark of a slot that
        is still out."""
        support = FakeSupport()
        streamer = self.streamer("cuda:0")

        with patch(
            "runai_model_streamer.distributed_streamer.distributed_streamer.device_support",
            return_value=support,
        ):
            first = streamer.slot_is_free()
            second = streamer.slot_is_free()

        self.assertIsNot(first, second)
        self.assertEqual(len(support.events), 2)


if __name__ == "__main__":
    unittest.main()
