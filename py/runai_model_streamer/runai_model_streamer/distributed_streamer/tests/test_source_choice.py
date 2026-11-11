# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which source the loop builds, for every combination of the two facts that decide it.

The gloo suite can only reach two of the four: device_support answers None on a host, so
sends_directly is never true there. The cell it cannot reach - an empty share on a device the
library can fill - crashed, because that rank has no send buffer to hand a staged source."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
)
from runai_model_streamer.distributed_streamer.distribution_source import (
    EmptyDistributionSource,
)
from runai_model_streamer.distributed_streamer.direct_source import (
    DirectDistributionSource,
)
from runai_model_streamer.distributed_streamer.staged_source import (
    StagedDistributionSource,
)


MODULE = "runai_model_streamer.distributed_streamer.distributed_streamer"


class SourceChoice(unittest.TestCase):
    def build(self, reading_from_storage, sends_directly):
        """Enter get_chunks far enough to build a source, and report which one.

        total_chunks_to_read is 0, so the loop body never runs and no collective is posted: what is
        under test is the setup before it."""
        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.file_streamer = MagicMock()
        streamer.device = torch.device("cpu")
        streamer.max_chunk = 2048
        streamer.rank_dicts_map = {}
        streamer.total_chunks_to_read = 0
        streamer.reading_from_storage = reading_from_storage
        streamer.sends_directly = sends_directly
        streamer.slot_shape = (4096, 3) if sends_directly else None

        built = []
        allocations = []
        real_allocate = __import__(MODULE, fromlist=["allocate_dist_staging"]).allocate_dist_staging

        def record_allocate(*a, **k):
            allocations.append(k)
            return real_allocate(*a, **k)

        with patch(f"{MODULE}.allocate_dist_staging", side_effect=record_allocate), \
             patch(f"{MODULE}.DirectDistributionSource", side_effect=lambda *a, **k: built.append(DirectDistributionSource) or MagicMock()), \
             patch(f"{MODULE}.StagedDistributionSource", side_effect=lambda *a, **k: built.append(StagedDistributionSource) or MagicMock()), \
             patch(f"{MODULE}.EmptyDistributionSource", side_effect=lambda *a, **k: built.append(EmptyDistributionSource) or MagicMock()):
            list(streamer.get_chunks())

        self.assertEqual(len(built), 1, "exactly one source per stream")
        self.asked = allocations[0]
        return built[0]

    def test_a_rank_that_reads_and_can_be_filled_sends_its_ring_slots(self):
        self.assertIs(self.build(True, True), DirectDistributionSource)

    def test_a_rank_that_reads_into_host_memory_stages(self):
        self.assertIs(self.build(True, False), StagedDistributionSource)

    def test_a_rank_with_nothing_to_send_allocates_no_send_buffer(self):
        """It broadcasts a count of zero, so the buffer would be a device allocation nothing ever
        writes to. The direct path has none either, for the same reason."""
        self.build(False, False)
        self.assertFalse(self.asked["send_buffer"], "an empty share stages nothing")

        self.build(True, True)
        self.assertFalse(self.asked["send_buffer"], "a sent ring slot stages nothing")

        self.build(True, False)
        self.assertTrue(self.asked["send_buffer"], "a staging rank needs one")

    def test_the_receive_buffer_is_sized_from_the_agreed_slot(self):
        """A receiver must hold the largest slot any peer will send, which on the direct path is
        the agreed size and not this rank's own staging size."""
        self.build(True, True)
        self.assertEqual(self.asked["receive_bytesize"], 4096)

        self.build(True, False)
        self.assertIsNone(self.asked["receive_bytesize"], "a staging group keeps its own size")

    def test_an_empty_share_sends_nothing_whatever_its_device(self):
        """BOTH cells. The direct one had no send buffer, so building a staged source over it
        raised TypeError on a real GPU - and no test on a host-only runner could reach it."""
        self.assertIs(self.build(False, True), EmptyDistributionSource)
        self.assertIs(self.build(False, False), EmptyDistributionSource)


if __name__ == "__main__":
    unittest.main()
