# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A slot that is never described to anyone is not capped at a row count.

MAX_CHUNKS_PER_BATCH is the height of the metadata tensor. The predicted path sends no metadata -
every rank works the layout out - so capping there would end a slot before its bytes ran out, and
the group would broadcast more slots, each one part empty. Measured on an 80B model: capped, the
ring sends 299 slots at 70% fill where uncapped it sends 209 at 100%.

Both the prediction and the ring must be uncapped, and by the same amount: they are two computations
of one layout, and a peer reads the bytes where the prediction says they are."""

import unittest
from unittest.mock import MagicMock, patch

from runai_model_streamer.distributed_streamer import distributed_streamer as module
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    AgreedRing,
    _distributedStreamer,
)
from runai_model_streamer.file_streamer.requests_iterator import FileChunks


class ThePredictedPathPassesNoCap(unittest.TestCase):
    def run_stream_files(self):
        """Drive stream_files far enough to reach the prediction, and report what it asked for."""
        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.file_streamer = MagicMock()
        streamer.distribution_group = None      # so the agreement check posts nothing
        streamer.group_size = 2
        streamer.original_group_rank = 0
        streamer.local_group_global_ranks = [0, 1]

        params = MagicMock()
        params.min_device_free_memory = 1 << 30

        predicted_calls = []

        def record_prediction(*args, **kwargs):
            predicted_calls.append((args, kwargs))
            return iter(())

        fake_dist = MagicMock()
        fake_dist.get_world_size.return_value = 2
        fake_dist.get_rank.return_value = 0

        # A share with something in it, so the stream reaches the ring instead of leaving at the
        # empty-share return just after the prediction.
        share = MagicMock()
        share.partition = [(FileChunks.contiguous(0, "f.bin", 0, [8, 8]), {})]

        with patch.object(module, "agreed_slot_shape", return_value=AgreedRing(4096, 2, 2)), \
             patch.object(module, "device_support", return_value=MagicMock()), \
             patch.object(module, "predicted_slots", side_effect=record_prediction), \
             patch.object(module, "partition", return_value=[[], []]), \
             patch.object(module, "partition_for_rank", return_value=share), \
             patch.object(module, "dist", fake_dist), \
             patch.object(_distributedStreamer, "create_group", lambda self, p: None):
            try:
                streamer.stream_files([], None, "cpu", params, False)
            except Exception:
                # The stream leaves by the empty-share path right after; not what this pins.
                pass

        self.assertTrue(predicted_calls, "the prediction was never asked for")
        return predicted_calls, streamer.file_streamer.stream_files.call_args

    def test_the_prediction_is_asked_for_without_a_cap(self):
        """The fifth argument is max_ranges_per_slot. None means the bytes decide."""
        calls, _ = self.run_stream_files()
        for args, kwargs in calls:
            self.assertIsNone(args[4], "the prediction was capped")

    def test_it_is_asked_once_per_rank_of_the_group(self):
        """Pins that the test above looked at every rank's plan, not just one."""
        calls, _ = self.run_stream_files()
        self.assertEqual(len(calls), 2)
        self.assertEqual([args[2] for args, _ in calls], [0, 1])

    def test_the_ring_is_built_without_one_either(self):
        """The ring and the prediction are two computations of one layout. A cap on one alone would
        send a peer to offsets that do not exist."""
        _, ring_call = self.run_stream_files()
        self.assertIsNotNone(ring_call, "stream_files never built a ring")
        _, kwargs = ring_call
        self.assertNotIn("max_ranges_per_slot", kwargs)


if __name__ == "__main__":
    unittest.main()
