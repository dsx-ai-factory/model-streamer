# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The group's slot shape never depends on a depth one rank was given.

The shape decides whether a rank sends its ring slots or stages through host memory, and the two
post a different number of collectives per round. A rank that answered differently would wait for
messages its peers never send, and the group would sit there until the timeout."""

import unittest
from unittest.mock import MagicMock, patch

from runai_model_streamer.distributed_streamer import distributed_streamer as module
from runai_model_streamer.distributed_streamer.distributed_streamer import _distributedStreamer


class DepthIsNotPerRank(unittest.TestCase):
    def shape_call(self, ring_buffers):
        """What stream_files hands the sizing, for a caller asking for this depth."""
        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.file_streamer = MagicMock()
        streamer.distribution_group = None

        params = MagicMock()
        params.min_device_free_memory = 1 << 30

        recorded = []

        def record(*args, **kwargs):
            recorded.append((args, kwargs))
            return None          # no shape, so the stream leaves by the shortest path

        fake_dist = MagicMock()
        fake_dist.get_world_size.return_value = 1
        fake_dist.get_rank.return_value = 0

        with patch.object(module, "agreed_slot_shape", side_effect=record), \
             patch.object(module, "device_support", return_value=None), \
             patch.object(module, "dist", fake_dist), \
             patch.object(_distributedStreamer, "create_group", lambda self, p: None):
            try:
                streamer.stream_files([], None, "cpu", params, False, ring_buffers=ring_buffers)
            except Exception:
                # Whatever happens after the sizing is not what this test is about.
                pass

        self.assertTrue(recorded, "the sizing was never called")
        return recorded[0]

    def test_the_depth_never_reaches_the_sizing(self):
        for depth in (1, 4, 16):
            with self.subTest(depth=depth):
                args, kwargs = self.shape_call(depth)
                self.assertNotIn(depth, args[2:])
                self.assertNotIn(depth, kwargs.values())

    def test_two_ranks_asking_for_different_depths_ask_the_sizing_the_same(self):
        """The property the group needs, stated directly."""
        self.assertEqual(self.shape_call(1), self.shape_call(16))
