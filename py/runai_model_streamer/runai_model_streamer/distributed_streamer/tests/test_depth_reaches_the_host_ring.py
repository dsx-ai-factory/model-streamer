# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A caller's depth reaches the host ring on the distributed path.

The device path ignores it on purpose - its slot shape is agreed across the group, and a rank that
used its own depth would post a different number of collectives. The host path has no such
agreement, so the caller's number has to arrive. It used to be accepted and dropped, which the
metadata sweeps paid for: they ask for a depth of 1 because each is one 8-byte range per file, and
at the default depth every file becomes its own submission.

The complement is test_depth_is_not_per_rank, which pins that the same number never reaches the
sizing the group agrees on."""

import unittest
from unittest.mock import MagicMock, patch

from runai_model_streamer.distributed_streamer import distributed_streamer as module
from runai_model_streamer.distributed_streamer.distributed_streamer import _distributedStreamer
from runai_model_streamer.file_streamer.requests_iterator import FileChunks


class DepthReachesTheHostRing(unittest.TestCase):
    def host_call(self, ring_buffers):
        """The kwargs stream_files hands FileStreamer when it takes the host path."""
        requests = [FileChunks(id=0, path="m.safetensors", offsets=[0], sizes=[8])]

        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.file_streamer = MagicMock()
        streamer.distribution_group = None
        streamer.rank = 0
        streamer.original_group_rank = 0

        params = MagicMock()
        params.min_device_free_memory = None
        params.num_processes_on_node = 1
        params.max_chunk = None
        params.metadata_rows = None

        span = MagicMock()
        span.partition = [(requests[0], {})]
        span.total_chunks = 1
        span.sizes_by_rank = [8]

        fake_dist = MagicMock()
        fake_dist.get_world_size.return_value = 1
        fake_dist.get_rank.return_value = 0

        with patch.object(module, "dist", fake_dist), \
             patch.object(module, "partition_for_rank", return_value=span), \
             patch.object(module, "agreed_slot_shape", return_value=None), \
             patch.object(module, "device_support", return_value=None), \
             patch.object(module, "log_partition_info", MagicMock()), \
             patch.object(module, "log_rank_span_info", MagicMock()), \
             patch.object(_distributedStreamer, "create_group", lambda self, p: None), \
             patch.object(_distributedStreamer, "set_rank", lambda self: None), \
             patch.object(_distributedStreamer, "check_the_group_agreed", lambda self: None), \
             patch.object(_distributedStreamer, "rank_memory_limit", lambda self: None):
            streamer.stream_files(requests, None, "cpu", params, False,
                                  ring_buffers=ring_buffers)

        streamer.file_streamer.stream_files.assert_called_once()
        return streamer.file_streamer.stream_files.call_args.kwargs

    def test_the_depth_the_caller_asked_for_arrives(self):
        for depth in (1, 4, 16):
            with self.subTest(depth=depth):
                self.assertEqual(self.host_call(depth).get("ring_buffers"), depth)

    def test_no_depth_stays_no_depth(self):
        """None must travel as None, so FileStreamer still chooses the default."""
        self.assertIsNone(self.host_call(None).get("ring_buffers"))


if __name__ == "__main__":
    unittest.main()
