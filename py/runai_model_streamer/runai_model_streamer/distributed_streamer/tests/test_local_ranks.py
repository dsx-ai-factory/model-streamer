# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import ANY, MagicMock, patch

from runai_model_streamer.distributed_streamer import distributed_streamer
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    DistributedStreamer,
    _distributedStreamerParams,
)
from runai_model_streamer.safetensors_streamer.safetensors_streamer import SafetensorsStreamer
from runai_model_streamer.safetensors_streamer.tests.test_metadata_ordering import write_safetensors


class TestDiscoverRanks(unittest.TestCase):
    """find_local_ranks() is a collective over all the ranks of the process group, which a read with
    discover_ranks=False must not enter."""

    def setUp(self):
        dist = MagicMock()
        dist.is_initialized.return_value = True
        dist.get_world_size.return_value = 8
        dist.get_rank.return_value = 3
        patches = [
            patch.object(distributed_streamer, "dist", dist),
            patch.object(_distributedStreamerParams, "find_local_ranks", return_value=(8, 3, [list(range(8))])),
            patch.dict(os.environ, {}, clear=True),
        ]
        self.find_local_ranks = [p.start() for p in patches][1]
        for p in patches:
            self.addCleanup(p.stop)

    def test_a_read_without_discovery_does_not_guess_the_count(self):
        streamer = DistributedStreamer()
        streamer.file_streamer = MagicMock()
        streamer.stream_files([], None, "cpu", False, discover_ranks=False)

        self.find_local_ranks.assert_not_called()
        self.assertEqual(streamer.params.my_global_rank, 3)
        # the CPP layer keeps 1, since a count larger than the readers on the node lowers their throughput
        streamer.file_streamer.set_process_group_size.assert_not_called()

    def test_safetensors_streamer_discovers_only_for_a_distributed_load(self):
        # a non distributed load may run on a single rank. A distributed load discovers in its first metadata read,
        # which fixes the queue depth of the CPP layer
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        path = os.path.join(temp_dir, "model.safetensors")
        write_safetensors(path, [("t", {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]})], b"\x00\x01\x02\x03")

        for is_distributed, discoveries in ((False, 0), (True, 1)):
            with self.subTest(is_distributed=is_distributed):
                self.find_local_ranks.reset_mock()
                with SafetensorsStreamer() as streamer:
                    streamer.stream_files([path], is_distributed=is_distributed)
                    self.assertEqual(len(list(streamer.get_tensors())), 1)

                self.assertEqual(self.find_local_ranks.call_count, discoveries)

    def test_safetensors_streamer_sets_the_process_count_only_for_a_distributed_load(self):
        # the count divides the queue depth of the CPP layer, which a read without its peers must keep whole
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        path = os.path.join(temp_dir, "model.safetensors")
        write_safetensors(path, [("t", {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]})], b"\x00\x01\x02\x03")

        for is_distributed, count in ((False, None), (True, 8)):
            with self.subTest(is_distributed=is_distributed), patch(
                "runai_model_streamer.file_streamer.file_streamer.runai_set_process_group_size"
            ) as set_count:
                with SafetensorsStreamer() as streamer:
                    streamer.stream_files([path], is_distributed=is_distributed)
                    self.assertEqual(len(list(streamer.get_tensors())), 1)

                if count is None:
                    set_count.assert_not_called()
                else:
                    set_count.assert_called_with(ANY, count)


if __name__ == "__main__":
    unittest.main()
