# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.distributed_streamer import _distributedStreamer
from runai_model_streamer.file_streamer.requests_iterator import (
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    DEFAULT_MEMORY_LIMIT_STRING,
)


class TestRankMemoryLimit(unittest.TestCase):
    """RUNAI_STREAMER_MEMORY_LIMIT is a NODE total: the ranks sharing a host share its RAM, so each
    rank gets the total divided by the ranks on THAT host."""

    def streamer(self, num_processes_on_node: int, max_chunk: int = 0) -> _distributedStreamer:
        streamer = _distributedStreamer(None)
        streamer.num_processes_on_node = num_processes_on_node
        streamer.max_chunk = max_chunk
        return streamer

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "8000"})
    def test_divides_the_node_total_between_the_ranks_on_the_node(self):
        self.assertEqual(self.streamer(8).rank_memory_limit(), 1000)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "8000"})
    def test_single_rank_gets_the_whole_total(self):
        self.assertEqual(self.streamer(1).rank_memory_limit(), 8000)

    @patch.dict(os.environ, {}, clear=True)
    def test_the_default_is_a_node_total_too(self):
        # the default has to be divided as well, or an unset variable means 40 GB PER RANK - which is
        # the multiplication this change exists to remove
        self.assertEqual(self.streamer(8).rank_memory_limit(), int(DEFAULT_MEMORY_LIMIT_STRING) // 8)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "8000"})
    def test_largest_range_floors_the_share(self):
        # a rank must be able to buffer one whole tensor or streaming cannot progress; with many ranks
        # on a node the plain share falls below that
        self.assertEqual(self.streamer(8, max_chunk=2500).rank_memory_limit(), 2500)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "-1"})
    def test_unlimited_passes_through(self):
        # -1 and 0 are modes, not quantities: dividing them would turn a mode into nonsense
        self.assertEqual(self.streamer(8).rank_memory_limit(), -1)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "0"})
    def test_largest_chunk_mode_passes_through(self):
        self.assertEqual(self.streamer(8).rank_memory_limit(), 0)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "8000"})
    def test_unknown_rank_count_is_treated_as_one(self):
        # find_local_ranks has not run (or returned nothing): fall back to the whole total rather than
        # dividing by zero
        self.assertEqual(self.streamer(None).rank_memory_limit(), 8000)
        self.assertEqual(self.streamer(0).rank_memory_limit(), 8000)


if __name__ == "__main__":
    unittest.main()
