# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from runai_model_streamer.distributed_streamer.distribution_source import SlotRange
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    write_metadata_rows,
)


class TestWriteMetadataRows(unittest.TestCase):
    """The row order is not SlotRange's field order, so it is pinned by number.

    A receiving rank reads these columns to find each range. Swapping size and offset leaves the
    ranges themselves correct and every other test green, and reads the data at wrong addresses."""

    def rows(self, ranges, capacity=8):
        metadata = torch.zeros(capacity, 4, dtype=torch.int64)
        write_metadata_rows(metadata, ranges)
        return metadata

    def test_the_columns_are_file_index_size_offset(self):
        metadata = self.rows([SlotRange(7, 3, 512, 100)])

        self.assertEqual(metadata[1].tolist(), [7, 3, 100, 512])
        self.assertEqual(metadata[1][2].item(), 100, "column 2 is the SIZE")
        self.assertEqual(metadata[1][3].item(), 512, "column 3 is the OFFSET")

    def test_row_zero_carries_the_count(self):
        metadata = self.rows([SlotRange(0, 0, 0, 10), SlotRange(1, 2, 512, 20)])

        self.assertEqual(metadata[0][0].item(), 2)
        self.assertEqual(metadata[1].tolist(), [0, 0, 10, 0])
        self.assertEqual(metadata[2].tolist(), [1, 2, 20, 512])

    def test_an_empty_slot_writes_a_count_of_zero(self):
        metadata = self.rows([])

        self.assertEqual(metadata[0][0].item(), 0)
        self.assertEqual(metadata[1].tolist(), [0, 0, 0, 0], "no row is written")


if __name__ == "__main__":
    unittest.main()
