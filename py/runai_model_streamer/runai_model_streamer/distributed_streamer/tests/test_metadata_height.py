# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The metadata tensor is as tall as the request, so a staged slot ends on its bytes.

It used to be 256 rows whatever was being read, and the staged source was capped to match. A slot
of small ranges then closed at 256 of them with most of its gibibyte still free, and the sizing had
no way to know. The height is a fact about the request now, and the cap is that same number."""

import unittest

import torch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    allocate_dist_staging,
    dist_device_bytes_required,
    metadata_rows,
    write_metadata_rows,
)
from runai_model_streamer.distributed_streamer.distribution_source import SlotRange
from runai_model_streamer.distributed_streamer.staged_source import StagedDistributionSource
from runai_model_streamer.file_streamer.requests_iterator import FileChunks

CPU = torch.device("cpu")


def files(*sizes_per_file):
    return [FileChunks.contiguous(i, f"{i}.bin", 0, list(sizes))
            for i, sizes in enumerate(sizes_per_file)]


class TheHeightFollowsTheRequest(unittest.TestCase):
    def test_it_counts_every_range_of_every_file(self):
        self.assertEqual(metadata_rows(files([1, 2, 3], [4, 5])), 5)

    def test_a_thousand_small_ranges_get_a_thousand_rows(self):
        """The case the fixed 256 could not describe."""
        self.assertEqual(metadata_rows(files([8] * 1000)), 1000)

    def test_an_empty_request_still_gets_one_row(self):
        """torch.zeros(0, 4) is a legal tensor and a broadcast of it carries no count row."""
        self.assertEqual(metadata_rows([]), 1)
        self.assertEqual(metadata_rows(files([])), 1)


class TheTensorIsBuiltThatTall(unittest.TestCase):
    def test_the_metadata_tensors_hold_one_row_per_range_plus_the_count(self):
        _, _, batch, received = allocate_dist_staging(4096, CPU, rows=1000)
        for name, tensor in (("batch", batch), ("received", received)):
            self.assertEqual(tensor.shape, (1001, 4), f"the {name} tensor is the wrong height")

    def test_a_slot_of_a_thousand_ranges_is_described_without_raising(self):
        """write_metadata_rows refuses a batch taller than its tensor, and that refusal is what the
        old cap existed to avoid."""
        _, _, batch, _ = allocate_dist_staging(4096, CPU, rows=1000)
        ranges = [SlotRange(0, i, i * 8, 8) for i in range(1000)]
        write_metadata_rows(batch, ranges)
        self.assertEqual(int(batch[0, 0].item()), 1000)

    def test_the_memory_check_counts_the_same_tensors(self):
        """The gate and the allocation must agree, or it guards a number nothing takes."""
        for rows in (1, 1000):
            with self.subTest(rows=rows):
                tensors = allocate_dist_staging(4096, CPU, rows=rows)
                taken = sum(t.untyped_storage().nbytes() for t in tensors)
                self.assertEqual(dist_device_bytes_required(4096, rows), taken)


class AStagedSlotEndsOnItsBytes(unittest.TestCase):
    """What the height buys. The source is capped at the height, so the cap only ever bites where
    the tensor really could not describe the slot."""

    def fill(self, count, size, capacity, cap):
        ranges = iter([(0, i, torch.zeros(1, size, dtype=torch.uint8)) for i in range(count)])
        originals = {0: {i: (0, i, 0) for i in range(count)}}
        source = StagedDistributionSource(ranges, torch.zeros(capacity, dtype=torch.uint8),
                                          originals, cap)
        return len(source.next_slot().ranges)

    def test_a_thousand_small_ranges_go_in_one_slot(self):
        self.assertEqual(self.fill(1000, 8, 1 << 20, cap=metadata_rows(files([8] * 1000))), 1000)

    def test_the_same_ranges_under_the_old_fixed_cap_stop_at_256(self):
        """Pins that the test above measures the cap and not the capacity: a mebibyte holds all
        thousand either way."""
        self.assertEqual(self.fill(1000, 8, 1 << 20, cap=256), 256)

    def test_bytes_still_end_a_slot_when_they_run_out_first(self):
        """The cap is a backstop, not the rule. Ten ranges of 8 need 80 bytes; a 40 byte buffer
        takes five of them whatever the cap says."""
        self.assertEqual(self.fill(10, 8, 40, cap=metadata_rows(files([8] * 10))), 5)


if __name__ == "__main__":
    unittest.main()
