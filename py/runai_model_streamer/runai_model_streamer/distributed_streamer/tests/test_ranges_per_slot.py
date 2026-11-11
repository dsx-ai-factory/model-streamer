# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A slot that is sent holds no more ranges than its peers can be told about.

The ring packs by BYTES. A range of four bytes costs a whole metadata row, so a slot of small
tensors packs thousands of them, and the row count a receiver agreed to read is fixed. Without a cap
the batch is refused mid stream - loudly, but the path is unusable for a model of small tensors."""

import unittest

from runai_model_streamer.file_streamer import FileChunks
from runai_model_streamer.file_streamer.requests_iterator import (
    FilesRequestsIterator,
    RANGE_OFFSET_ALIGNMENT,
)


SMALL = 4


def one_file(count, size=SMALL):
    """count ranges of `size` bytes, laid end to end in one file."""
    return [FileChunks(0, "model.bin",
                       [i * size for i in range(count)],
                       [size] * count)]


class RangesPerSlot(unittest.TestCase):
    def ranges_in_first_request(self, files, budget, cap=None):
        iterator = FilesRequestsIterator(budget, files, RANGE_OFFSET_ALIGNMENT, cap)
        request = iterator.next_request()
        return request.num_ranges if request else 0

    def test_without_a_cap_the_bytes_decide(self):
        """The defect, stated as a number: a slot holds as many ranges as fit, which for small
        ranges is the slot size divided by the alignment."""
        budget = RANGE_OFFSET_ALIGNMENT * 300
        self.assertEqual(self.ranges_in_first_request(one_file(1000), budget), 300)

    def test_a_cap_stops_the_packing_by_count(self):
        budget = RANGE_OFFSET_ALIGNMENT * 300
        self.assertEqual(self.ranges_in_first_request(one_file(1000), budget, cap=256), 256)

    def test_the_cap_is_spent_across_the_files_of_one_request(self):
        """Not per file. A peer receives one request, and its rows describe all of it."""
        files = [FileChunks(i, f"model{i}.bin", [0, SMALL], [SMALL, SMALL]) for i in range(5)]
        budget = RANGE_OFFSET_ALIGNMENT * 300
        self.assertEqual(self.ranges_in_first_request(files, budget, cap=3), 3)

    def test_the_bytes_still_decide_when_they_run_out_first(self):
        budget = RANGE_OFFSET_ALIGNMENT * 10
        self.assertEqual(self.ranges_in_first_request(one_file(1000), budget, cap=256), 10)

    def test_the_ranges_left_over_go_to_the_next_request(self):
        """Capped, not dropped."""
        budget = RANGE_OFFSET_ALIGNMENT * 300
        iterator = FilesRequestsIterator(budget, one_file(10), RANGE_OFFSET_ALIGNMENT, 4)
        counts = []
        while True:
            request = iterator.next_request()
            if request is None:
                break
            counts.append(request.num_ranges)

        self.assertEqual(counts, [4, 4, 2])
        self.assertEqual(sum(counts), 10, "every range is still streamed")

    def test_a_cap_below_one_is_refused(self):
        """It would return an empty request, which a caller reads as end of stream."""
        with self.assertRaises(ValueError):
            FilesRequestsIterator(1024, one_file(4), RANGE_OFFSET_ALIGNMENT, 0)


if __name__ == "__main__":
    unittest.main()
