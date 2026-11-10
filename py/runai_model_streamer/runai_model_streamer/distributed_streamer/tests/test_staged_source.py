# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from runai_model_streamer.distributed_streamer.distribution_source import (
    RANGE_OFFSET_ALIGNMENT,
    next_offset,
)
from runai_model_streamer.distributed_streamer.staged_source import (
    StagedDistributionSource,
)
from runai_model_streamer.distributed_streamer.tests.test_distribution_source import (
    DistributionSourceContract,
)


# Sizes that are NOT multiples of the alignment, so every slot has to pad.
PLAN = [
    (0, 0, 100),
    (0, 1, 200),
    (1, 0, 300),
    (1, 1, 50),
]

# This rank's numbering -> the caller's. Deliberately NOT the identity, and not order preserving, so
# a source that forgot to convert cannot pass by accident.
ORIGINALS = {
    0: {0: (7, 3, 100), 1: (7, 1, 200)},
    1: {0: (2, 0, 300), 1: (7, 2, 50)},
}

CALLER_RANGES = {(7, 3), (7, 1), (2, 0), (7, 2)}


def payload(file_id: int, range_index: int, size: int) -> torch.Tensor:
    """Bytes that say where they came from, so a wrong offset cannot read as correct."""
    first = 1 + file_id * 16 + range_index
    return (torch.arange(size, dtype=torch.int64) + first).remainder(251).to(torch.uint8)


def ranges_of(plan):
    for file_id, range_index, size in plan:
        yield file_id, range_index, payload(file_id, range_index, size).view(1, -1)


def make(plan=PLAN, capacity=4096, max_ranges=8):
    buffer = torch.zeros(capacity, dtype=torch.uint8)
    source = StagedDistributionSource(ranges_of(plan), buffer, ORIGINALS, max_ranges)
    return source, buffer


class TestStagedSourceHonoursTheContract(DistributionSourceContract, unittest.TestCase):
    def make_source(self):
        return make()[0]

    def expected_ranges(self):
        return CALLER_RANGES

    def source_depth(self):
        return StagedDistributionSource.DEPTH


class TestStagedSource(unittest.TestCase):
    def drain(self, source):
        """Every slot, as (ranges, a copy of the bytes). Copied because one buffer is reused."""
        slots = []
        while True:
            filled = source.next_slot()
            if filled is None:
                return slots
            slots.append((list(filled.ranges), filled.memory.clone()))
            source.release(filled)

    def test_the_bytes_land_where_the_ranges_say(self):
        source, _ = make()
        back = {}
        for ranges, memory in self.drain(source):
            for r in ranges:
                back[(r.file_id, r.range_index)] = memory[r.offset : r.offset + r.size]

        self.assertEqual(set(back), CALLER_RANGES)
        for file_id, range_index, size in PLAN:
            caller_file, caller_range, _ = ORIGINALS[file_id][range_index]
            self.assertTrue(
                torch.equal(back[(caller_file, caller_range)], payload(file_id, range_index, size)),
                f"range {file_id}.{range_index} did not read back as itself",
            )

    def test_a_range_that_does_not_fit_waits_for_the_next_slot(self):
        # A 1000-byte buffer holds 100 and 200 (ending at 712) and not the 300 after them, so the
        # split falls inside the plan rather than at its end.
        source, _ = make(capacity=1000)
        slots = self.drain(source)

        self.assertEqual([len(ranges) for ranges, _ in slots], [2, 2])
        self.assertEqual([r.offset for r in slots[0][0]], [0, 512])
        self.assertEqual(len(slots[0][1]), 712)
        self.assertEqual([(r.file_id, r.range_index) for r in slots[1][0]], [(2, 0), (7, 2)])

    def test_a_slot_holds_no_more_ranges_than_it_was_given(self):
        source, _ = make(max_ranges=3)
        self.assertEqual([len(ranges) for ranges, _ in self.drain(source)], [3, 1])

    def test_the_offsets_follow_the_layout_rule(self):
        source, _ = make()
        ranges, memory = self.drain(source)[0]

        # 100 -> pad to 512 -> +200 -> pad to 1024 -> +300 -> pad to 1536 -> +50.
        self.assertEqual([r.offset for r in ranges], [0, 512, 1024, 1536])
        for r in ranges:
            self.assertEqual(r.offset % RANGE_OFFSET_ALIGNMENT, 0)
        self.assertEqual(next_offset(512), 512, "an offset already on the rule is not pushed on")
        self.assertEqual(len(memory), 1586, "the used prefix ends at the last range, not at a pad")

    def test_an_empty_share_answers_none(self):
        source, _ = make(plan=[])
        self.assertIsNone(source.next_slot())

    def test_next_slot_refuses_while_a_slot_is_out(self):
        source, _ = make(max_ranges=2)
        filled = source.next_slot()
        with self.assertRaisesRegex(RuntimeError, "one buffer"):
            source.next_slot()
        source.release(filled)
        self.assertIsNotNone(source.next_slot())


if __name__ == "__main__":
    unittest.main()
