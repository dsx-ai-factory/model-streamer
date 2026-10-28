# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import random
import unittest
from unittest.mock import patch

from runai_model_streamer.file_streamer.requests_iterator import FileChunks
from runai_model_streamer.file_streamer.yield_order import (
    RUNAI_STREAMER_ORDERED_YIELD_ENV_VAR_NAME,
    UnorderedYield,
    YieldOrder,
    yield_order_for,
)


def files(*range_counts):
    """One FileChunks per count, with ids that do NOT follow the list order.

    Deliberate: the order must come from the position in the list, so an id order that disagrees
    with it is what tells the two apart."""
    return [
        FileChunks(id=100 - position, path=f"f{position}.bin",
                   offsets=[i * 8 for i in range(count)], sizes=[8] * count)
        for position, count in enumerate(range_counts)
    ]


class TestOrdinals(unittest.TestCase):
    def test_a_later_file_starts_after_the_earlier_one(self):
        chunks = files(3, 2)
        order = YieldOrder(chunks)

        first, second = chunks[0].id, chunks[1].id
        self.assertEqual([order.ordinal(first, i) for i in range(3)], [0, 1, 2])
        self.assertEqual([order.ordinal(second, i) for i in range(2)], [3, 4])

    def test_the_file_list_decides_the_order_not_the_ids(self):
        chunks = files(2, 2)
        order = YieldOrder(chunks)

        # ids descend while positions ascend, so ordering by id would reverse the files
        self.assertGreater(chunks[0].id, chunks[1].id)
        self.assertLess(order.ordinal(chunks[0].id, 0), order.ordinal(chunks[1].id, 0))

    def test_a_repeated_file_id_is_rejected(self):
        # Sharing an id would share a base, so one of the two files' responses would overwrite the
        # other's and a tensor would be lost with nothing raised.
        chunks = files(2, 2)
        chunks[1].id = chunks[0].id

        with self.assertRaises(ValueError):
            YieldOrder(chunks)

    def test_a_repeated_file_id_is_rejected_with_ordering_off_too(self):
        """Turning ordering off must not turn the check off with it."""
        chunks = files(2, 2)
        chunks[1].id = chunks[0].id

        with self.assertRaises(ValueError):
            UnorderedYield(chunks)

    def test_a_response_for_a_file_outside_the_stream_is_named(self):
        order = YieldOrder(files(2))
        with self.assertRaises(ValueError) as caught:
            order.ordinal(999, 0)
        self.assertIn("999", str(caught.exception))

    def test_an_empty_file_takes_no_ordinals(self):
        chunks = files(2, 0, 1)
        order = YieldOrder(chunks)
        self.assertEqual(order.ordinal(chunks[2].id, 0), 2)


class TestOrderedYield(unittest.TestCase):
    """Responses go in in some order and come out in ordinal order."""

    def deliver(self, order, chunks, arrival):
        """Feed (file position, range index) pairs in `arrival` order, collect what comes out."""
        out = []
        for position, index in arrival:
            out.extend(order.arrived(chunks[position].id, index, (position, index)))
        return out

    def test_responses_in_order_pass_straight_through(self):
        chunks = files(3)
        order = YieldOrder(chunks)
        arrival = [(0, 0), (0, 1), (0, 2)]
        self.assertEqual(self.deliver(order, chunks, arrival), arrival)

    def test_a_late_response_releases_the_run_parked_behind_it(self):
        chunks = files(4)
        order = YieldOrder(chunks)

        # 1, 2 and 3 all wait for 0
        self.assertEqual(order.arrived(chunks[0].id, 1, "b"), [])
        self.assertEqual(order.arrived(chunks[0].id, 2, "c"), [])
        self.assertEqual(order.arrived(chunks[0].id, 3, "d"), [])
        self.assertEqual(order.arrived(chunks[0].id, 0, "a"), ["a", "b", "c", "d"])

    def test_the_interleave_the_mock_produces_comes_out_sorted(self):
        # Two submissions of four ranges each, served round robin - the real order observed from the
        # mock library: [0, 4, 1, 5, 2, 6, 3, 7]
        chunks = files(8)
        order = YieldOrder(chunks)
        arrival = [(0, i) for i in (0, 4, 1, 5, 2, 6, 3, 7)]
        self.assertEqual(self.deliver(order, chunks, arrival),
                         [(0, i) for i in range(8)])

    def test_every_arrival_order_of_two_files_comes_out_sorted(self):
        """Exhaustive over a small case, so no particular shuffle is being relied on."""
        chunks = files(2, 2)
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for arrival in itertools.permutations(expected):
            with self.subTest(arrival=arrival):
                self.assertEqual(self.deliver(YieldOrder(chunks), chunks, list(arrival)),
                                 expected)

    def test_a_large_shuffle_comes_out_sorted(self):
        chunks = files(40, 35, 25)
        expected = [(p, i) for p, count in enumerate((40, 35, 25)) for i in range(count)]

        arrival = list(expected)
        random.Random(17).shuffle(arrival)        # seeded, so a failure can be repeated

        self.assertEqual(self.deliver(YieldOrder(chunks), chunks, arrival), expected)

    def test_nothing_is_handed_over_while_the_first_range_is_missing(self):
        chunks = files(5)
        order = YieldOrder(chunks)
        for index in range(1, 5):
            self.assertEqual(order.arrived(chunks[0].id, index, index), [])


class TestUnorderedYield(unittest.TestCase):
    def test_every_response_is_handed_over_as_it_arrives(self):
        chunks = files(3)
        order = UnorderedYield(chunks)
        self.assertEqual(order.arrived(chunks[0].id, 2, "c"), ["c"])
        self.assertEqual(order.arrived(chunks[0].id, 0, "a"), ["a"])


class TestYieldOrderFor(unittest.TestCase):
    @patch.dict(os.environ, {RUNAI_STREAMER_ORDERED_YIELD_ENV_VAR_NAME: "false"})
    def test_a_value_that_is_not_an_integer_names_the_variable(self):
        with self.assertRaises(ValueError) as raised:
            yield_order_for(files(3))
        self.assertIn(RUNAI_STREAMER_ORDERED_YIELD_ENV_VAR_NAME, str(raised.exception))
        self.assertIn("'false'", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
