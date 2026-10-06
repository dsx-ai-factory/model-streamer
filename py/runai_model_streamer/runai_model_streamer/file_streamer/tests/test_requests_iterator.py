# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import unittest
from unittest.mock import patch
from runai_model_streamer.file_streamer import requests_iterator
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunksIterator,
    FilesRequestsIterator,
    FilesRequestsIteratorWithBuffer,
    FileChunks,
    MemoryCapMode,
    RunaiStreamerMemoryLimitException,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
    RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME,
    DIRECT_IO_BLOCK,
    HUGE_PAGE,
    RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME,
    _mapping_memory,
    _max_pads_per_buffer,
    _ring_sizing,
)


class TestRingSizing(unittest.TestCase):
    """The ring's (buffer_size, num_buffers). The DEPTH is configured and the buffer size follows from
    it: B = max(largest_range, budget // N), then N is clamped by what the budget affords and by how
    many requests there will be. Ranges are sized in bytes and the depth is overridden per test so the
    fixtures stay small and readable."""

    def files(self, *sizes_per_file):
        return [
            FileChunks.contiguous(i, f"{i}.txt", 0, list(sizes))
            for i, sizes in enumerate(sizes_per_file)
        ]

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_depth_is_the_target_and_the_buffer_follows(self):
        # budget 400 over 4 buffers -> 100 each, and 500 bytes of data means all 4 get used
        self.assertEqual(_ring_sizing(MemoryCapMode.limited, self.files([100] * 5), 400), (100, 4))

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_the_depth_target_is_honoured(self):
        # same data and budget as above, but asking for 2 buffers gives 2 twice the size - the whole
        # point of the knob is that it decides the shape while the limit decides the memory
        self.assertEqual(_ring_sizing(MemoryCapMode.limited, self.files([100] * 5), 400), (200, 2))

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_largest_range_floors_the_buffer_and_so_caps_the_depth(self):
        # a range carries one destination and cannot span two buffers, so a 250 byte range forces a 250
        # byte buffer even though budget // 4 is 87 - and a 350 byte budget then pays for only one.
        # This is the mechanism that caps reachable depth at budget // largest_range.
        #
        # The limit is the stream size on purpose. A limit ABOVE the stream leaves headroom, and the
        # sizing spends it on bigger buffers (see test_headroom_grows_the_buffer_...) - which escapes
        # the very cap this test is about, so the fixture would no longer be testing its own name.
        self.assertEqual(
            _ring_sizing(MemoryCapMode.limited, self.files([250], [10] * 10), 350), (250, 1)
        )

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_depth_never_exceeds_the_number_of_requests(self):
        # 150 bytes in 50 byte buffers is 3 requests, so the 4th buffer could never hold one
        self.assertEqual(_ring_sizing(MemoryCapMode.limited, self.files([50] * 3), 10_000), (50, 3))

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_a_single_request_stream_gets_a_single_buffer(self):
        # one range, so one request: a ring here would be buffers with nothing to pipeline against.
        # This is what the safetensors metadata reads look like - a handful of bytes, twice per load.
        self.assertEqual(_ring_sizing(MemoryCapMode.limited, self.files([8]), 10_000_000_000), (8, 1))

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_unlimited_budgets_the_whole_stream(self):
        # -1 keeps its documented meaning: the budget is the data, so the ring spans it at the target
        # depth - 700 bytes over 4 buffers of 175
        self.assertEqual(_ring_sizing(MemoryCapMode.unlimited, self.files([100] * 7), None), (175, 4))

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_largest_chunk_mode_stays_a_single_buffer(self):
        # mode 0 exists to mean minimal memory; a ring would defeat the only reason to ask for it
        self.assertEqual(_ring_sizing(MemoryCapMode.largest_chunk, self.files([10, 40], [25]), None), (40, 1))

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_nothing_to_read_gives_one_empty_buffer(self):
        self.assertEqual(_ring_sizing(MemoryCapMode.unlimited, self.files([], []), None), (0, 1))
        self.assertEqual(_ring_sizing(MemoryCapMode.unlimited, self.files([0, 0]), None), (0, 1))

    def test_limit_below_largest_range_is_rejected(self):
        with self.assertRaises(RunaiStreamerMemoryLimitException):
            _ring_sizing(MemoryCapMode.limited, self.files([250]), 100)

    def test_limited_without_a_limit_is_rejected(self):
        with self.assertRaises(RunaiStreamerMemoryLimitException):
            _ring_sizing(MemoryCapMode.limited, self.files([10]), None)

    def test_the_ring_never_exceeds_the_limit(self):
        """The ring never allocates more than the caller asked for.

        This replaced a runtime warning: an earlier draft of the sizing rule had a minimum depth that
        could push N x B past the limit, so it warned instead of preventing it. Nothing in the code now
        can produce that, by two separate arguments - depth is clamped by budget // buffer_size on the
        ordinary path, and the span search is capped at limit // target - which is exactly why it is
        worth asserting as a property rather than trusting either argument. Covers the shapes that used
        to trip it: a limit far below the data, a limit equal to one range, awkward divisors."""
        for depth in ("1", "2", "3", "4", "16"):
            for sizes, limit in (
                ([10] * 20, 60),        # limit far below the data - the old warning case
                ([10] * 20, 10),        # limit exactly one range: one buffer, no more
                ([7] * 13, 100),        # nothing divides evenly
                ([250], 250),           # a single range exactly filling the limit
                ([1] * 100, 999_999),   # limit far above the data
            ):
                with patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: depth}):
                    buffer_size, num_buffers = _ring_sizing(
                        MemoryCapMode.limited, self.files(sizes), limit
                    )
                self.assertLessEqual(
                    buffer_size * num_buffers, limit,
                    f"depth={depth} sizes={sizes[:3]}... limit={limit} -> "
                    f"{num_buffers} x {buffer_size}",
                )
                self.assertGreaterEqual(num_buffers, 1)

    def requests_to_stream(self, files, buffer_size):
        """How many requests the real packer needs - the thing the sizing is trying to predict."""
        iterator = FilesRequestsIterator(buffer_size, files)
        requests = 0
        while iterator.next_request() is not None:
            requests += 1
        return requests

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_headroom_grows_the_buffer_so_the_ring_spans_the_stream(self):
        # 7 ranges of 30 against a limit far above the 210 byte stream. budget // 4 is 52, which holds
        # only ONE range (two would be 60), so the stream needs 7 requests behind a ring of 4 and the
        # tail waits for a buffer to come back. This is Llama-3-8B in miniature: 4 buffers sized to
        # exactly total // 4 stranded a 5th request, because tensors are indivisible and B left no room
        # for the waste. The limit has room, so the buffer grows until the ring spans the whole stream.
        buffer_size, num_buffers = _ring_sizing(MemoryCapMode.limited, self.files([30] * 7), 1000)
        self.assertEqual((buffer_size, num_buffers), (60, 4))
        self.assertLessEqual(buffer_size * num_buffers, 1000)
        # the point of the exercise: every request has its own buffer, so nothing is ever recycled
        self.assertEqual(self.requests_to_stream(self.files([30] * 7), buffer_size), num_buffers)

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_a_binding_limit_is_left_alone(self):
        # The same stream under a limit BELOW it. There is no headroom to spend, recycling is the whole
        # point, and the sizing must not go walking the ranges - a 150k tensor model would pay for it.
        self.assertEqual(_ring_sizing(MemoryCapMode.limited, self.files([30] * 7), 120), (30, 4))

    def test_the_span_search_is_skipped_whenever_there_is_no_headroom(self):
        """The search must not RUN when the stream is not smaller than the limit - not merely return
        the same answer.

        Asserting the answer is not enough: with a binding limit the search would find nothing and fall
        back, so every outcome-based test still passes with the guard deleted (verified). The guard is
        there for COST - a model with 150k tensors would pay for a prefix sum over every one of them to
        decide the shape of a four buffer ring - so the call itself is what has to be asserted.
        """
        cases = [
            ("limit below the stream",  MemoryCapMode.limited,       210 // 2),
            ("limit equal to the stream", MemoryCapMode.limited,     210),
            ("unlimited",               MemoryCapMode.unlimited,     -1),
            ("largest_chunk",           MemoryCapMode.largest_chunk, 0),
        ]
        for name, mode, limit in cases:
            with self.subTest(name):
                with patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"}):
                    with patch(
                        "runai_model_streamer.file_streamer.requests_iterator._span_whole_stream"
                    ) as span:
                        _ring_sizing(mode, self.files([30] * 7), limit)
                span.assert_not_called()

        # ... and it DOES run when there is headroom, so the assertions above cannot pass vacuously
        with patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"}):
            with patch(
                "runai_model_streamer.file_streamer.requests_iterator._span_whole_stream"
            ) as span:
                span.return_value = None
                _ring_sizing(MemoryCapMode.limited, self.files([30] * 7), 1000)
        span.assert_called_once()

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_headroom_too_small_to_help_changes_nothing(self):
        # A limit only just above the stream: 220 // 4 = 55 still holds one range, so no reachable
        # buffer size spans the stream in 4 requests. Growing to 55 would cost memory and fix nothing,
        # so the sizing keeps the ordinary answer and recycles.
        self.assertEqual(_ring_sizing(MemoryCapMode.limited, self.files([30] * 7), 220), (52, 4))

    def test_growth_never_exceeds_the_limit(self):
        """The invariant the growth must not break: the ring is still bounded by what was asked for."""
        for depth in ("1", "2", "4", "8"):
            for sizes, limit in (
                ([30] * 7, 1000),          # generous headroom - growth applies
                ([30] * 7, 220),           # marginal headroom
                ([7] * 13, 5_000),         # nothing divides evenly
                ([1000], 1_000_000),       # one range, enormous limit
                ([3] * 100, 100_000),
            ):
                with patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: depth}):
                    buffer_size, num_buffers = _ring_sizing(
                        MemoryCapMode.limited, self.files(sizes), limit
                    )
                self.assertLessEqual(
                    buffer_size * num_buffers, limit,
                    f"depth={depth} sizes={sizes[:3]}... limit={limit} -> {num_buffers} x {buffer_size}",
                )
                # and the ring never has more buffers than there are requests to put in them
                self.assertLessEqual(
                    num_buffers, self.requests_to_stream(self.files(sizes), buffer_size)
                )

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "0"})
    def test_zero_ring_buffers_still_yields_a_ring(self):
        # a hostile 0 must not produce a pool with no buffers to hand out
        _, num_buffers = _ring_sizing(MemoryCapMode.largest_chunk, self.files([10]), None)
        self.assertEqual(num_buffers, 1)
        _, num_buffers = _ring_sizing(MemoryCapMode.unlimited, self.files([10] * 5), None)
        self.assertGreaterEqual(num_buffers, 1)


class TestFileChunks(unittest.TestCase):
    def test_contiguous_computes_offsets(self):
        file_chunks = FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4])
        self.assertEqual(file_chunks.offsets, [10, 11, 13, 16])
        self.assertEqual(file_chunks.sizes, [1, 2, 3, 4])
        self.assertEqual(file_chunks.total_size(), 10)
        self.assertEqual(file_chunks.max_chunk_size(), 4)

    def test_contiguous_with_zero_sized_range(self):
        # a zero sized range does not advance the offset, so it shares the next range's offset
        file_chunks = FileChunks.contiguous(17, "a.txt", 10, [5, 0, 3])
        self.assertEqual(file_chunks.offsets, [10, 15, 15])

    def test_scattered_ranges_are_kept_as_given(self):
        # ranges need not be contiguous, ordered, or even distinct - nothing is derived from them
        file_chunks = FileChunks(17, "a.txt", [100, 5, 60], [4, 4, 4])
        self.assertEqual(file_chunks.offsets, [100, 5, 60])
        self.assertEqual(file_chunks.total_size(), 12)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            FileChunks(17, "a.txt", [0, 10], [4])


class TestFileChunksIterator(unittest.TestCase):
    # The budget tests below covered ChunksIterator before it was folded into FileChunksIterator:
    # ranges carry their own offsets now, so there is no separate size-only iterator to drive.
    def test_next_request_exact_limit(self):
        file_chunks_iterator = FileChunksIterator(FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]))

        file_chunks = file_chunks_iterator.next_chunks(6)
        self.assertFalse(file_chunks_iterator.is_finished())
        self.assertIsNotNone(file_chunks)
        self.assertEqual(file_chunks.id, 17)
        self.assertEqual(file_chunks.path, "a.txt")
        self.assertEqual(file_chunks.offsets, [10, 11, 13])
        self.assertEqual(file_chunks.sizes, [1, 2, 3])

        file_chunks = file_chunks_iterator.next_chunks(4)
        self.assertTrue(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.offsets, [16])
        self.assertEqual(file_chunks.sizes, [4])

    def test_next_request_high_limit(self):
        file_chunks_iterator = FileChunksIterator(FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]))

        file_chunks = file_chunks_iterator.next_chunks(6)
        self.assertFalse(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.sizes, [1, 2, 3])

        file_chunks = file_chunks_iterator.next_chunks(6)
        self.assertTrue(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.sizes, [4])

    def test_next_request_low_limit(self):
        file_chunks_iterator = FileChunksIterator(FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]))

        file_chunks = file_chunks_iterator.next_chunks(6)
        self.assertFalse(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.sizes, [1, 2, 3])

        # the next range does not fit the remaining budget: nothing is handed out, and the file is
        # not finished - the caller must offer it again with a fresh budget
        file_chunks = file_chunks_iterator.next_chunks(2)
        self.assertFalse(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.sizes, [])
        self.assertEqual(file_chunks.offsets, [])

    def test_next_request_with_item_zero(self):
        file_chunks_iterator = FileChunksIterator(FileChunks.contiguous(17, "a.txt", 10, [1, 0, 3, 4]))

        file_chunks = file_chunks_iterator.next_chunks(4)
        self.assertFalse(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.sizes, [1, 0, 3])
        self.assertEqual(file_chunks.offsets, [10, 11, 11])

        file_chunks = file_chunks_iterator.next_chunks(4)
        self.assertTrue(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.sizes, [4])

    def test_next_request_offset_for_non_exact_memory_limit(self):
        file_chunks_iterator = FileChunksIterator(FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]))

        file_chunks_iterator.next_chunks(5)
        file_chunks = file_chunks_iterator.next_chunks(6)
        self.assertFalse(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.offsets, [13])

    def test_scattered_ranges_survive_slicing(self):
        file_chunks_iterator = FileChunksIterator(FileChunks(17, "a.txt", [100, 5, 60], [4, 4, 4]))

        file_chunks = file_chunks_iterator.next_chunks(8)
        self.assertEqual(file_chunks.offsets, [100, 5])
        self.assertEqual(file_chunks.sizes, [4, 4])

        file_chunks = file_chunks_iterator.next_chunks(8)
        self.assertTrue(file_chunks_iterator.is_finished())
        self.assertEqual(file_chunks.offsets, [60])


class TestFilesRequestsIterator(unittest.TestCase):
    def test_next_request_single_file(self):
        files_requests_iterator = FilesRequestsIterator(5, [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4])])

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 1)
        self.assertEqual(files_requests.files[0].id, 17)
        self.assertEqual(files_requests.files[0].path, "a.txt")
        self.assertEqual(files_requests.files[0].offsets, [10, 11])
        self.assertEqual(files_requests.files[0].sizes, [1, 2])

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 1)
        self.assertEqual(files_requests.files[0].offsets, [13])
        self.assertEqual(files_requests.files[0].sizes, [3])

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 1)
        self.assertEqual(files_requests.files[0].offsets, [16])
        self.assertEqual(files_requests.files[0].sizes, [4])

        files_requests = files_requests_iterator.next_request()
        self.assertIsNone(files_requests)

    def test_next_request_multi_file(self):
        files_requests_iterator = FilesRequestsIterator(
            7,
            [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]),
             FileChunks.contiguous(18, "b.txt", 10, [1, 2, 3, 4])],
        )

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 1)
        self.assertEqual(files_requests.files[0].id, 17)
        self.assertEqual(files_requests.files[0].offsets, [10, 11, 13])
        self.assertEqual(files_requests.files[0].sizes, [1, 2, 3])

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 2)
        self.assertEqual(files_requests.files[0].id, 17)
        self.assertEqual(files_requests.files[0].offsets, [16])
        self.assertEqual(files_requests.files[0].sizes, [4])
        self.assertEqual(files_requests.files[1].id, 18)
        self.assertEqual(files_requests.files[1].path, "b.txt")
        self.assertEqual(files_requests.files[1].offsets, [10, 11])
        self.assertEqual(files_requests.files[1].sizes, [1, 2])
        # file_base makes (file index, range index) -> flat position O(1)
        self.assertEqual(files_requests.file_base, [0, 1])
        self.assertEqual(files_requests.num_ranges, 3)
        self.assertEqual(files_requests.flat_index(1, 1), 2)

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 1)
        self.assertEqual(files_requests.files[0].id, 18)
        self.assertEqual(files_requests.files[0].offsets, [13, 16])
        self.assertEqual(files_requests.files[0].sizes, [3, 4])

        files_requests = files_requests_iterator.next_request()
        self.assertIsNone(files_requests)

    def test_next_request_multi_file_block_on_file(self):
        files_requests_iterator = FilesRequestsIterator(
            5,
            [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]),
             FileChunks.contiguous(18, "b.txt", 10, [1, 2, 3, 4])],
        )

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 1)
        self.assertEqual(files_requests.files[0].id, 17)
        self.assertEqual(files_requests.files[0].sizes, [1, 2])

        files_requests = files_requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(len(files_requests.files), 1)
        self.assertEqual(files_requests.files[0].id, 17)
        self.assertEqual(files_requests.files[0].sizes, [3])

    def test_get_global_file_and_range(self):
        files_requests_iterator = FilesRequestsIterator(3, [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4])])

        first = files_requests_iterator.next_request()

        file_id, range_index = files_requests_iterator.get_global_file_and_range(first, 0, 0)
        self.assertEqual(file_id, 17)
        self.assertEqual(range_index, 0)

        file_id, range_index = files_requests_iterator.get_global_file_and_range(first, 0, 1)
        self.assertEqual(file_id, 17)
        self.assertEqual(range_index, 1)

        second = files_requests_iterator.next_request()

        file_id, range_index = files_requests_iterator.get_global_file_and_range(second, 0, 0)
        self.assertEqual(file_id, 17)
        self.assertEqual(range_index, 2)

        # The earlier request is still resolvable now that a later one has been built - each request
        # carries its own range_base rather than reading iterator state that has moved on. This is what
        # allows a response to arrive while a subsequent submission is already in flight.
        self.assertEqual(files_requests_iterator.get_global_file_and_range(first, 0, 1), (17, 1))

    def test_file_chunks_zero_chunks(self):
        requests_iterator = FilesRequestsIterator(10, [FileChunks(17, "a.txt", [], [])])

        res = requests_iterator.next_request()
        self.assertIsNone(res)

    def test_next_request_empty_file_first_does_not_terminate(self):
        # regression for issue #157: an empty shard at the head of the queue used to close the request,
        # which get_chunks reads as end of stream, silently dropping every remaining file
        requests_iterator = FilesRequestsIterator(
            100, [FileChunks(17, "empty.txt", [], []), FileChunks.contiguous(18, "a.txt", 10, [5])]
        )

        files_requests = requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual([f.id for f in files_requests.files], [18])
        self.assertEqual(files_requests.files[0].sizes, [5])

        self.assertIsNone(requests_iterator.next_request())

    def test_next_request_empty_files_between_data_files(self):
        requests_iterator = FilesRequestsIterator(
            100,
            [
                FileChunks.contiguous(17, "a.txt", 10, [1, 2]),
                FileChunks(18, "empty1.txt", [], []),
                FileChunks(19, "empty2.txt", [], []),
                FileChunks.contiguous(20, "b.txt", 10, [3]),
            ],
        )

        files_requests = requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual([f.id for f in files_requests.files], [17, 20])

        self.assertIsNone(requests_iterator.next_request())

    def test_file_of_only_zero_sized_ranges_is_streamed(self):
        # A file whose tensors are all zero sized is NOT an empty shard: it has header entries, and
        # safetensors yields those tensors (shape preserved, data_offsets [x, x]). They are submitted
        # so the C++ layer answers one response per range and the caller's tensor indexing lines up.
        requests_iterator = FilesRequestsIterator(
            100,
            [FileChunks.contiguous(17, "zeros.txt", 10, [0, 0]),
             FileChunks.contiguous(18, "a.txt", 10, [5])],
        )

        files_requests = requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual([f.id for f in files_requests.files], [17, 18])
        self.assertEqual(files_requests.files[0].sizes, [0, 0])
        self.assertEqual(files_requests.num_ranges, 3)

        self.assertIsNone(requests_iterator.next_request())

    def test_zero_sized_range_among_non_zero(self):
        requests_iterator = FilesRequestsIterator(100, [FileChunks.contiguous(17, "a.txt", 10, [5, 0, 3])])

        files_requests = requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(files_requests.files[0].sizes, [5, 0, 3])
        self.assertEqual(files_requests.files[0].offsets, [10, 15, 15])
        self.assertEqual(files_requests.num_ranges, 3)

    def test_only_zero_sized_ranges_in_the_whole_request(self):
        # every range is zero sized, so the memory limit is 0 - the request must still be produced,
        # with one range (and so one response) each
        requests_iterator = FilesRequestsIterator(0, [FileChunks.contiguous(17, "zeros.txt", 10, [0, 0, 0])])

        files_requests = requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(files_requests.num_ranges, 3)

        self.assertIsNone(requests_iterator.next_request())


class TestFilesRequestsIteratorWithBuffer(unittest.TestCase):
    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "1"})
    def test_memory_cap_unlimited(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.unlimited, [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4])], 100
        )
        self.assertEqual(requests_iterator.buffer_size, 10)

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "1"})
    def test_memory_cap_limited(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.limited, [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4])], 5
        )
        self.assertEqual(requests_iterator.buffer_size, 5)

    def test_memory_cap_largest_chunk(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk, [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4])], 5
        )
        self.assertEqual(requests_iterator.buffer_size, 4)

    def test_memory_cap_largest_chunk_multi_file(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk,
            [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]),
             FileChunks.contiguous(18, "b.txt", 10, [1, 2, 7, 4])],
            5,
        )
        self.assertEqual(requests_iterator.buffer_size, 7)

    def test_memory_cap_largest_chunk_no_files(self):
        # the largest_chunk branch used to call a bare max() and raise on an empty sequence, while the
        # limited branch guarded with default=0. Both go through _largest_range now.
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk, [], None
        )
        self.assertEqual(requests_iterator.buffer_size, 0)
        self.assertIsNone(requests_iterator.next_request())

    def test_memory_cap_largest_chunk_empty_shard(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk,
            [FileChunks(17, "empty.txt", [], []), FileChunks.contiguous(18, "a.txt", 10, [4])],
            None,
        )
        self.assertEqual(requests_iterator.buffer_size, 4)

    def test_memory_cap_largest_chunk_only_zero_sized_ranges(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk, [FileChunks.contiguous(17, "zeros.txt", 10, [0, 0])], None
        )
        self.assertEqual(requests_iterator.buffer_size, 0)

        files_requests = requests_iterator.next_request()
        self.assertIsNotNone(files_requests)
        self.assertEqual(files_requests.num_ranges, 2)

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "1"})
    def test_limited_memory_cap_and_smaller_chunks(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.limited,
            [FileChunks.contiguous(17, "a.txt", 10, [1, 2]), FileChunks.contiguous(18, "b.txt", 10, [3, 4])],
            50,
        )
        self.assertEqual(requests_iterator.buffer_size, 10)

    def assert_congruent(self, request):
        """Every range's address and its file offset leave the same remainder for the block size.

        This is the property a direct read needs. Aligning the address alone is not enough: the file
        offset cannot be chosen, so the address has to be moved to match it.
        """
        for file_index, file_chunks in enumerate(request.files):
            for range_index, offset in enumerate(file_chunks.offsets):
                dst = request.range_dsts[request.flat_index(file_index, range_index)]
                self.assertEqual(
                    dst % DIRECT_IO_BLOCK, offset % DIRECT_IO_BLOCK,
                    f"file {file_chunks.path} range {range_index} at offset {offset} is not congruent",
                )

    def test_range_dsts_place_the_request_congruently(self):
        # replaces the old per-file file_buffers: destinations are per range now, one per range across
        # the whole request, handed to runai_request as absolute addresses.
        #
        # They are NOT packed back to back. A range is pushed forward so that its address and its file
        # offset leave the same remainder - the condition for reading it directly.
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk,
            [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4]),
             FileChunks.contiguous(18, "b.txt", 10, [1, 2, 7, 4])],
            5,
        )
        self.assertEqual(requests_iterator.buffer_size, 7)
        base = requests_iterator.buffer_addresses[0]

        files_requests = requests_iterator.next_request()
        self.assertEqual([f.id for f in files_requests.files], [17])
        # a.txt starts at file offset 10, so the first range skips 10 bytes to match it. The two after
        # it follow back to back in the file, so the cursor stays in step and costs nothing.
        self.assertEqual(files_requests.range_dsts, [base + 10, base + 11, base + 13])
        self.assert_congruent(files_requests)

        # largest_chunk mode is a ring of one, so the single buffer has to come back before the next
        # request can be built - and the next request then places from that same base.
        requests_iterator.release(files_requests)

        files_requests = requests_iterator.next_request()
        self.assertEqual([f.id for f in files_requests.files], [17, 18])
        # a.txt's remaining 4 bytes at file offset 16, then b.txt's 1 and 2 at file offsets 10 and 11.
        # Crossing into b.txt sends the file offset BACKWARDS (16 -> 10) while the cursor only moves
        # forward, so re-syncing costs nearly a whole block. That is the worst case, and it is why the
        # slot reserves room for many pads rather than a few bytes.
        # Expressed in terms of the block, not as 4106: the re-sync lands at the next address
        # congruent with file offset 10, which is 10 + one whole block past the base. Hardcoding it
        # tied this test to a 4096 block, and it broke the moment the block was raised.
        resync = 10 + DIRECT_IO_BLOCK
        self.assertEqual(files_requests.range_dsts, [base + 16, base + resync, base + resync + 1])
        self.assert_congruent(files_requests)
        self.assertEqual(files_requests.file_base, [0, 1])

    @patch.dict(os.environ, {RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME: "0"})
    def test_no_pad_budget_falls_back_to_tight_packing(self):
        # With no room reserved, the aligned placement cannot fit and the request packs back to back
        # instead. Correct, just not readable directly - the reader falls back to a buffered read.
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk,
            [FileChunks.contiguous(17, "a.txt", 10, [1, 2, 3, 4])],
            5,
        )
        base = requests_iterator.buffer_addresses[0]

        files_requests = requests_iterator.next_request()
        self.assertEqual(files_requests.range_dsts, [base, base + 1])

        # and the fallback really is the reason: a.txt starts at file offset 10, so a placement that
        # had any room would have moved the first range 10 bytes in rather than leaving it at the base
        self.assertNotEqual(base % DIRECT_IO_BLOCK, files_requests.files[0].offsets[0] % DIRECT_IO_BLOCK)

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "1"})
    def test_get_global_file_and_range_aliases_the_buffer(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.unlimited,
            [FileChunks.contiguous(17, "a.txt", 10, [1, 2]), FileChunks.contiguous(18, "b.txt", 10, [3, 4])],
            5,
        )
        self.assertEqual(requests_iterator.buffer_size, 10)

        request = requests_iterator.next_request()

        # Write at the address the request will hand to runai_request, then read it back through the
        # view. That is the invariant: the two must agree. Deriving the index from range_dsts rather
        # than hardcoding it keeps this test about aliasing, not about where placement chose to put
        # things - test_range_dsts_place_the_request_congruently covers that.
        base = requests_iterator.buffer_addresses[0]
        requests_iterator.buffers[0][request.range_dsts[request.flat_index(0, 0)] - base] = 9
        requests_iterator.buffers[0][request.range_dsts[request.flat_index(1, 0)] - base] = 8

        file_id, range_index, view = requests_iterator.get_global_file_and_range(request, 0, 0)
        self.assertEqual((file_id, range_index), (17, 0))
        self.assertEqual(len(view), 1)
        self.assertEqual(view[0], 9)

        file_id, range_index, view = requests_iterator.get_global_file_and_range(request, 1, 0)
        self.assertEqual((file_id, range_index), (18, 0))
        self.assertEqual(len(view), 3)
        self.assertEqual(view[0], 8)

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_consecutive_requests_get_distinct_buffers(self):
        # 3 ranges of 4 bytes and an 8 byte budget over 2 buffers: 4 bytes each, one range per request,
        # and the second request must NOT land in the first one's buffer - that is the whole point of
        # the pool.
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.limited, [FileChunks.contiguous(17, "a.txt", 10, [4, 4, 4])], 8
        )
        self.assertEqual((requests_iterator.buffer_size, requests_iterator.num_buffers), (4, 2))

        first = requests_iterator.next_request()
        first_index = first.buffer_index
        second = requests_iterator.next_request()
        self.assertNotEqual(first_index, second.buffer_index)
        self.assertNotEqual(first.range_dsts, second.range_dsts)

        # exhausted: a third request has nowhere to go until one comes back
        self.assertFalse(requests_iterator.has_free_buffer())
        with self.assertRaises(RuntimeError):
            requests_iterator.next_request()

        # released, so the third request reuses the first one's buffer and places from its base. The
        # range it carries sits at file offset 18 (10 + 4 + 4), so it is placed 18 bytes in.
        requests_iterator.release(first)
        third = requests_iterator.next_request()
        self.assertEqual(third.buffer_index, first_index)
        self.assertEqual(third.range_dsts, [requests_iterator.buffer_addresses[first_index] + 18])

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_releasing_twice_is_rejected(self):
        # a double release would put the same buffer in the free list twice, so two live requests would
        # silently share it and overwrite each other's data
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.limited, [FileChunks.contiguous(17, "a.txt", 10, [4, 4])], 8
        )
        request = requests_iterator.next_request()
        requests_iterator.release(request)
        with self.assertRaises(ValueError):
            requests_iterator.release(request)

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "3"})
    def test_buffers_are_distinct_slices_of_one_allocation(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.limited, [FileChunks.contiguous(17, "a.txt", 10, [4, 4, 4])], 12
        )
        self.assertEqual(len(requests_iterator.buffers), 3)

        # A slot is the caller's bytes plus room for the pads that make a direct read possible, so the
        # pool is bigger than the memory cap asked for. That overshoot is fixed per slot, not
        # proportional, so it disappears next to a real ring.
        slot_size = requests_iterator.buffer_size + DIRECT_IO_BLOCK * _max_pads_per_buffer()
        self.assertEqual(len(requests_iterator.pool), slot_size * 3)

        # The base must be block aligned. Without it no range in any slot could be placed congruently,
        # because a pad can only move an address forward within the block it already sits in.
        self.assertEqual(requests_iterator.pool.ctypes.data % DIRECT_IO_BLOCK, 0)
        self.assertEqual(
            requests_iterator.buffer_addresses,
            [requests_iterator.pool.ctypes.data + slot_size * i for i in range(3)],
        )

        # writing through one buffer must not disturb another, and each view must alias the pool
        for index, buffer in enumerate(requests_iterator.buffers):
            buffer[:] = index
        for index in range(3):
            # first and last byte of each slot, rather than the whole pool - the slots are megabytes
            self.assertEqual(requests_iterator.pool[slot_size * index], index)
            self.assertEqual(requests_iterator.pool[slot_size * (index + 1) - 1], index)

    def test_end_of_stream_does_not_consume_a_buffer(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.largest_chunk, [], None
        )
        self.assertIsNone(requests_iterator.next_request())
        self.assertTrue(requests_iterator.has_free_buffer())

    @patch.dict(os.environ, {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "1"})
    def test_get_global_file_and_range_zero_sized_range(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.unlimited, [FileChunks.contiguous(17, "a.txt", 10, [5, 0, 3])], None
        )

        request = requests_iterator.next_request()

        file_id, range_index, view = requests_iterator.get_global_file_and_range(request, 0, 1)
        self.assertEqual((file_id, range_index), (17, 1))
        self.assertEqual(len(view), 0)

        # the zero sized range consumes no buffer, so the range after it starts where it did
        _, _, view = requests_iterator.get_global_file_and_range(request, 0, 2)
        self.assertEqual(len(view), 3)


def _resident_bytes(address: int, length: int) -> int:
    """How many bytes of [address, address+length) the kernel currently holds in memory.

    mincore(2), so it is the kernel's own answer for the same range that _mapping_memory reads out of
    /proc/self/smaps. Two independent routes to one number: if they disagree, the text parsing is
    wrong, and if they agree the machine simply reclaimed the pages."""
    import ctypes

    page = os.sysconf("SC_PAGESIZE")
    start = address - (address % page)
    span = length + (address - start)
    pages = (span + page - 1) // page

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    vec = (ctypes.c_ubyte * pages)()
    if libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(span), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")

    return sum(1 for byte in vec if byte & 1) * page


class TestPoolHugePages(unittest.TestCase):
    """The pool base and the huge-page hint.

    Whether the kernel actually gives huge pages depends on the machine, so nothing here asserts
    that. What is asserted is what we control: the base is placed correctly, and asking for huge
    pages never breaks a load however the host is configured."""

    def _pool(self, num_buffers=2, buffer_size=4 * 1024 * 1024):
        return FilesRequestsIteratorWithBuffer(
            buffer_size=buffer_size,
            num_buffers=num_buffers,
            files_chunks=[FileChunks.contiguous(0, "a.bin", 0, [1024, 1024])],
        )

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "1"})
    def test_pool_base_is_huge_page_aligned(self):
        # A 2 MiB base is what lets the whole pool sit on huge pages. It is also a multiple of
        # DIRECT_IO_BLOCK, so it replaces the old 4096 shift rather than adding to it.
        self.assertEqual(self._pool().pool.ctypes.data % HUGE_PAGE, 0)

    def test_every_slot_base_is_block_aligned(self):
        # Follows from the base only because the slot size is a multiple of the block. If that ever
        # stops holding, direct reads into later slots lose congruence while the first slot keeps it
        # - which would look like a bug in one file rather than in the pool.
        pool = self._pool(num_buffers=4)
        for address in pool.buffer_addresses:
            self.assertEqual(address % DIRECT_IO_BLOCK, 0)

    def test_the_pool_still_holds_every_buffer(self):
        # The base shift grew from 4096 to 2 MiB, so the over-allocation had to grow with it.
        pool = self._pool(num_buffers=3)
        self.assertEqual(len(pool.buffers), 3)
        last = pool.buffers[-1]
        self.assertEqual(len(last), pool._slot_size)

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "0"})
    def test_switching_huge_pages_off_restores_the_old_alignment(self):
        # Off must mean exactly what it meant before huge pages existed: a block-aligned base, no
        # madvise, no report. A switch that still did the work and only stayed quiet would be no use
        # as a rollback.
        pool = self._pool()
        self.assertEqual(pool.pool.ctypes.data % DIRECT_IO_BLOCK, 0)
        self.assertTrue(pool._huge_pages_reported, "nothing to report when the feature is off")

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "0"})
    def test_switching_off_still_aligns_every_slot(self):
        # Congruent placement depends on this, so the direct-read path must keep working with the
        # feature off. Otherwise the switch would quietly disable direct reads as well.
        pool = self._pool(num_buffers=4)
        for address in pool.buffer_addresses:
            self.assertEqual(address % DIRECT_IO_BLOCK, 0)

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "0"})
    def test_the_report_is_skipped_when_switched_off(self):
        pool = self._pool()
        with patch(
            "runai_model_streamer.file_streamer.requests_iterator._mapping_memory"
        ) as measured:
            request = pool.next_request()
            pool.release(request)
        measured.assert_not_called()

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "yes"})
    def test_a_value_that_is_not_one_turns_it_off(self):
        # A typo must be safe rather than surprising, so only "1" enables it.
        self.assertFalse(requests_iterator._huge_pages_enabled())

    def test_huge_pages_are_off_by_default(self):
        # Opt in, not opt out. The gain is large and measured (see _huge_pages_enabled), but the risk -
        # direct compaction at fault time on a fragmented node - is not, so the default stays off until
        # someone measures the bad case.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME, None)
            self.assertFalse(requests_iterator._huge_pages_enabled())

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "1"})
    def test_the_default_pool_alignment_is_a_block(self):
        # With the feature off, the pool must be laid out exactly as it was before huge pages existed.
        # Checked here as well as through _huge_pages_enabled, because the alignment is what a direct
        # read actually depends on.
        with patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "0"}):
            pool = self._pool()
            self.assertEqual(pool.pool.ctypes.data % DIRECT_IO_BLOCK, 0)
            self.assertNotEqual(requests_iterator._pool_alignment(), HUGE_PAGE)

    def test_the_pad_budget_does_not_grow_with_huge_pages(self):
        """A slot reserves room for pads at DIRECT_IO_BLOCK each, whatever the page size.

        The trap this guards: if congruent placement had to hold modulo 2 MiB, one pad would cost
        2 MiB and the 1024-pad budget would want 2 GB per buffer instead of 4 MB.

        It does not, because O_DIRECT alignment comes from the storage device's logical block while a
        huge page is a property of the memory mapping. Measured: a direct read into an address one
        block inside a huge page succeeds and is exactly as cheap as one at the huge page's start.
        """
        with patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "0"}):
            without = self._pool()._slot_size
        with patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "1"}):
            with_huge = self._pool()._slot_size

        self.assertEqual(without, with_huge, "the pad budget must not scale with the page size")

        # And it really is the block, not the page: the reserved room is exactly one block per pad.
        expected = 4 * 1024 * 1024 + DIRECT_IO_BLOCK * requests_iterator._max_pads_per_buffer()
        self.assertEqual(with_huge, expected)

    def test_mapping_memory_tolerates_a_bad_address(self):
        # It reads /proc/self/smaps, which may not exist. An address in no mapping must come back as
        # "no answer" rather than raising - the caller treats None as unknown.
        self.assertIsNone(_mapping_memory(0x1, 4096))

    def test_mapping_memory_reports_resident_bytes(self):
        # Rss is what says whether anything has been written. Without it a pool nobody has touched
        # reads as "no huge pages" and looks broken, which is how this check first cried wolf.
        #
        # What is asserted is our PARSE, checked against the kernel's own answer for the same memory.
        # How much stays resident is the host's decision, not ours.
        #
        # This used to require half the pool, and CI reported 128 KiB of a 16 MiB pool. Reproduced
        # against the real library: the pool covered TWO mappings, 128 KiB then 18 MiB, and only the
        # first was read. Our own madvise() on a sub-range is what splits them.
        #
        # mincore is the oracle because it answers the same question with no text to parse, so a
        # disagreement is our parse and not the machine reclaiming pages.
        pool = self._pool()
        pool.pool[:] = 1

        # Either side of the smaps read, so a host actively reclaiming these pages moves the band
        # rather than failing the test. A wrong mapping is out by orders of magnitude and still fails.
        before = _resident_bytes(pool.pool.ctypes.data, pool.pool.nbytes)

        measured = _mapping_memory(pool.pool.ctypes.data, pool.pool.nbytes)
        self.assertIsNotNone(measured)
        resident, _ = measured

        after = _resident_bytes(pool.pool.ctypes.data, pool.pool.nbytes)

        low = min(before, after) // 2
        high = max(before, after) * 2 + 64 * 1024

        self.assertTrue(
            low <= resident <= high,
            f"smaps reports {resident} bytes resident for this mapping, while mincore reports "
            f"{before}..{after} for the same range - the parse is reading a different mapping")

        # And something must be resident: distinguishing "written" from "untouched" is the only thing
        # the caller uses this number for.
        self.assertGreater(resident, 0)

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "1"})
    def test_an_untouched_pool_is_not_judged(self):
        # A request of only zero-sized ranges writes nothing, so its buffer comes back untouched and
        # there is nothing to conclude. The report must stay unanswered rather than warn - it cried
        # wolf exactly this way before Rss was consulted.
        #
        # _mapping_memory is faked rather than driven through a real pool, because what is under test
        # is the decision, not what this machine's allocator happens to do.
        pool = self._pool()
        with patch(
            "runai_model_streamer.file_streamer.requests_iterator._mapping_memory",
            return_value=(0, 0),
        ), patch.object(requests_iterator.logger, "warning") as warned:
            pool._report_huge_pages_once(0)
        warned.assert_not_called()
        self.assertFalse(pool._huge_pages_reported, "an untouched pool must be asked again later")

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "1"})
    def test_a_touched_pool_with_no_huge_pages_warns(self):
        # The case the check exists for: pages were written and none came back huge, so the O_DIRECT
        # pinning cost is back and nothing else would say so.
        pool = self._pool()
        with patch(
            "runai_model_streamer.file_streamer.requests_iterator._mapping_memory",
            return_value=(64 * 1024 * 1024, 0),
        ), patch.object(requests_iterator.logger, "warning") as warned:
            pool._report_huge_pages_once(0)
        warned.assert_called_once()
        self.assertIn("no huge pages", warned.call_args[0][0])
        self.assertTrue(pool._huge_pages_reported)

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "1"})
    def test_a_touched_pool_with_huge_pages_does_not_warn(self):
        pool = self._pool()
        with patch(
            "runai_model_streamer.file_streamer.requests_iterator._mapping_memory",
            return_value=(64 * 1024 * 1024, 64 * 1024 * 1024),
        ), patch.object(requests_iterator.logger, "warning") as warned:
            pool._report_huge_pages_once(0)
        warned.assert_not_called()
        self.assertTrue(pool._huge_pages_reported)

    @patch.dict(os.environ, {RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME: "1"})
    def test_the_report_runs_at_most_once(self):
        # It reads /proc/self/smaps, which is not free, and the answer cannot change. Once per pool.
        #
        # _mapping_memory is faked so the answer does not depend on how much of this small pool the
        # allocator happens to have made resident. Asserting on the real value made this test pass or
        # fail according to what ran before it - which it did, once the suite grew.
        pool = self._pool()
        with patch(
            "runai_model_streamer.file_streamer.requests_iterator._mapping_memory",
            return_value=(64 * 1024 * 1024, 64 * 1024 * 1024),
        ), patch.object(
            type(pool), "_report_huge_pages_once", wraps=pool._report_huge_pages_once
        ) as reported:
            request = pool.next_request()
            pool.release(request)
            self.assertEqual(reported.call_count, 1)
            self.assertTrue(pool._huge_pages_reported)

    def test_releasing_twice_does_not_report_twice(self):
        pool = self._pool()
        request = pool.next_request()
        pool.release(request)
        pool._huge_pages_reported = False   # pretend it never ran, to prove release drives it
        request = pool.next_request()
        if request is not None:
            pool.release(request)
            self.assertTrue(pool._huge_pages_reported)
