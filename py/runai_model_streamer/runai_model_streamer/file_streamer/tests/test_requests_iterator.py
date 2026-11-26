# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Roman Baron
# SPDX-FileCopyrightText: Copyright (c) 2026 Jan Bernlöhr
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
import unittest
import weakref
from unittest.mock import patch
from runai_model_streamer.file_streamer.requests_iterator import (
    advance,
    _alloc_slot,
    next_offset,
    SlotBounds,
    pack,
    rows_ended_a_slot,
    FileChunksIterator,
    FilesRequestsIterator,
    FilesRequestsIteratorWithBuffer,
    FileChunks,
    MemoryCapMode,
    RunaiStreamerMemoryLimitException,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
    RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME,
    DIRECT_IO_BLOCK,
    SlotMemory,
    _max_pads_per_buffer,
    _ring_sizing,
    device_ring_sizing,
    DEFAULT_DEVICE_RING_MIN_TOTAL_GIB,
    RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB_ENV_VAR_NAME,
    DEVICE_RING_PER_READER,
    DEFAULT_DEVICE_RING_SLOTS,
    MAX_DEVICE_RING_SLOTS,
    RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME,
    device_ring_target,
    device_slot_size,
    RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME,
    HOST_SLOTS,
    host_ring_sizing,
)
from runai_model_streamer.file_streamer.yield_order import YieldOrder
from runai_model_streamer.file_streamer.file_streamer import reads_object_storage


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
        base = requests_iterator._pool._slots[0].address

        files_requests = requests_iterator.next_request()
        self.assertEqual([f.id for f in files_requests.files], [17])
        # a.txt starts at file offset 10, so the first range skips 10 bytes to match it. The two after
        # it follow back to back in the file, so the cursor stays in step and costs nothing.
        self.assertEqual(files_requests.range_dsts, [base + 10, base + 11, base + 13])
        self.assert_congruent(files_requests)

        # largest_chunk mode is a ring of one, so the single slot has to come back before the next
        # request can be built. Owning it means the memory that comes back is NOT the memory that went
        # away, so the base is read again - the same index, a different address.
        requests_iterator.release(files_requests)

        files_requests = requests_iterator.next_request()
        base = requests_iterator._pool._slots[0].address
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
        base = requests_iterator._pool._slots[0].address

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
        base = requests_iterator._pool._slots[0].address
        requests_iterator._pool._slots[0].memory[request.range_dsts[request.flat_index(0, 0)] - base] = 9
        requests_iterator._pool._slots[0].memory[request.range_dsts[request.flat_index(1, 0)] - base] = 8

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
        self.assertEqual(third.range_dsts, [requests_iterator._pool._slots[first_index].address + 18])

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
    def test_buffers_are_distinct_and_do_not_overlap(self):
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.limited, [FileChunks.contiguous(17, "a.txt", 10, [4, 4, 4])], 12
        )
        self.assertEqual(len(requests_iterator._pool._slots), 3)

        # A slot is the caller's bytes plus room for the pads that make a direct read possible, so it
        # is bigger than the memory cap asked for. That overshoot is fixed per slot, not proportional,
        # so it disappears next to a real ring.
        slot_size = requests_iterator.buffer_size + DIRECT_IO_BLOCK * _max_pads_per_buffer()
        for slot in requests_iterator._pool._slots:
            self.assertEqual(len(slot.memory), slot_size)

        self.assertEqual(
            [slot.address for slot in requests_iterator._pool._slots],
            [slot.memory.ctypes.data for slot in requests_iterator._pool._slots],
        )

        # Writing through one slot must not disturb another. Separate allocations make this nearly
        # tautological, but it is the property the ring depends on and it held under the old shared
        # pool too - so it is asserted rather than assumed from the layout.
        for index, slot in enumerate(requests_iterator._pool._slots):
            slot.memory[:] = index
        for index, slot in enumerate(requests_iterator._pool._slots):
            # first and last byte of each slot, rather than the whole slot - they are megabytes
            self.assertEqual(slot.memory[0], index)
            self.assertEqual(slot.memory[-1], index)

    def test_every_slot_base_is_block_aligned(self):
        """Unconditional, because _alloc_slot() aligns each slot itself rather than carving them out
        of one region. The buffer size no longer reaches this: it decides how long a slot is, not
        where it starts.

        A convenience, not a requirement - _place() reaches a congruent address from any base.
        """
        requests_iterator = FilesRequestsIteratorWithBuffer(
            buffer_size=4 * 1024 * 1024,
            num_buffers=4,
            files_chunks=[FileChunks.contiguous(0, "a.bin", 0, [1024, 1024])],
        )
        for slot in requests_iterator._pool._slots:
            self.assertEqual(slot.address % DIRECT_IO_BLOCK, 0)

    def test_place_reaches_congruence_from_an_unaligned_base(self):
        """Why the slot base may use the page size while the pads use the probed block.

        _place() pads absolute addresses, not offsets within the slot, so it arrives at a congruent
        address wherever the slot happens to start. The sibling test above records that bases ARE
        aligned; this one records that they do not have to be."""
        offsets = [3, 1003, 2003]
        iterator = FilesRequestsIteratorWithBuffer(
            buffer_size=4 * 1024 * 1024,
            num_buffers=1,
            files_chunks=[FileChunks(0, "a.bin", offsets, [1000, 1000, 1000])],
        )
        request = iterator.next_request()   # only to get a request; its own layout is not used here

        # _place computes addresses without touching memory, so any integer serves as a base. Both of
        # these are taken against the block _place actually pads to, so neither can come out aligned.
        block = iterator.direct_block
        for base in (block + 1, 3 * block + 517):
            dsts = iterator._place(request, base, aligned=True)
            self.assertIsNotNone(dsts, f"no layout found from base {base}")
            for dst, offset in zip(dsts, offsets):
                self.assertEqual(dst % block, offset % block,
                                 f"address {dst} is not congruent to offset {offset}")

    def test_every_slot_is_a_whole_slot(self):
        # The over-allocation has to cover the base shift AND the slot itself. A short slot would be
        # found only by a read that happened to land in it.
        requests_iterator = FilesRequestsIteratorWithBuffer(
            buffer_size=4 * 1024 * 1024,
            num_buffers=4,
            files_chunks=[FileChunks.contiguous(0, "a.bin", 0, [1024, 1024])],
        )
        for slot in requests_iterator._pool._slots:
            self.assertEqual(len(slot.memory), requests_iterator._slot_size)

    def test_a_dropped_ring_releases_its_buffers_without_the_collector(self):
        """No reference cycle, so refcounting alone frees the ring.

        The pool keeps the allocator it is given. A bound method would point back at the iterator, and
        a cycle is invisible to refcounting - a replaced ring would hold its buffers until a gen2
        collection, which for a long-lived iterator is a long way off. owned=False is the case that
        matters: an owned pool has given its slots away by the end of a stream, an unowned one is
        still holding all of them."""
        requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_cap(
            MemoryCapMode.limited, [FileChunks.contiguous(17, "a.txt", 10, [4, 4])], 8, owned=False
        )
        ref = weakref.ref(requests_iterator)

        requests_iterator = None    # the only strong reference

        self.assertIsNone(ref(), "the ring outlived its last reference, so something holds a cycle")

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


class TestSubmissionsAreIssuedInOrdinalOrder(unittest.TestCase):
    """YieldOrder parks early responses and hands them over in ordinal order. That cannot stall only
    because a submission is never issued before an earlier ordinal, so whatever a parked view waits
    for is already in flight. These tests record that property on the iterators themselves.

    The ordinal is computed by YieldOrder, so the test follows the mapping the parking uses."""

    # Ids disagree with the list order, an empty file sits in the middle, and a limit of 8 splits
    # files across several requests while the largest range (7) still fits.
    RANGE_COUNTS = [[3, 3, 3, 3, 3], [4], [], [2, 2, 2], [7, 1, 1, 1, 1, 1, 1]]
    LIMIT = 8

    def files(self):
        return [FileChunks.contiguous(100 - position, f"{position}.bin", 0, list(sizes))
                for position, sizes in enumerate(self.RANGE_COUNTS)]

    def total_ranges(self):
        return sum(len(sizes) for sizes in self.RANGE_COUNTS)

    def ordinals_of(self, order, request):
        return [order.ordinal(file_chunks.id, request.global_range_index(file_index, range_index))
                for file_index, file_chunks in enumerate(request.files)
                for range_index in range(len(file_chunks.sizes))]

    def test_the_plain_iterator_issues_every_ordinal_once_in_order(self):
        files = self.files()
        order = YieldOrder(files)
        iterator = FilesRequestsIterator(self.LIMIT, files)

        issued = []
        requests = 0
        while True:
            request = iterator.next_request()
            if request is None:
                break
            issued += self.ordinals_of(order, request)
            requests += 1

        self.assertGreater(requests, 1, "one request would not test the order across submissions")
        self.assertEqual(issued, list(range(self.total_ranges())))

    def test_the_ring_issues_every_ordinal_once_in_order_while_slots_are_recycled(self):
        files = self.files()
        order = YieldOrder(files)
        ring = FilesRequestsIteratorWithBuffer(self.LIMIT, 2, files)

        issued = []
        live = []
        requests = 0
        while True:
            if not ring.has_free_buffer():
                ring.release(live.pop(0))      # oldest first, as a stream that consumes in order does
            request = ring.next_request()
            if request is None:
                break
            issued += self.ordinals_of(order, request)
            live.append(request)
            requests += 1

        self.assertGreater(requests, 2, "the ring has two slots, so recycling needs more requests")
        self.assertEqual(issued, list(range(self.total_ranges())))


GiB = 1 << 30


class TestPacking(unittest.TestCase):
    """What a slot size really achieves. A range cannot span two slots, so what does not fit is
    waste - and a slot can also end on its range cap, with its bytes still free."""

    def files(self, *sizes):
        return [FileChunks.contiguous(0, "f.bin", 0, list(sizes))]

    def test_ranges_that_tile_the_slot_exactly_waste_nothing(self):
        self.assertEqual(pack(self.files(4, 4, 4, 4), 8, SlotBounds()).fill, 1.0)

    def test_a_range_that_does_not_fit_leaves_the_rest_of_the_slot_unused(self):
        # 5 then 5: the second cannot join the first, so each slot of 8 holds one. 10 of 16 used.
        self.assertEqual(pack(self.files(5, 5), 8, SlotBounds()).fill, 10 / 16)

    def test_a_slot_the_size_of_one_range_is_still_full(self):
        # The floor is not automatically bad packing - this is what made the first hypothesis wrong.
        self.assertEqual(pack(self.files(8, 8, 8), 8, SlotBounds()).fill, 1.0)


class TestPackingHonoursEveryBound(unittest.TestCase):
    """THE DEFECT THIS REPLACED. The sizing used to count bytes only, while the caller passed a
    range cap - so it scored slots the ring could not produce and picked a size 40% larger than the
    cap could fill."""

    def files(self, *sizes):
        return [FileChunks.contiguous(0, "f.bin", 0, list(sizes))]

    def test_a_range_cap_ends_a_slot_before_its_bytes_run_out(self):
        packing = pack(self.files(*([1] * 8)), 100, SlotBounds(max_ranges=2))
        self.assertEqual(packing.slots, 4)
        self.assertEqual(packing.fill, 8 / 400)

    def test_the_same_ranges_uncapped_need_one_slot(self):
        """Pins that the test above measures the cap: 8 bytes fit a 100 byte slot either way."""
        packing = pack(self.files(*([1] * 8)), 100, SlotBounds())
        self.assertEqual(packing.slots, 1)
        self.assertEqual(packing.fill, 8 / 100)

    def test_it_reports_when_rows_ended_a_slot_rather_than_bytes(self):
        # 8 ranges of 1 fit a slot of 100 by bytes; a cap of 2 splits them into four slots.
        self.assertTrue(rows_ended_a_slot(self.files(*([1] * 8)), 100, SlotBounds(max_ranges=2)))

    def test_a_cap_that_never_binds_is_not_reported(self):
        self.assertFalse(rows_ended_a_slot(self.files(*([1] * 8)), 100, SlotBounds(max_ranges=99)))

    def test_a_cap_the_bytes_reach_first_is_not_reported(self):
        """THE FALSE POSITIVE a row count alone would give. Four ranges of 25 fill a slot of 100
        exactly, so a slot holds four and stops - on its bytes. A cap of 4 changes nothing, and
        saying the cap ended the slot would send someone to look at the wrong number."""
        self.assertEqual(pack(self.files(*([25] * 8)), 100, SlotBounds(max_ranges=4)).widest, 4)
        self.assertFalse(rows_ended_a_slot(self.files(*([25] * 8)), 100, SlotBounds(max_ranges=4)))

    def test_no_cap_is_never_reported(self):
        self.assertFalse(rows_ended_a_slot(self.files(*([1] * 8)), 100, SlotBounds()))

    def test_alignment_is_honoured_too(self):
        """The other bound the byte-only count ignored. Three ranges of 3 packed one after another
        fit a slot of 9; aligned to 4 they start at 0, 4 and 8, so the third needs its own slot."""
        self.assertEqual(pack(self.files(3, 3, 3), 9, SlotBounds()).slots, 1)
        self.assertEqual(pack(self.files(3, 3, 3), 9, SlotBounds(range_alignment=4)).slots, 2)


class TestDeviceRingShape(unittest.TestCase):
    """Which buffer size the device ring picks, given a budget it must recycle within."""

    def files(self, *sizes_per_file):
        return [FileChunks.contiguous(i, f"{i}.bin", 0, list(sizes))
                for i, sizes in enumerate(sizes_per_file)]

    @patch.dict(os.environ, {}, clear=True)
    def test_the_slot_is_the_largest_range_when_that_is_the_bigger_floor(self):
        """The data decides. A range carries one destination address, so it cannot span two slots -
        the largest is a hard floor, and nothing above it is asked for."""
        size = device_slot_size(self.files([3 * GiB, GiB, GiB]))
        self.assertEqual(size, 3 * GiB)

    @patch.dict(os.environ, {}, clear=True)
    def test_small_ranges_get_a_share_of_the_minimum_total_rather_than_a_tiny_slot(self):
        """Without this floor a model of 1 MiB tensors would get a slot of 1 MiB. The floor is a
        TOTAL, so what a slot gets is that total divided by the depth."""
        # 97.6 GiB of 1 MiB ranges: far above the floor, so the stream cap cannot bind instead.
        size = device_slot_size(self.files([1 << 20] * 100000))
        self.assertEqual(size, 2 * GiB)
        self.assertEqual(size * DEFAULT_DEVICE_RING_SLOTS, DEFAULT_DEVICE_RING_MIN_TOTAL_GIB * GiB)

    @patch.dict(os.environ, {}, clear=True)
    def test_the_total_grows_with_the_readers_that_fill_it(self):
        """More readers reach a higher rate, and a higher rate empties the ring sooner."""
        self.assertEqual(device_ring_target(1), 8 * GiB)
        # the minimum outranks the per-reader term until there are more readers than gibibytes in it
        self.assertEqual(device_ring_target(8), 8 * GiB)
        self.assertEqual(device_ring_target(32), 32 * GiB)

        small = self.files([1 << 20] * 100000)
        self.assertEqual(device_slot_size(small, 1), 2 * GiB)
        self.assertEqual(device_slot_size(small, 32), 8 * GiB)

    @patch.dict(os.environ, {}, clear=True)
    def test_a_reader_count_below_one_is_read_as_one(self):
        """Zero readers is not a ring of nothing: the library answers at least one, and this is the
        same answer for a caller that passed nothing."""
        self.assertEqual(device_ring_target(0), DEFAULT_DEVICE_RING_MIN_TOTAL_GIB * GiB)
        self.assertEqual(device_ring_target(-3), DEFAULT_DEVICE_RING_MIN_TOTAL_GIB * GiB)

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB_ENV_VAR_NAME: "32"}, clear=True)
    def test_the_minimum_total_can_be_raised_from_the_environment(self):
        """The limit and the fraction beside it only ever LOWER the ring. How much a ring must hold
        depends on the storage behind it, which this layer cannot see, so this one raises it."""
        self.assertEqual(device_ring_target(1), 32 * GiB)
        # ...and it is a MINIMUM, so the per-reader term still wins where it is larger
        self.assertEqual(device_ring_target(64), 64 * GiB)
        self.assertEqual(device_slot_size(self.files([1 << 20] * 100000), 1), 8 * GiB)

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB_ENV_VAR_NAME: "0"}, clear=True)
    def test_a_minimum_total_of_zero_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as raised:
            device_ring_target(1)
        self.assertIn(RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB_ENV_VAR_NAME: "sixteen"}, clear=True)
    def test_a_minimum_total_that_is_not_a_number_names_the_variable(self):
        with self.assertRaises(ValueError) as raised:
            device_ring_target(1)
        self.assertIn(RUNAI_STREAMER_DEVICE_RING_MIN_TOTAL_GIB_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME: "8"}, clear=True)
    def test_the_slot_count_can_be_raised_from_the_environment(self):
        """The total is split between the slots, so more slots means smaller ones."""
        shape = device_ring_sizing(1 << 40, self.files([1 << 20] * 100000))
        self.assertEqual(shape, (DEFAULT_DEVICE_RING_MIN_TOTAL_GIB * GiB // 8, 8))
        self.assertEqual(device_slot_size(self.files([1 << 20] * 100000)), GiB)

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME: "8"}, clear=True)
    def test_the_largest_range_still_outranks_a_raised_slot_count(self):
        """A range carries one destination address and cannot span two slots, however many there are."""
        self.assertEqual(device_slot_size(self.files([3 * GiB] + [1 << 20] * 100000)), 3 * GiB)

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME: "10000000"}, clear=True)
    def test_an_absurd_slot_count_is_capped_rather_than_refused(self):
        """A number this far out is a typo, not a request worth failing a load over. Uncapped, the
        depth walk and the allocation that follows it both run once per slot."""
        with self.assertLogs(level="WARNING") as logs:
            size, slots = device_ring_sizing(1 << 40, self.files([1 << 20] * 100000))
        self.assertEqual(slots, MAX_DEVICE_RING_SLOTS)
        self.assertIn(RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME, "".join(logs.output))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME: "0"}, clear=True)
    def test_a_slot_count_below_one_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as raised:
            device_slot_size(self.files([GiB]))
        self.assertIn(RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME: "two"}, clear=True)
    def test_a_slot_count_that_is_not_a_number_names_the_variable(self):
        with self.assertRaises(ValueError) as raised:
            device_slot_size(self.files([GiB]))
        self.assertIn(RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_the_readers_never_lower_the_largest_range(self):
        """The hard floor outranks the share: one reader asks for 2 GiB a slot, and a 3 GiB range
        still cannot span two slots."""
        self.assertEqual(device_slot_size(self.files([3 * GiB, GiB]), 1), 3 * GiB)

    @patch.dict(os.environ, {}, clear=True)
    def test_the_readers_do_not_lift_a_slot_past_the_whole_stream(self):
        """A 64 MiB model gets its share of the depth, not what the readers asked for - slots that
        can overlap, rather than one slot holding everything."""
        self.assertEqual(device_slot_size(self.files([1 << 20] * 64), 8), 16 << 20)

    @patch.dict(os.environ, {}, clear=True)
    def test_a_slot_is_never_bigger_than_the_whole_stream(self):
        """A ring cannot read ahead past the end, so reserving for bytes that do not exist is the
        waste this rule exists to stop."""
        self.assertEqual(device_slot_size(self.files([4096])), 4096)

    @patch.dict(os.environ, {}, clear=True)
    def test_a_target_that_does_not_fit_shrinks_rather_than_refusing(self):
        """The readers ask for more than the device has, and the answer is a SMALLER device ring -
        not the host one. The refusal gate is the overlap floor, which this does not reach."""
        small = self.files([1 << 20] * 100000)

        # The ring asks for its minimum; half of 6 GiB free is 3 GiB, so the slot gives way first.
        shape = device_ring_sizing(6 * GiB, small, readers=8)
        self.assertIsNotNone(shape)
        buffer_size, num_buffers = shape
        self.assertEqual(num_buffers, DEFAULT_DEVICE_RING_SLOTS)
        self.assertEqual(buffer_size, 768 * (1 << 20))
        self.assertEqual(buffer_size * num_buffers, 3 * GiB)

    @patch.dict(os.environ, {}, clear=True)
    def test_the_readers_cannot_push_a_device_off_the_device_ring(self):
        """A bigger ASK must not make the fallback more likely: the gate is two of the largest
        range, which the reader count does not touch."""
        small = self.files([1 << 20] * 100000)

        # Half of 8 MiB is 4 MiB - four of the largest range, so a ring still fits.
        for readers in (1, 8):
            shape = device_ring_sizing(8 << 20, small, readers=readers)
            self.assertIsNotNone(shape, f"{readers} reader(s) refused a ring that fits")
            self.assertEqual(shape, (1 << 20, DEFAULT_DEVICE_RING_SLOTS))

    @patch.dict(os.environ, {}, clear=True)
    def test_nothing_to_read_asks_for_nothing(self):
        self.assertEqual(device_slot_size([]), 0)
        self.assertEqual(device_slot_size(self.files([])), 0)

    @patch.dict(os.environ, {}, clear=True)
    def test_the_depth_is_the_overlap_floor_and_the_memory_follows(self):
        """The depth the ring asks for, and no more - not whatever a configured budget would have
        paid for. One slot never overlaps anything, so the depth is where the overlap lives."""
        size, slots = device_ring_sizing(100 * GiB, self.files([3 * GiB] * 8))
        self.assertEqual((size, slots), (3 * GiB, DEFAULT_DEVICE_RING_SLOTS))

    @patch.dict(os.environ, {}, clear=True)
    def test_what_the_caller_will_allocate_beside_the_ring_is_counted_here(self):
        """One solver. A reservation taken afterwards would shrink a shape this already chose."""
        size, slots = device_ring_sizing(30 * GiB, self.files([3 * GiB] * 8), extra_slots=2)
        self.assertLessEqual((slots + 2) * size, int(30 * GiB * 0.5))
        self.assertEqual(slots, 3, "the depth gave one slot to the reservation and kept the overlap")

    @patch.dict(os.environ, {}, clear=True)
    def test_the_slot_gives_way_before_the_depth_does(self):
        """Dropping to one slot is the one cliff both models measure. A slot somewhat below what
        the data asked for is flat over a wide range, so that is what gives."""
        # Half of 5 GiB cannot pay for the depth at any slot size, so both have to give - and the
        # order is what this pins: the slot reaches the largest range FIRST, and only then does the
        # depth come down to what that size fits.
        size, slots = device_ring_sizing(5 * GiB, self.files([GiB] * 8))
        self.assertEqual(size, GiB, "the slot gave way to the largest range before the depth moved")
        self.assertEqual(slots, 2, "and the depth stopped at the overlap floor, not below it")

    @patch.dict(os.environ, {}, clear=True)
    def test_two_slots_beat_one_that_cannot_span_the_stream(self):
        """THE REGRESSION. The one-slot shortcut once ran BEFORE the shrink, so a shape was pinned
        at one slot using a size the allowance could not pay for - the slot then shrank below the
        stream and the ring could neither span it nor overlap, where two smaller slots fitted the
        same allowance."""
        MiB = 1 << 20
        stream = self.files([10 * MiB] * 20)                 # 200 MiB in 10 MiB ranges
        size, slots = device_ring_sizing(600 * MiB, stream, extra_slots=2)

        self.assertEqual(slots, DEFAULT_DEVICE_RING_SLOTS,
                         "one slot here cannot span the stream, so it must not be chosen")
        self.assertLess(size, 200 * MiB, "and the fixture really does not fit in one slot")
        self.assertLessEqual((slots + 2) * size, 300 * MiB, "within the allowance")

    @patch.dict(os.environ, {}, clear=True)
    def test_a_range_cap_does_not_change_the_shape(self):
        """A slot is capped at the stream's share of the depth, so a stream of SEVERAL ranges keeps
        its slots whether or not a cap also ends them early. The cap is the caller's business; it
        must not reach the sizing."""
        MiB = 1 << 20
        stream = self.files([MiB] * 4)                       # 4 MiB over four ranges
        bounds = SlotBounds(max_ranges=2)                    # only two ranges at a time

        size, slots = device_ring_sizing(64 * MiB, stream, bounds)

        self.assertEqual(size, MiB, "the stream's share of the depth, so the rest has slots to go in")
        self.assertEqual(slots, DEFAULT_DEVICE_RING_SLOTS, "every slot, so reading and handing over overlap")
        self.assertEqual(device_ring_sizing(64 * MiB, stream), (size, slots),
                         "the cap changed nothing: the slot size alone decided the shape")


    @patch.dict(os.environ, {}, clear=True)
    def test_one_range_that_is_the_whole_stream_takes_one_slot(self):
        """The one case a single slot is right: there is no second range to overlap with, and a
        second slot could never be filled."""
        MiB = 1 << 20
        self.assertEqual(device_ring_sizing(64 * MiB, self.files([4 * MiB])), (4 * MiB, 1))

    @patch.dict(os.environ, {}, clear=True)
    def test_alignment_that_pads_the_stream_keeps_every_slot(self):
        """The same for padding: each range is rounded up, so the packed stream outgrows a slot
        that its raw bytes fit inside."""
        stream = self.files([8] * 4)                         # 32 bytes raw
        bounds = SlotBounds(range_alignment=1 << 20)         # a mebibyte each once padded

        size, slots = device_ring_sizing(64 << 20, stream, bounds)

        self.assertEqual(slots, DEFAULT_DEVICE_RING_SLOTS, "padded, this stream still fills every slot")

    @patch.dict(os.environ, {}, clear=True)
    def test_one_slot_is_still_right_when_it_does_span(self):
        """Pins that the test above is about SPANNING and not about always answering two: a stream
        a slot holds whole recycles nothing, so a second slot could never be filled."""
        self.assertEqual(device_ring_sizing(100 * GiB, self.files([GiB]))[1], 1)

    @patch.dict(os.environ, {}, clear=True)
    def test_too_little_for_one_slot_is_the_host_path(self):
        self.assertIsNone(device_ring_sizing(GiB, self.files([4 * GiB])))

class TestDeviceRingSizing(unittest.TestCase):
    """What the ring asks for, and the two backstops that can lower it.

    free_bytes is a parameter, so none of this needs a GPU."""

    def files(self, *sizes_per_file):
        return [FileChunks.contiguous(i, f"{i}.bin", 0, list(sizes))
                for i, sizes in enumerate(sizes_per_file)]

    @patch.dict(os.environ, {}, clear=True)
    def test_an_empty_device_does_not_make_the_ring_bigger(self):
        """THE POINT OF THE RULE. 100 GiB free would allow 50, and the ring still takes two slots
        of what the target asks for. The old rule turned free memory into a budget and the budget
        into a slot size, so a model got a bigger slot for being loaded on an emptier device."""
        # The SAME answer on a nearly empty device and on a far fuller one, which is the rule.
        self.assertEqual(device_ring_sizing(100 * GiB, self.files([GiB] * 8)),
                         (2 * GiB, DEFAULT_DEVICE_RING_SLOTS))
        self.assertEqual(device_ring_sizing(20 * GiB, self.files([GiB] * 8)),
                         (2 * GiB, DEFAULT_DEVICE_RING_SLOTS))

    @patch.dict(os.environ, {"RUNAI_STREAMER_DEVICE_MEMORY_LIMIT": str(3 * GiB)}, clear=True)
    def test_a_limit_that_is_set_is_a_hard_cap(self):
        buffer_size, num_buffers = device_ring_sizing(100 * GiB, self.files([GiB] * 8))
        self.assertLessEqual(buffer_size * num_buffers, 3 * GiB)
        self.assertGreaterEqual(buffer_size, GiB)      # never below the largest range

    @patch.dict(os.environ, {}, clear=True)
    def test_an_unset_limit_caps_nothing(self):
        """Pins that the test above measures the setting. The same call without it is not capped."""
        self.assertEqual(device_ring_sizing(100 * GiB, self.files([GiB] * 8)),
                         (2 * GiB, DEFAULT_DEVICE_RING_SLOTS))

    @patch.dict(os.environ, {}, clear=True)
    def test_the_fraction_binds_when_little_is_free(self):
        # 4 GiB free -> 2 GiB allowed, and two slots of the largest range is exactly that.
        buffer_size, num_buffers = device_ring_sizing(4 * GiB, self.files([GiB] * 8))
        self.assertLessEqual(buffer_size * num_buffers, 2 * GiB)

    def test_nothing_outside_can_name_the_device_depth(self):
        """The host ring takes a caller's depth; the device ring does not. A depth from outside
        could only ask for memory the data does not need - naming 100 once claimed 50 GiB - and on
        the distributed path a rank choosing its own would agree a shape its peers did not."""
        plain = device_ring_sizing(100 * GiB, self.files([GiB] * 8))
        for value in ("1", "4", "18", "100"):
            with self.subTest(setting=value):
                with patch.dict(os.environ,
                                {RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: value}, clear=True):
                    self.assertEqual(device_ring_sizing(100 * GiB, self.files([GiB] * 8)), plain)
        self.assertEqual(plain[1], DEFAULT_DEVICE_RING_SLOTS)

    @patch.dict(os.environ, {}, clear=True)
    def test_below_two_largest_ranges_it_falls_back(self):
        # 3 GiB free -> 1.5 GiB budget, under the 2 GiB floor for a 1 GiB range.
        self.assertIsNone(device_ring_sizing(3 * GiB, self.files([GiB] * 8)))

    @patch.dict(os.environ, {}, clear=True)
    def test_exactly_two_largest_ranges_is_enough(self):
        sizing = device_ring_sizing(4 * GiB, self.files([GiB] * 8))
        self.assertIsNotNone(sizing)
        self.assertEqual(sizing[1], 2, "the floor is two buffers, so nothing deeper fits here")

    @patch.dict(os.environ, {}, clear=True)
    def test_a_device_with_nothing_free_falls_back(self):
        self.assertIsNone(device_ring_sizing(0, self.files([GiB] * 8)))

    @patch.dict(os.environ, {}, clear=True)
    def test_a_device_torch_cannot_measure_falls_back(self):
        """None is what device_free_memory() reports for a device it cannot measure, and it answers
        here rather than at each call site, so no caller invents its own meaning for unknown."""
        self.assertIsNone(device_ring_sizing(None, self.files([GiB] * 8)))

    @patch.dict(os.environ, {}, clear=True)
    def test_the_limit_does_not_veto_a_ring_the_device_can_afford(self):
        """A 3 GiB range needs a 6 GiB floor, above the 4 GiB limit - but the device has 100 GiB.
        The limit is a target, so the data raises it rather than losing the device ring.

        Eight ranges, not three: the ring never exceeds the stream, so a total below the floor would
        give one buffer however large the budget is, and the depth below would say nothing."""
        sizing = device_ring_sizing(100 * GiB, self.files([3 * GiB] * 8))
        self.assertIsNotNone(sizing, "a setting refused a ring the device had room for")
        buffer_size, num_buffers = sizing
        self.assertGreaterEqual(buffer_size, 3 * GiB)      # a range cannot span two buffers
        self.assertGreaterEqual(num_buffers, 2)            # and two is what lets them overlap

    @patch.dict(os.environ, {}, clear=True)
    def test_the_fraction_still_refuses_what_the_device_cannot_hold(self):
        """The one thing that can still say no. 20 GiB free allows 10, and two 50 GiB ranges need
        100 - raising the target past that would hand out memory the device does not have."""
        self.assertIsNone(device_ring_sizing(20 * GiB, self.files([50 * GiB, GiB])))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME: str(2 * GiB)}, clear=True)
    def test_the_limit_is_configurable(self):
        buffer_size, num_buffers = device_ring_sizing(100 * GiB, self.files([GiB] * 8))
        self.assertLessEqual(buffer_size * num_buffers, 2 * GiB)

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME: "1.0"}, clear=True)
    def test_the_fraction_is_configurable(self):
        # The whole 3 GiB is now usable, so the 2 GiB floor is met where 0.5 would have refused.
        self.assertIsNotNone(device_ring_sizing(3 * GiB, self.files([GiB] * 8)))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME: "1.5"}, clear=True)
    def test_a_fraction_above_one_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            device_ring_sizing(100 * GiB, self.files([GiB]))
        self.assertIn(RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME: "-1"}, clear=True)
    def test_the_hosts_unlimited_mode_is_rejected_rather_than_guessed(self):
        with self.assertRaises(ValueError) as raised:
            device_ring_sizing(100 * GiB, self.files([GiB]))
        self.assertIn(RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME: "4.5"}, clear=True)
    def test_a_limit_that_is_not_a_number_names_the_variable(self):
        with self.assertRaises(ValueError) as raised:
            device_ring_sizing(100 * GiB, self.files([GiB]))
        self.assertIn(RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME: "half"}, clear=True)
    def test_a_fraction_that_is_not_a_number_names_the_variable(self):
        with self.assertRaises(ValueError) as raised:
            device_ring_sizing(100 * GiB, self.files([GiB]))
        self.assertIn(RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME, str(raised.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_a_stream_smaller_than_the_budget_gets_fewer_buffers(self):
        """The floor is on the BUDGET, not the depth. One range has nothing to overlap with, so one
        buffer is the whole ring - and that still beats the host path, which would copy."""
        self.assertEqual(device_ring_sizing(100 * GiB, self.files([GiB]))[1], 1)
        self.assertEqual(device_ring_sizing(100 * GiB, self.files([GiB] * 2))[1], 2)

    @patch.dict(os.environ, {}, clear=True)
    def test_only_the_requested_ranges_decide_the_floor(self):
        """A caller that asks for small tensors is not charged for a huge one it did not request -
        SafetensorsStreamer filters before building FileChunks, so the giant range is simply absent."""
        self.assertIsNone(device_ring_sizing(8 * GiB, self.files([90 * GiB, GiB])))
        self.assertIsNotNone(device_ring_sizing(8 * GiB, self.files([GiB])))


class TestAnImpossibleBufferTruncatesSilently(unittest.TestCase):
    """WHY stream_files refuses a dictated shape that cannot hold the largest range.

    The packer carries no guard and never will - stream_files is what refuses an impossible shape
    before one is ever built. So this drives the packer directly and shows what the refusal
    prevents: the range is handed over as NOTHING, the stream simply ends, and nothing is raised."""

    def requests(self, sizes):
        return [FileChunks.contiguous(0, "/m/a.safetensors", 0, sizes)]

    def test_a_range_larger_than_the_buffer_ends_the_stream_with_bytes_unread(self):
        packer = FilesRequestsIterator(1000, self.requests([800, 5000, 800]))

        handed_over = []
        while True:
            request = packer.next_request()
            if request is None:
                break
            handed_over.extend(size for fc in request.files for size in fc.sizes)

        self.assertEqual(handed_over, [800],
                         "the 5000-byte range stops the stream, and the 800 after it is lost too")
        self.assertEqual(sum(handed_over), 800, "800 of 6600 bytes, silently")


class TestSlotMemoryOf(unittest.TestCase):
    """The used prefix of a slot: what a caller broadcasts out of it.

    The whole slot would be correct and simpler, but the last slot of a stream is usually part full,
    so its unused tail is most of it - and on a collective that tail costs the same as real data."""

    def two_files(self, sizes):
        """Two files, so an aligned layout has somewhere to insert a pad.

        Within one file read from offset 0 no pad is ever inserted: the slot base is page aligned,
        so address and file offset stay in step by themselves."""
        return [FileChunks.contiguous(0, "/m/a.safetensors", 0, sizes),
                FileChunks.contiguous(1, "/m/b.safetensors", 0, sizes)]

    def test_an_aligned_layout_measures_the_extent_not_the_sum(self):
        sizes = [1000, 2000]
        iterator = FilesRequestsIteratorWithBuffer(
            1 << 20, 2, self.two_files(sizes), 4096, owned=False)
        request = iterator.next_request()

        used = len(iterator.slot_memory_of(request))
        total = sum(sizes) * 2

        # The pad at the file boundary makes the extent LONGER than the data. A caller that sent
        # `total` bytes would cut the last range short.
        self.assertGreater(used, total, "an aligned layout leaves a gap, so the extent exceeds the sum")
        # max(), not [-1]: that is what makes this independent of the implementation, which takes
        # the LAST destination on the grounds that _place only advances. A placement that reordered
        # ranges would make the two disagree, and this is the assertion that would notice.
        base = iterator._pool.slot(request.buffer_index).address
        furthest = max(dst for dst in request.range_dsts) + sizes[-1] - base
        self.assertEqual(used, furthest, "the prefix ends where the FURTHEST range ends")

    def test_a_tight_layout_measures_exactly_the_data(self):
        sizes = [1000, 2000]
        iterator = FilesRequestsIteratorWithBuffer(
            1 << 20, 2, self.two_files(sizes), 4096, owned=False,
            slots=HOST_SLOTS._replace(direct_io=False))
        request = iterator.next_request()
        self.assertEqual(len(iterator.slot_memory_of(request)), sum(sizes) * 2)

    def test_a_part_full_slot_sends_less_than_the_whole_slot(self):
        # One small range in a slot sized for far more.
        iterator = FilesRequestsIteratorWithBuffer(
            1 << 20, 2, [FileChunks.contiguous(0, "/m/a.safetensors", 0, [4096])], 4096, owned=False)
        request = iterator.next_request()
        self.assertEqual(len(iterator.slot_memory_of(request)), 4096)
        self.assertLess(len(iterator.slot_memory_of(request)),
                        len(iterator._pool.slot(request.buffer_index).memory))


class TestHostRingSizing(unittest.TestCase):
    """The shape the host ring would choose, asked without building one.

    Peers agreeing a shape have to know it before any of them allocates. The property that matters
    is that the answer is the shape the ring ACTUALLY takes - so these compare against a real ring
    rather than against a second calculation."""

    def chunks(self, sizes):
        return [FileChunks.contiguous(0, "/m/a.safetensors", 0, sizes)]

    def test_it_matches_the_ring_that_would_be_built(self):
        for limit in (1 << 20, 1 << 24, 1 << 30):
            for sizes in ([4096] * 50, [1 << 16, 4096, 1 << 18]):
                with self.subTest(limit=limit, sizes=sizes):
                    predicted = host_ring_sizing(self.chunks(sizes), limit)
                    built = FilesRequestsIteratorWithBuffer.with_memory_mode(
                        self.chunks(sizes), limit, 4096, False)
                    self.assertEqual(predicted, (built.buffer_size, built.num_buffers))

    def test_the_depth_is_honoured(self):
        predicted = host_ring_sizing(self.chunks([4096] * 50), 1 << 24, ring_buffers=3)
        built = FilesRequestsIteratorWithBuffer.with_memory_mode(
            self.chunks([4096] * 50), 1 << 24, 4096, False, ring_buffers=3)
        self.assertEqual(predicted, (built.buffer_size, built.num_buffers))
        self.assertEqual(predicted[1], 3)


class TestSlotMemoryOfRefusesAReleasedSlot(unittest.TestCase):
    """A released request has no slot, and saying so beats indexing with None.

    Without this the pool is indexed with None and raises `TypeError: list indices must be
    integers`, which says nothing about what the caller did wrong."""

    def test_asking_after_release_says_what_happened(self):
        chunks = [FileChunks.contiguous(0, "/m/a.safetensors", 0, [1000])]
        iterator = FilesRequestsIteratorWithBuffer(1 << 20, 2, chunks, 4096, owned=False)
        request = iterator.next_request()
        iterator.release(request)

        with self.assertRaises(RuntimeError) as caught:
            iterator.slot_memory_of(request)
        self.assertIn("released", str(caught.exception))


class TestHostRingSizingAcrossModes(unittest.TestCase):
    """The modes and the branch the agreement actually uses.

    TestHostRingSizing compares against a built ring, but every one of its cases lands in `limited`
    with a limit that never binds - so it only ever exercises the whole-stream branch. The
    distributed case is the opposite: a limit that BINDS, giving a recycling ring."""

    def chunks(self, sizes):
        return [FileChunks.contiguous(0, "/m/a.safetensors", 0, sizes)]

    def matches_the_built_ring(self, sizes, limit, ring_buffers=None):
        predicted = host_ring_sizing(self.chunks(sizes), limit, ring_buffers)
        built = FilesRequestsIteratorWithBuffer.with_memory_mode(
            self.chunks(sizes), limit, 4096, False, ring_buffers)
        self.assertEqual(predicted, (built.buffer_size, built.num_buffers))
        return predicted

    def test_a_limit_that_binds(self):
        # total 331776, limit 300000: the ring must RECYCLE, which is the distributed case and the
        # branch TestHostRingSizing never reaches.
        sizes = [65536, 4096, 262144]
        self.assertGreater(sum(sizes), 300000, "the limit has to bind, or this repeats the other test")
        self.matches_the_built_ring(sizes, 300000)

    def test_largest_chunk_mode(self):
        # 0 is a MODE, not a quantity: one buffer, the size of the largest range.
        buffer_size, num_buffers = self.matches_the_built_ring([65536, 4096, 262144], 0)
        self.assertEqual((buffer_size, num_buffers), (262144, 1))

    def test_unlimited_mode(self):
        # -1 is the other mode: spend what the stream needs.
        self.matches_the_built_ring([65536, 4096, 262144], -1)

    def test_the_memory_limit_is_required(self):
        # Defaulting it would hand every rank the NODE total - the hazard the signature exists to
        # prevent, and one that no other test would notice.
        with self.assertRaises(TypeError):
            host_ring_sizing(self.chunks([4096]))


# A ring nothing reads into directly, which is the only kind that may be sent from.
SENT_SLOTS = SlotMemory(_alloc_slot, direct_io=False, label="TEST", new_event=None,
                        current_stream=None)

ALIGNMENT = 512


class TestRangeAlignment(unittest.TestCase):
    """A ring that is SENT FROM lays its ranges out by a rule, so a peer can compute where each one
    is instead of being told."""

    def ring(self, sizes, buffer_size, alignment=ALIGNMENT, slots=SENT_SLOTS):
        files = [FileChunks.contiguous(17, "a.txt", 0, sizes)]
        return FilesRequestsIteratorWithBuffer(
            buffer_size, 2, files, owned=True, slots=slots, range_alignment=alignment,
        )

    def offsets_of(self, iterator, request):
        """Each range's offset inside its slot, measured FROM THE SLOT.

        Not from range_dsts[0]: that would compare the layout with its own first element, and a
        layout shifted bodily down the slot - one that runs off the end in production - would read
        as correct."""
        base = iterator._pool.slot(request.buffer_index).address
        return [dst - base for dst in request.range_dsts]

    def test_the_offsets_follow_the_rule(self):
        iterator = self.ring([100, 200, 300], buffer_size=ALIGNMENT * 4)
        request = iterator.next_request()

        self.assertEqual(self.offsets_of(iterator, request), [0, 512, 1024])

    def test_a_size_already_on_the_rule_is_not_pushed_on(self):
        iterator = self.ring([ALIGNMENT, 100], buffer_size=ALIGNMENT * 4)
        request = iterator.next_request()

        self.assertEqual(self.offsets_of(iterator, request), [0, 512])

    def test_the_packer_counts_the_pads(self):
        """Three 200-byte ranges fit a 600-byte slot packed tight, and one fits once they are spaced
        512 apart. A packer that still counted the sizes would fill a slot the placer cannot lay
        out."""
        tight = self.ring([200, 200, 200], buffer_size=600, alignment=1)
        self.assertEqual(len(tight.next_request().range_dsts), 3)

        spaced = self.ring([200, 200, 200], buffer_size=600)
        self.assertEqual(len(spaced.next_request().range_dsts), 1)

    def test_a_range_ending_the_slot_exactly_still_fits(self):
        """The last range needs room for its BYTES, not for a pad after it."""
        iterator = self.ring([ALIGNMENT, 100], buffer_size=ALIGNMENT + 100)
        request = iterator.next_request()

        self.assertEqual(self.offsets_of(iterator, request), [0, 512])

    def test_the_whole_stream_fits_slot_after_slot(self):
        """The property the two have to share: whatever the packer puts in a request, the placer can
        lay out. A disagreement raises, so reaching the end of the stream is the assertion."""
        sizes = [1, 511, 512, 513, 1023, 7, 4096, 3, 2048, 9]
        iterator = self.ring(sizes, buffer_size=ALIGNMENT * 12)

        placed = 0
        while True:
            request = iterator.next_request()
            if request is None:
                break
            offsets = self.offsets_of(iterator, request)
            self.assertEqual(offsets[0], 0, "every request starts at its slot's base")
            for offset, size in zip(offsets, [s for f in request.files for s in f.sizes]):
                if size:
                    self.assertEqual(offset % ALIGNMENT, 0)
            extent = len(iterator.slot_memory_of(request))
            self.assertLessEqual(extent, ALIGNMENT * 12, "the placer overran the packer's slot")
            placed += len(offsets)
            iterator.release(request)

        self.assertEqual(placed, len(sizes), "a range never reached a slot")

    def test_the_extent_sent_to_a_peer_includes_the_pads(self):
        """slot_memory_of is what a caller broadcasts. Under the rule the ranges are spaced, so the
        sum of the sizes is SHORTER than the extent, and sending that many bytes would truncate the
        last range."""
        iterator = self.ring([100, 200, 300], buffer_size=ALIGNMENT * 4)
        request = iterator.next_request()

        self.assertEqual(len(iterator.slot_memory_of(request)), 1024 + 300)
        self.assertGreater(len(iterator.slot_memory_of(request)), 100 + 200 + 300)

    def test_the_views_land_where_the_layout_says(self):
        """Written through the destinations and read back through the views the caller is given.

        Distinct bytes per range, so an overlap or an off-by-one pad reads as another range's data
        rather than as the right answer."""
        sizes = [100, 200, 300]
        iterator = self.ring(sizes, buffer_size=ALIGNMENT * 4)
        request = iterator.next_request()

        slot = iterator._pool.slot(request.buffer_index)
        for index, (dst, size) in enumerate(zip(request.range_dsts, sizes)):
            at = dst - slot.address
            slot.memory[at:at + size] = index + 1

        for index, size in enumerate(sizes):
            _, _, view = iterator.get_global_file_and_range(request, 0, index)
            self.assertEqual(bytes(view), bytes([index + 1]) * size,
                             f"range {index} did not read back as itself")

    def test_a_trailing_pad_would_cost_a_range(self):
        """The walk ends at the last range's DATA. A cursor left after a pad would make the packer
        think less room is left than there is, which this slot is sized to expose: 100 then 412 fit
        exactly, and only if the cursor after the first range is 100 rather than 512."""
        iterator = self.ring([100, 412], buffer_size=ALIGNMENT, alignment=1)
        self.assertEqual(len(iterator.next_request().range_dsts), 2)

        spaced = self.ring([100, 412], buffer_size=ALIGNMENT)
        self.assertEqual(len(spaced.next_request().range_dsts), 1,
                         "with the rule the second range no longer fits")

    def test_a_ring_that_is_read_into_directly_is_refused(self):
        """Congruent placement and the rule cannot both hold, and preferring one silently would give
        a peer the wrong offsets."""
        with self.assertRaisesRegex(ValueError, "cannot also be aligned"):
            self.ring([100], buffer_size=ALIGNMENT * 2, slots=HOST_SLOTS)

    def test_a_second_file_in_the_same_request_keeps_the_cursor(self):
        """next_chunks is asked per FILE, so the pad before a file's first range depends on where
        the previous file's ranges ended. A packer that restarted at zero for each file would fill a
        request the placer cannot lay out."""
        files = [FileChunks.contiguous(1, "a.txt", 0, [64]),
                 FileChunks.contiguous(2, "b.txt", 0, [100, 200])]
        iterator = FilesRequestsIteratorWithBuffer(
            ALIGNMENT * 4, 2, files, owned=True, slots=SENT_SLOTS, range_alignment=ALIGNMENT,
        )
        request = iterator.next_request()

        self.assertEqual([f.id for f in request.files], [1, 2])
        self.assertEqual(self.offsets_of(iterator, request), [0, 512, 1024])

    def test_an_alignment_that_does_not_divide_the_slot_base(self):
        """The rule is counted FROM THE BASE. A slot's address is page aligned, so an alignment that
        divides the page size cannot tell base-relative from absolute - this one cannot."""
        alignment = 1000
        iterator = self.ring([100, 200, 300], buffer_size=alignment * 4, alignment=alignment)
        request = iterator.next_request()

        base = iterator._pool.slot(request.buffer_index).address
        if base % alignment == 0:
            # A page-aligned address is a multiple of 1000 every 125 pages. Skipping says the run
            # proved nothing; failing would be a flake.
            self.skipTest("this slot address is already aligned, so the test cannot tell the two apart")

        self.assertEqual(self.offsets_of(iterator, request), [0, 1000, 2000])

    def test_a_zero_sized_range_takes_no_pad(self):
        """An empty tensor in a shard is a real case. It has no bytes to align, and a pad for it
        would spend room a real range could use."""
        iterator = self.ring([100, 0, 200], buffer_size=ALIGNMENT * 4)
        request = iterator.next_request()

        self.assertEqual(self.offsets_of(iterator, request), [0, 100, 512])

    def test_an_alignment_below_one_is_refused(self):
        for alignment in (0, -ALIGNMENT):
            with self.subTest(alignment=alignment):
                with self.assertRaisesRegex(ValueError, "at least 1"):
                    self.ring([100], buffer_size=ALIGNMENT * 2, alignment=alignment)

    def test_alignment_one_changes_nothing(self):
        iterator = self.ring([100, 200, 300], buffer_size=600, alignment=1)
        request = iterator.next_request()

        self.assertEqual(self.offsets_of(iterator, request), [0, 100, 300])


class TestTheRingAccessors(unittest.TestCase):
    """What a caller that will SEND a slot asks the ring."""

    def ring(self, sizes, alignment=ALIGNMENT):
        files = [FileChunks.contiguous(17, "a.txt", 0, sizes)]
        return FilesRequestsIteratorWithBuffer(
            ALIGNMENT * 4, 2, files, owned=True, slots=SENT_SLOTS, range_alignment=alignment,
        )

    def test_the_alignment_is_the_one_the_ring_was_built_with(self):
        self.assertEqual(self.ring([100]).range_alignment, ALIGNMENT)
        self.assertEqual(self.ring([100], alignment=1).range_alignment, 1)

    def test_the_offsets_start_at_the_slot_not_at_the_first_range(self):
        """The reason this is asked of the ring: range_dsts[0] is the layout's own first element."""
        iterator = self.ring([100, 200])
        request = iterator.next_request()

        self.assertEqual(iterator.range_offsets_of(request), [0, 512])
        # Independent of how the accessor finds the base: whatever it is, the ranges sit this far
        # apart, and the first one starts at it.
        self.assertEqual(request.range_dsts[1] - request.range_dsts[0], 512)

    def test_a_shifted_layout_is_visible_here(self):
        """A layout that did not start at the slot base reads as such, where range_dsts[0] would
        have hidden it."""
        iterator = self.ring([100, 200])
        request = iterator.next_request()
        request.range_dsts = [dst + ALIGNMENT for dst in request.range_dsts]

        self.assertEqual(iterator.range_offsets_of(request), [512, 1024])

    def test_a_released_request_has_no_offsets(self):
        iterator = self.ring([100])
        request = iterator.next_request()
        iterator.release(request)

        with self.assertRaisesRegex(RuntimeError, "already been released"):
            iterator.range_offsets_of(request)


def walk_end(sizes, alignment, room=1 << 30):
    """Where the packer's cursor ends after taking these sizes.

    Asked of the PACKER, which is the only thing that walks a layout now. It used to be a free
    helper the request loop called to re-walk ranges the packer had already walked."""
    iterator = FileChunksIterator(FileChunks.contiguous(0, "f.bin", 0, list(sizes)))
    iterator.next_chunks(room, 0, alignment)
    return iterator.cursor_after


class TestTheLayoutWalk(unittest.TestCase):
    """The rule the packer and the placer share."""

    def test_the_rule_rounds_up(self):
        self.assertEqual(next_offset(0, ALIGNMENT), 0)
        self.assertEqual(next_offset(1, ALIGNMENT), 512)
        self.assertEqual(next_offset(512, ALIGNMENT), 512)
        self.assertEqual(next_offset(513, ALIGNMENT), 1024)

    def test_an_alignment_of_one_is_a_running_sum(self):
        self.assertEqual(walk_end([100, 200, 300], 1), 600)
        self.assertEqual(next_offset(7, 1), 7)

    def test_a_zero_sized_range_neither_pads_nor_advances(self):
        self.assertEqual(advance(100, 0, ALIGNMENT), (100, 100))
        self.assertEqual(advance(100, 7, ALIGNMENT), (512, 519))

    def test_the_walk_ends_at_the_last_range(self):
        """Not at a pad after it: the end of the data is what a caller sends."""
        self.assertEqual(walk_end([100, 200], ALIGNMENT), 712)

    def test_the_cursor_is_where_laying_the_ranges_out_again_would_end(self):
        """THE SECOND WALK, kept as a CHECK rather than as production code.

        next_chunks hands back the cursor it reached while deciding what fits, and the request loop
        trusts it. This recomputes it the long way over a randomised corpus - including zero-sized
        ranges, which take no pad - so a packer whose cursor drifted from its own placement would
        be caught here rather than by a peer reading the wrong offsets."""
        rng = random.Random(20261114)
        for trial in range(200):
            sizes = [rng.choice([0, 1, 7, 100, 512, 513, 4096]) for _ in range(rng.randint(1, 40))]
            alignment = rng.choice([1, 8, 512])
            with self.subTest(trial=trial, alignment=alignment):
                expected = 0
                for size in sizes:
                    _, expected = advance(expected, size, alignment)
                self.assertEqual(walk_end(sizes, alignment), expected)

    def test_nothing_handed_out_leaves_the_cursor_where_it_was(self):
        """A file whose first range does not fit contributes nothing, and the request's cursor must
        stay where the files before it left it.

        FROM A NON-ZERO CURSOR, so this can tell a real answer from one that echoes its argument:
        at cursor 0 an implementation that always reported 0 would pass."""
        iterator = FileChunksIterator(FileChunks.contiguous(0, "f.bin", 0, [100]))
        self.assertEqual(iterator.cursor_after, 0, "nothing handed out yet")
        iterator.next_chunks(10, 64, 1)
        self.assertEqual(iterator.cursor_after, 64, "a range that did not fit moved the cursor")


class TestTheRingNeverOutgrowsItsStream(unittest.TestCase):
    """A slot the data cannot reach is device memory taken and never written.

    The largest range outranks the share a slot is sized from, so a slot can be bigger than the
    stream divided by the depth. The depth then has to follow what the stream fills."""

    def files(self, sizes):
        return [FileChunks.contiguous(0, "f.bin", 0, sizes)]

    @patch.dict(os.environ, {}, clear=True)
    def test_the_depth_follows_the_stream_when_one_range_outgrows_the_share(self):
        """An 8 GiB range beside 2 GiB of small ones: the slot is 8 GiB, which two slots already
        span, so the other two could never be filled."""
        stream = self.files([8 * GiB] + [100 << 20] * 20)
        size, slots = device_ring_sizing(100 * GiB, stream)
        self.assertEqual(size, 8 * GiB)
        self.assertEqual(slots, 2)
        self.assertLessEqual(size * slots, 2 * sum(stream[0].sizes))

    @patch.dict(os.environ, {RUNAI_STREAMER_DEVICE_RING_SLOTS_ENV_VAR_NAME: "8"}, clear=True)
    def test_a_raised_slot_count_cannot_outgrow_the_stream_either(self):
        self.assertEqual(device_ring_sizing(100 * GiB, self.files([GiB] * 3))[1], 3)

    @patch.dict(os.environ, {}, clear=True)
    def test_a_stream_that_fills_every_slot_keeps_every_slot(self):
        """The cap must not shorten a ring the data does fill: 64 GiB of 1 MiB ranges at a 4 GiB
        slot is sixteen slot-fulls, far more than the depth."""
        size, slots = device_ring_sizing(100 * GiB, self.files([1 << 20] * 65536))
        self.assertEqual(slots, DEFAULT_DEVICE_RING_SLOTS)

    @patch.dict(os.environ, {}, clear=True)
    def test_a_slot_sized_range_gets_a_slot_each_and_no_more(self):
        """Three ranges, each the whole slot: three slots fill and the fourth cannot, so the depth
        comes down to three rather than reserving a slot nothing can reach."""
        size, slots = device_ring_sizing(64 << 20, self.files([6 << 20] * 3))
        self.assertEqual((size, slots), (6 << 20, 3))


class TestTheReaderCountOnlyGrowsAFileSystemRing(unittest.TestCase):
    """The per-reader growth is for the readers that FILL the ring.

    Object storage is served by its own pool, sized by RUNAI_STREAMER_OBJ_CONCURRENCY. A file
    system reader count there grows the ring for threads nothing builds."""

    def test_an_object_storage_path_is_recognised(self):
        for path in ("s3://bucket/model/f.safetensors",
                     "gs://bucket/model/f.safetensors",
                     "az://account/model/f.safetensors"):
            with self.subTest(path=path):
                self.assertTrue(reads_object_storage(path))

    def test_a_file_system_path_is_not(self):
        for path in ("/mnt/models/f.safetensors", "relative/f.safetensors", "/s3/not-a-uri"):
            with self.subTest(path=path):
                self.assertFalse(reads_object_storage(path))
