import logging
import os
import unittest
import weakref
from unittest.mock import patch
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
