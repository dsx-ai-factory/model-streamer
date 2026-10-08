import unittest
import tempfile
import shutil
import os
from unittest.mock import patch
from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.file_streamer.requests_iterator import (
    DIRECT_IO_BLOCK,
    FilesRequestsIteratorWithBuffer,
    MemoryCapMode,
    FileChunks,
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
)


class TestRingConcurrency(unittest.TestCase):
    """Several submissions in flight at once. Each range carries a distinct, position-encoding payload,
    so a response attributed to the wrong submission or the wrong buffer produces the wrong text rather
    than passing by luck."""

    RANGE_SIZE = 8

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write_ranges(self, name: str, count: int):
        """A file of `count` ranges, range i holding exactly "rng{i:05d}"."""
        expected = {i: f"rng{i:05d}" for i in range(count)}
        path = os.path.join(self.temp_dir, name)
        with open(path, "w") as handle:
            handle.write("".join(expected[i] for i in range(count)))
        return path, expected

    def stream(self, path: str, count: int):
        """Stream the file one range at a time, returning (results, max concurrently live submissions)."""
        results = {}
        max_live = 0
        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * count)])
            for file_id, range_index, buffer in fs.get_chunks():
                self.assertEqual(file_id, 17)
                # read the payload BEFORE resuming the generator: once we resume, the submission may
                # complete and its buffer be recycled under us
                results[range_index] = buffer.numpy().tobytes().decode("utf-8")
                max_live = max(max_live, len(fs.live_requests))
        return results, max_live

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "64",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "8"})
    def test_many_submissions_in_flight_deliver_every_range(self):
        # 8 ranges of 8 bytes and a 64 byte budget over 8 buffers: 8 bytes each, so all 8 requests are
        # submitted up front and every response has to be attributed to its own submission.
        path, expected = self.write_ranges("concurrent.txt", 8)
        results, max_live = self.stream(path, 8)
        self.assertEqual(results, expected)
        self.assertGreater(max_live, 1, "expected several submissions in flight at once")

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_owned_tensors_survive_a_later_unowned_stream(self):
        """Ownership is per stream, so the two modes must not reach each other.

        A ring is built per stream_files and carries that call's mode, and the previous ring is
        dropped - its memory living on only for whatever the caller still holds. So tensors kept from
        an owned stream cannot be written to by a later stream that recycles its own buffers.
        """
        path, expected = self.write_ranges("owned_then_not.txt", 6)
        chunks = [FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 6)]

        with FileStreamer() as fs:
            fs.stream_files(chunks, owned=True)
            self.assertTrue(fs.requests_iterator._pool._owned)
            held = {index: buffer for _, index, buffer in fs.get_chunks()}

            # A second stream, recycling its buffers, over the same ranges. If the two rings could
            # reach each other this is what would overwrite what the caller is holding.
            fs.stream_files(chunks, owned=False)
            self.assertFalse(fs.requests_iterator._pool._owned)
            for _, _, buffer in fs.get_chunks():
                buffer.numpy()[:] = 0xFF

        self.assertEqual(
            {index: buffer.numpy().tobytes().decode("utf-8") for index, buffer in held.items()},
            expected,
        )

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_an_unowned_stream_can_be_followed_by_an_owned_one(self):
        # The other order. Nothing carries over between the two rings, so the second call owns its
        # buffers whatever the first one did.
        path, expected = self.write_ranges("not_then_owned.txt", 6)
        chunks = [FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 6)]

        with FileStreamer() as fs:
            fs.stream_files(chunks, owned=False)
            borrowed = {index: buffer.numpy().tobytes().decode("utf-8")
                        for _, index, buffer in fs.get_chunks()}
            self.assertEqual(borrowed, expected)

            fs.stream_files(chunks, owned=True)
            self.assertTrue(fs.requests_iterator._pool._owned)
            held = {index: buffer for _, index, buffer in fs.get_chunks()}

        self.assertEqual(
            {index: buffer.numpy().tobytes().decode("utf-8") for index, buffer in held.items()},
            expected,
        )

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_a_device_destination_keeps_the_ring_even_when_asked_to_own(self):
        """FileStreamer is exported, so it is reached without the routing layer above it.

        Checked on the ring, which needs no GPU: nothing reaches CUDA until the first .to()."""
        path, _ = self.write_ranges("device_ring.txt", 6)
        chunks = [FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 6)]

        with FileStreamer() as fs:
            fs.stream_files(chunks, device="cuda:0", owned=True)

            self.assertEqual(fs.device_str, "cuda:0")
            self.assertFalse(fs.requests_iterator._pool._owned)

            # Nothing consumed the submission - get_chunks would need a GPU - and the C layer must
            # not be left holding destinations.
            fs.drain_live_submissions()

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_a_none_device_is_the_host_and_still_owns(self):
        """device=None has always meant the host: it reaches tensor.to(None), which returns the tensor
        unchanged, so the caller is handed a view into a ring buffer.

        It is tested because None does NOT equal "cpu", and the comparisons against "cpu" decide where
        a tensor lands and whether its buffer is handed over. Left alone, `owned=True` with
        `device=None` recycled the buffers under tensors the caller had been promised."""
        path, expected = self.write_ranges("none_device.txt", 6)
        chunks = [FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 6)]

        with FileStreamer() as fs:
            fs.stream_files(chunks, device=None, owned=True)
            self.assertEqual(fs.device_str, "cpu")
            self.assertTrue(fs.requests_iterator._pool._owned)
            held = {index: buffer for _, index, buffer in fs.get_chunks()}

        self.assertEqual(
            {index: buffer.numpy().tobytes().decode("utf-8") for index, buffer in held.items()},
            expected,
        )

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_ring_recycles_buffers_across_many_requests(self):
        # a ring of 2 serving 10 ranges: buffers must be released and refilled 5 times over, and a range
        # must never be read out of a buffer that a later request has already overwritten
        path, expected = self.write_ranges("recycled.txt", 10)
        results, max_live = self.stream(path, 10)
        self.assertEqual(results, expected)
        self.assertEqual(max_live, 2)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "32",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "4"})
    def test_deep_ring_recycles_while_other_submissions_are_still_in_flight(self):
        # The steady state of a real load, which neither of the tests above reaches: a ring of 4 serving
        # 20 ranges, so buffers are handed back and immediately re-submitted while three other
        # submissions are still filling. Covers depth AND recycling at once - a refill that stops
        # keeping the ring full stays correct, so only the max_live assertion catches it.
        path, expected = self.write_ranges("deep.txt", 20)
        results, max_live = self.stream(path, 20)
        self.assertEqual(results, expected)
        self.assertEqual(max_live, 4)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "64",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "8"})
    def test_a_second_streamer_does_not_disturb_one_still_streaming(self):
        # Streamer state belongs to the HANDLE, not the process: starting a second streamer must not
        # touch the submissions a live one is still draining. Reachable in the product, not just here -
        # FileStreamer.list_files starts a temporary streamer when used outside its context manager, so
        # listing during a stream takes this path.
        #
        # This is a MOCK-FIDELITY test as much as a product one. The mock used to keep submissions in
        # process-wide globals that runai_start cleared, so this failed against the mock ("response for
        # unknown submission 0") while passing against the real library - the direction that makes a
        # green suite meaningless.
        path, expected = self.write_ranges("two_streamers.txt", 8)
        with FileStreamer() as first:
            first.stream_files([FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 8)])
            chunks = first.get_chunks()
            results = {}
            file_id, range_index, buffer = next(chunks)
            results[range_index] = buffer.numpy().tobytes().decode("utf-8")
            self.assertGreater(len(first.live_requests), 1)

            with FileStreamer() as second:
                # a whole independent stream on the second streamer, start to finish
                second.stream_files([FileChunks.contiguous(18, path, 0, [self.RANGE_SIZE] * 8)])
                second_results = {}
                for _file_id, index, second_buffer in second.get_chunks():
                    second_results[index] = second_buffer.numpy().tobytes().decode("utf-8")
                self.assertEqual(second_results, expected)

            # the first streamer must still deliver every remaining range, correctly
            for _file_id, index, buffer in chunks:
                results[index] = buffer.numpy().tobytes().decode("utf-8")
            self.assertEqual(results, expected)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "64",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "8"})
    def test_abandoning_the_generator_drains_every_live_submission(self):
        # the consumer walks away with several submissions still in flight. Every one of them must be
        # drained before the streamer is torn down, or a late write lands in freed memory.
        path, _ = self.write_ranges("abandoned.txt", 8)
        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 8)])
            chunks = fs.get_chunks()
            next(chunks)
            self.assertGreater(len(fs.live_requests), 1)
            chunks.close()     # resumes the generator at its yield, running the finally
            self.assertEqual(fs.outstanding, 0)
            self.assertEqual(fs.live_requests, {})

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "64",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "8"})
    def test_streaming_again_before_draining_is_rejected(self):
        # stream_files replaces the requests iterator, which drops the only reference to the previous
        # ring's pool while the C++ layer still holds raw destination pointers into it. Left unguarded
        # that is a use after free, and it surfaces (if at all) as a confusing unknown-submission error
        # much later, so it has to fail here instead - see the comment on stream_files.
        path, _ = self.write_ranges("undrained.txt", 8)
        request = [FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 8)]
        with FileStreamer() as fs:
            fs.stream_files(request)
            # bind the generator: an unreferenced one is collected as soon as next() returns, and
            # closing it runs the drain - which would leave nothing for the guard to catch
            chunks = fs.get_chunks()
            next(chunks)                   # submissions live, nothing drained
            self.assertGreater(fs.outstanding, 0)

            with self.assertRaises(ValueError) as caught:
                fs.stream_files(request)
            self.assertIn("outstanding", str(caught.exception))

            # the rejected call must leave the streamer exactly as it was, or the guard would turn a
            # recoverable mistake into the corruption it exists to prevent
            self.assertGreater(fs.outstanding, 0)
            self.assertGreater(len(fs.live_requests), 0)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "64",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "8"})
    def test_streaming_again_after_draining_is_allowed(self):
        # The other half of the guard: abandoning a stream is legitimate, because closing the generator
        # runs the drain. A guard that also blocked THIS would make an abandoned stream unrecoverable.
        path, expected = self.write_ranges("redrained.txt", 8)
        request = [FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 8)]
        with FileStreamer() as fs:
            fs.stream_files(request)
            chunks = fs.get_chunks()
            next(chunks)
            chunks.close()

            fs.stream_files(request)       # drained, so this is fine
            results = {}
            for _file_id, range_index, buffer in fs.get_chunks():
                results[range_index] = buffer.numpy().tobytes().decode("utf-8")
            # and the second stream is correct, not just accepted
            self.assertEqual(results, expected)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "64",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "8"})
    def test_a_generator_outliving_its_context_does_not_brick_the_streamer(self):
        # The drain cannot consume anything once __exit__ has cleared the handle, so it returns with
        # the counter still raised. stream_files reads that same counter and sets it only AFTER the
        # guard, so a stale value rejects every later stream on this object - permanently. Zeroing is
        # safe here precisely because runai_end has already joined the workers: nothing is in flight.
        path, expected = self.write_ranges("outlived.txt", 8)
        request = [FileChunks.contiguous(17, path, 0, [self.RANGE_SIZE] * 8)]

        fs = FileStreamer()
        with fs:
            fs.stream_files(request)
            chunks = fs.get_chunks()
            next(chunks)
            self.assertGreater(fs.outstanding, 0)
        # the context is gone and the handle with it; the generator is still alive, and closing it now
        # runs the finally against a streamer that no longer exists
        chunks.close()
        self.assertEqual(fs.outstanding, 0)

        with fs:
            fs.stream_files(request)       # re-entered, so this must be accepted
            results = {}
            for _file_id, range_index, buffer in fs.get_chunks():
                results[range_index] = buffer.numpy().tobytes().decode("utf-8")
            self.assertEqual(results, expected)

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_a_consumer_that_gives_up_leaves_the_streamer_usable(self):
        """The sibling test closes the generator; this one lets a consumer raise, which is what
        safetensors_pytorch does, and then uses the streamer again.

        Reuse is the part that matters: a drain that only zeroed the counters would pass the
        assertions below and still hand the next stream a ring the C layer is writing into."""
        path, expected = self.write_ranges("abandoned.bin", 12)
        chunks = [FileChunks.contiguous(0, path, 0, [self.RANGE_SIZE] * 12)]

        with FileStreamer() as streamer:
            streamer.stream_files(chunks)
            with self.assertRaises(RuntimeError):
                for _ in streamer.get_chunks():
                    raise RuntimeError("consumer gave up")

            self.assertEqual(streamer.outstanding, 0, "submissions were left in flight")
            self.assertEqual(streamer.live_requests, {})

            # The assertions above are bookkeeping; this is the behaviour they stand for.
            streamer.stream_files(chunks)
            again = {i: b.numpy().tobytes().decode("utf-8")
                     for _, i, b in streamer.get_chunks()}

        self.assertEqual(again, expected)


class TestBindings(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_runai_library(self):
        # Prepare file with content
        file_content = "XTest Text1TestText2Test-Text3\n"
        file_id = 17
        file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(file_path, "w") as file:
            file.write(file_content)

        request_sizes = [10, 0, 9, 10]
        id_to_results = {
            0: {"expected_text": "Test Text1"},
            1: {"expected_text": ""},
            2: {"expected_text": "TestText2"},
            3: {"expected_text": "Test-Text3"},
        }
        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(file_id, file_path, 1, request_sizes)])
            seen = set()
            for res_file_id, id, dst in fs.get_chunks():
                seen.add(id)
                self.assertEqual(res_file_id, file_id)
                self.assertEqual(
                    dst.numpy().tobytes().decode("utf-8"),
                    id_to_results[id]["expected_text"],
                )
            # every range produced exactly one response (full drain, incl. the zero-size range)
            self.assertEqual(seen, set(id_to_results))

    @patch("runai_model_streamer.file_streamer.requests_iterator._get_memory_mode")
    def test_min_memory_cap(self, mock_get_memory_mode):
        mock_get_memory_mode.return_value = MemoryCapMode.largest_chunk
        file_content = "XTest Text1TestText2Test-Text3\n"
        file_id = 17
        file_path = os.path.join(self.temp_dir, "min_test_file.txt")
        with open(file_path, "w") as file:
            file.write(file_content)

        request_sizes = [10, 9, 10]
        id_to_results = {
            0: {"expected_text": "Test Text1"},
            1: {"expected_text": "TestText2"},
            2: {"expected_text": "Test-Text3"},
        }
        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(file_id, file_path, 1, request_sizes)])
            seen = set()
            for res_file_id, id, dst in fs.get_chunks():
                seen.add(id)
                self.assertEqual(res_file_id, file_id)
                self.assertEqual(
                    dst.numpy().tobytes().decode("utf-8"),
                    id_to_results[id]["expected_text"],
                )
            # every range produced exactly one response (full drain)
            self.assertEqual(seen, set(id_to_results))

    # Patch the ENVIRONMENT, not os.getenv itself. Patching the lookup function makes every variable
    # answer the same value, so this test used to set the ring's buffer size and buffer-count floor to 6
    # as well as the memory limit - silently, and only because requests_iterator grew more variables.
    # The real _get_memory_mode derives `limited` from 6 on its own, so it needs no patch either.
    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "6"})
    def test_limited_memory_cap(self):
        file_content = "XABBCCCDDDDEEEEEFFFFGGGHHI"
        file_id = 17
        file_path = os.path.join(self.temp_dir, "limited_test_file.txt")
        with open(file_path, "w") as file:
            file.write(file_content)

        request_sizes = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        id_to_results = {
            0: {"expected_text": "A"},
            1: {"expected_text": "BB"},
            2: {"expected_text": "CCC"},
            3: {"expected_text": "DDDD"},
            4: {"expected_text": "EEEEE"},
            5: {"expected_text": "FFFF"},
            6: {"expected_text": "GGG"},
            7: {"expected_text": "HH"},
            8: {"expected_text": "I"},
        }
        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(file_id, file_path, 1, request_sizes)])
            seen = set()
            for res_file_id, id, dst, in fs.get_chunks():
                seen.add(id)
                self.assertEqual(res_file_id, file_id)
                self.assertEqual(
                    dst.numpy().tobytes().decode("utf-8"),
                    id_to_results[id]["expected_text"],
                )
            # every range produced exactly one response (full drain across multiple buffer requests)
            self.assertEqual(seen, set(id_to_results))

    def test_empty_file_first_does_not_drop_the_rest(self):
        # end-to-end regression for issue #157: an empty shard ahead of a data shard used to close the
        # request, which get_chunks reads as end of stream, silently dropping every remaining file
        empty_path = os.path.join(self.temp_dir, "empty.txt")
        open(empty_path, "w").close()
        data_path = os.path.join(self.temp_dir, "data.txt")
        with open(data_path, "w") as file:
            file.write("HelloWorld")

        with FileStreamer() as fs:
            fs.stream_files([
                FileChunks(17, empty_path, [], []),
                FileChunks.contiguous(18, data_path, 0, [5, 5]),
            ])
            results = [(file_id, index, dst.numpy().tobytes().decode("utf-8"))
                       for file_id, index, dst in fs.get_chunks()]

        self.assertEqual(sorted(results), [(18, 0, "Hello"), (18, 1, "World")])

    def test_file_of_only_zero_sized_ranges_is_answered(self):
        # A file whose tensors are all zero sized is not an empty shard - it has header entries, and
        # safetensors yields those tensors. Each zero-sized range must still get its own response, or
        # the caller's tensor indexing drifts. This used to be skipped in Python because the C++ layer
        # could not answer a zero-sized range (an empty batch range never completed its tasks).
        file_path = os.path.join(self.temp_dir, "zeros.txt")
        with open(file_path, "w") as file:
            file.write("content")

        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(17, file_path, 0, [0, 0, 0])])
            results = [(file_id, index, len(dst.numpy().tobytes()))
                       for file_id, index, dst in fs.get_chunks()]

        self.assertEqual(sorted(results), [(17, 0, 0), (17, 1, 0), (17, 2, 0)])

    def test_get_chunks_raises_on_read_error(self):
        # FileStreamer is single-submission and fail-fast: a per-sub-range error from the streamer must raise
        # out of get_chunks (the low-level binding no longer raises, so FileStreamer itself enforces this).
        # Requesting more bytes than the file holds makes the read fall short -> a per-sub-range error.
        file_id = 17
        file_path = os.path.join(self.temp_dir, "small.txt")
        with open(file_path, "w") as file:
            file.write("short")   # 5 bytes

        with FileStreamer() as fs:
            fs.stream_files([FileChunks.contiguous(file_id, file_path, 0, [20])])   # one 20-byte range from a 5-byte file
            with self.assertRaises(ValueError):
                list(fs.get_chunks())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestObjectStorageSkipsTheProbe(unittest.TestCase):
    """An object-storage submission must not probe for a direct-I/O block.

    There is no mount behind an s3:// path, so there is nothing to measure. It takes the other branch
    and uses the page size for layout.

    Every other test here streams local files, so this branch had no coverage at all - which is how a
    plain NameError on DIRECT_IO_BLOCK reached review. The name was used but never imported, so any
    object-storage submission raised before it read a byte.
    """

    def test_object_storage_uses_the_default_block_and_does_not_probe(self):
        seen = {}

        def capture(requests, memory_limit, direct_block, owned):
            seen["direct_block"] = direct_block
            seen["owned"] = owned
            raise _Stop()

        with FileStreamer() as fs:
            with patch.object(FileStreamer, "handle_object_store",
                              side_effect=lambda path, streamer, credentials: path), \
                 patch("runai_model_streamer.file_streamer.file_streamer.runai_probe_direct_block_size") as probe, \
                 patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", side_effect=capture):
                # Stopped as soon as the block is decided: building the ring and submitting would need
                # a real bucket, and the block is the whole question here.
                with self.assertRaises(_Stop):
                    fs.stream_files([FileChunks.contiguous(17, "s3://bucket/model.safetensors", 0, [8])])

        self.assertEqual(seen["direct_block"], DIRECT_IO_BLOCK)
        # The contract reaches the ring by default: nothing between here and the caller has to
        # ask for owned tensors.
        self.assertTrue(seen["owned"])
        probe.assert_not_called()


class _Stop(Exception):
    """Ends stream_files once the value under test has been captured."""


if __name__ == "__main__":
    unittest.main()
