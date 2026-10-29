# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.file_streamer import file_streamer as file_streamer_module
from runai_model_streamer.file_streamer.file_streamer import FileStreamer, device_slots
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    FilesRequestsIteratorWithBuffer,
    HOST_SLOTS,
)

RANGE_SIZE = 8

# What a real ring records its event on. These tests run on the host, where there is no stream, and
# FakeCudaEvent.record takes it and ignores it.
FAKE_STREAM = lambda device: f"stream-of-{device}"


class FakeCudaEvent:
    """A torch CUDA event is unavailable on a host-only runner, so these tests build a "device" ring
    out of CPU tensors and pass this in as its event factory. Completing at once keeps the ring
    flowing; what is under test is that an event is recorded and handed to release(), not when it
    fires."""

    recorded = 0
    # The stream each record() was given, so a test can show WHICH device was timed.
    streams: list = []

    def record(self, stream=None) -> None:
        FakeCudaEvent.recorded += 1
        FakeCudaEvent.streams.append(stream)

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class SlowCudaEvent:
    """Never ready until something waits on it. The only way a stream using this can finish is the
    blocking fallback in _submit_while_buffers_free - so a test that completes proves it was taken."""

    synchronized = 0

    def __init__(self) -> None:
        self._done = False

    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return self._done

    def synchronize(self) -> None:
        SlowCudaEvent.synchronized += 1
        self._done = True


class TestDeviceRingPath(unittest.TestCase):
    """Drives the whole device path on the host: the ring is torch memory, so every step except the
    CUDA event itself is the real one."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        FakeCudaEvent.recorded = 0
        FakeCudaEvent.streams = []
        self.released = []

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write_ranges(self, count: int):
        expected = {i: f"rng{i:05d}" for i in range(count)}
        path = os.path.join(self.temp_dir, "model.bin")
        with open(path, "w") as handle:
            handle.write("".join(expected[i] for i in range(count)))
        return path, expected

    def stream(self, count, ring_slots, buffer_size=RANGE_SIZE * 4, num_buffers=2):
        """Stream `count` ranges through a ring made of `ring_slots`, recording every release."""
        path, expected = self.write_ranges(count)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * count)]

        def build_ring(*args, **kwargs):
            # owned=False for a device ring: its slots are REUSED, because the caller receives
            # clones. Production derives the same from `owned and device_str == "cpu"`; here the
            # factory that does so is patched out, so it is stated.
            ring = FilesRequestsIteratorWithBuffer(
                buffer_size, num_buffers, chunks,
                owned=not ring_slots.reuse_needs_event, slots=ring_slots,
            )
            real_release = ring.release

            def record_release(request, ready_event=None):
                self.released.append(ready_event)
                return real_release(request, ready_event)

            ring.release = record_release
            self.ring = ring
            return ring

        with patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build_ring):
            with FileStreamer() as fs:
                fs.stream_files(chunks)
                got = [(index, bytes(buffer.numpy().tobytes()).decode(), buffer)
                       for _, index, buffer in fs.get_chunks()]
        return got, expected

    def test_a_device_ring_yields_the_right_bytes(self):
        got, expected = self.stream(12, device_slots("cpu", FakeCudaEvent, FAKE_STREAM))
        self.assertEqual({i: text for i, text, _ in got}, expected)

    def test_the_event_is_recorded_on_the_ring_s_own_device(self):
        """WHICH DEVICE THE EVENT TIMES. A clone runs on the device of the tensor it copies, while an
        event given no stream records on whichever device is current - and it would then clear at
        once instead of when the clones finish, freeing the slot while they still read it.

        The ring binds its device when it is built, so the stream here is the one device_slots was
        given. A bare record() records None and fails this."""
        self.stream(12, device_slots("cpu", FakeCudaEvent, FAKE_STREAM))

        self.assertGreater(FakeCudaEvent.recorded, 0, "no event was recorded at all")
        self.assertEqual(set(FakeCudaEvent.streams), {FAKE_STREAM("cpu")},
                         "an event was recorded on a stream that is not the ring's device")

    def test_the_yielded_tensor_is_a_clone_not_a_view_into_the_slot(self):
        """The point of the clone. Overwriting every slot after the stream must not change a tensor
        the caller is still holding - with a view it would."""
        got, expected = self.stream(12, device_slots("cpu", FakeCudaEvent, FAKE_STREAM))
        kept = {i: buffer.clone() for i, _, buffer in got}     # what the caller saw, snapshotted

        for slot in self.ring._pool._slots:
            if slot is not None:
                slot.memory.fill_(0xAB)

        for i, _, buffer in got:
            self.assertTrue(torch.equal(buffer, kept[i]),
                            f"range {i} changed when its slot was overwritten - it was a view")

    def test_every_device_release_carries_an_event(self):
        self.stream(12, device_slots("cpu", FakeCudaEvent, FAKE_STREAM))
        self.assertGreater(len(self.released), 0, "no slot was released at all")
        self.assertTrue(all(isinstance(e, FakeCudaEvent) for e in self.released))
        self.assertEqual(FakeCudaEvent.recorded, len(self.released),
                         "an event must be RECORDED, not just constructed")

    def test_get_chunks_before_stream_files_is_an_empty_generator(self):
        """It asks the ring what it is made of, and there is no ring until stream_files has run."""
        with FileStreamer() as fs:
            self.assertEqual(list(fs.get_chunks()), [])

    def test_a_slot_that_only_frees_on_a_wait_still_streams_everything(self):
        """The fallback is a CORRECTNESS path, not just a deadlock guard. Without it, releasing the
        last live request into a pending state submits nothing, the response loop ends because
        live_requests is empty, and the stream finishes short - no hang, just missing data."""
        SlowCudaEvent.synchronized = 0
        got, expected = self.stream(12, device_slots("cpu", SlowCudaEvent, FAKE_STREAM))
        self.assertEqual({i: text for i, text, _ in got}, expected,
                         "the stream ended short - the blocking fallback was not taken")
        self.assertGreater(SlowCudaEvent.synchronized, 0, "no slot was ever waited on")

    def test_a_teardown_mid_stream_ends_cleanly_with_slots_pending(self):
        """runai_response returns None when the library is torn down. Slots released before that are
        pending on events, and nothing must wait on them: the generator returns, and the drain that
        follows breaks on the same None rather than looping."""
        path, _ = self.write_ranges(12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]

        def build_ring(*args, **kwargs):
            return FilesRequestsIteratorWithBuffer(
                RANGE_SIZE * 4, 2, chunks, owned=False, slots=device_slots("cpu", SlowCudaEvent, FAKE_STREAM)
            )

        real_response = file_streamer_module.runai_response
        calls = {"n": 0}

        def response_then_teardown(streamer, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 5:
                return None                     # the library has gone away
            return real_response(streamer, *args, **kwargs)

        with patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build_ring), \
                patch.object(file_streamer_module, "runai_response", response_then_teardown):
            with FileStreamer() as fs:
                fs.stream_files(chunks)
                got = list(fs.get_chunks())

        self.assertLess(len(got), 12, "the teardown should have cut the stream short")

    def test_a_read_error_propagates_and_does_not_hang(self):
        """A device ring leaves slots pending behind it. The error must still come out, and the
        teardown must not wait on an event nobody will ever record."""
        missing = os.path.join(self.temp_dir, "absent.bin")
        chunks = [FileChunks.contiguous(0, missing, 0, [RANGE_SIZE] * 4)]

        def build_ring(*args, **kwargs):
            return FilesRequestsIteratorWithBuffer(
                RANGE_SIZE * 2, 2, chunks, owned=False, slots=device_slots("cpu", FakeCudaEvent, FAKE_STREAM)
            )

        with patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build_ring):
            with self.assertRaises(ValueError):
                with FileStreamer() as fs:
                    fs.stream_files(chunks)
                    list(fs.get_chunks())

    def test_a_host_ring_releases_without_an_event(self):
        got, expected = self.stream(12, HOST_SLOTS)
        self.assertEqual({i: text for i, text, _ in got}, expected)
        self.assertTrue(all(e is None for e in self.released))


if __name__ == "__main__":
    unittest.main()
