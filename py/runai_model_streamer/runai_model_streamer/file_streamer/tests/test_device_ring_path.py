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


class FakeStream:
    """What a real ring records its event on, and what __exit__ waits for. A torch CUDA stream is
    unavailable on a host-only runner, so this stands in for one.

    ONE OBJECT PER DEVICE, so a test can say which stream an event was recorded on."""

    _by_device: dict = {}

    def __init__(self, device) -> None:
        self.device = device

    def synchronize(self) -> None:
        """Here only so a test can show it is NOT called: the wait is on an event."""

    @classmethod
    def of(cls, device) -> "FakeStream":
        return cls._by_device.setdefault(device, cls(device))


FAKE_STREAM = FakeStream.of


class FakeCudaEvent:
    """A torch CUDA event is unavailable on a host-only runner, so these tests build a "device" ring
    out of CPU tensors and pass this in as its event factory. Completing at once keeps the ring
    flowing; what is under test is that an event is recorded and handed to release(), not when it
    fires."""

    recorded = 0
    # The stream each record() was given, so a test can show WHICH device was timed.
    streams: list = []

    def __init__(self) -> None:
        # Its OWN count as well as the class total: the copy event is recorded again and again,
        # while a slot's event is recorded once.
        self.records = 0

    def record(self, stream=None) -> None:
        FakeCudaEvent.recorded += 1
        self.records += 1
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

    def recording_ring(self, chunks, buffer_size=RANGE_SIZE * 4, num_buffers=2):
        """A device ring that records every release, for a test that builds its own rather than
        going through stream()."""
        ring = FilesRequestsIteratorWithBuffer(
            buffer_size, num_buffers, chunks, owned=False,
            slots=device_slots("cpu", FakeCudaEvent, FAKE_STREAM),
        )
        real_release = ring.release

        def record_release(request, ready_event=None):
            self.released.append(ready_event)
            return real_release(request, ready_event)

        ring.release = record_release
        return ring

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
        self.assertTrue(all(event.records == 1 for event in self.released),
                        "a slot's event is recorded ONCE: the pool sweeps pending slots in release "
                        "order, which holds only while their events were recorded in that order")

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

    def test_leaving_the_context_waits_for_the_copies_before_dropping_the_ring(self):
        """A slot is freed into the pool of the stream that ALLOCATED it, which is the one
        stream_files ran on. A caller that copied out of the ring on another stream would otherwise
        have a slot handed to the next allocation there while those copies still read it.

        The ORDER is what is asserted, not only the wait: a wait after the ring is dropped is a wait
        for memory that is already back in the pool."""
        path, _ = self.write_ranges(12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        held_streamer = []
        ring_at_wait = []
        waited_on = []

        def build_ring(*args, **kwargs):
            return self.recording_ring(chunks)

        with patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build_ring), \
                patch.object(FakeCudaEvent, "synchronize",
                             lambda event: (ring_at_wait.append(held_streamer[0].requests_iterator),
                                            waited_on.append(event))):
            with FileStreamer() as fs:
                held_streamer.append(fs)
                fs.stream_files(chunks)
                list(fs.get_chunks())

        self.assertEqual(len(ring_at_wait), 1,
                         "__exit__ did not wait on the event the copies were recorded against")
        self.assertIsNotNone(ring_at_wait[0], "the ring was already dropped when the wait ran")
        self.assertIs(waited_on[0], self.released[-1],
                      "the wait was not on the event of the LAST slot released, which is the only "
                      "one recorded after every copy out of the ring")

    def test_a_stream_that_stops_part_way_waits_on_the_stream_instead(self):
        """A slot that is still TAKEN has no event: a slot is released when its LAST range is handed
        over. The newest event belongs to an earlier slot, and in stream order it completes before
        the copies out of the taken one - so waiting on it would prove nothing about them.

        Here the first slot has been released and the second has not, which is the case that tells
        the two apart."""
        path, _ = self.write_ranges(12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        waits = []

        def build_ring(*args, **kwargs):
            return self.recording_ring(chunks)

        with patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build_ring), \
                patch.object(FakeCudaEvent, "synchronize", lambda _event: waits.append("event")), \
                patch.object(FakeStream, "synchronize", lambda _stream: waits.append("stream")):
            with FileStreamer() as fs:
                fs.stream_files(chunks)
                tensors = fs.get_chunks()
                for _ in range(5):            # four ranges fill the first slot, the fifth starts the next
                    next(tensors)
                self.assertEqual(len(self.released), 1, "the setup released no slot, or released both")
                tensors.close()

        self.assertEqual(waits, ["stream"],
                         "the teardown waited on an event that is older than the copies out of the "
                         "slot that was never released")

    def test_a_second_stream_waits_for_the_copies_before_replacing_the_ring(self):
        """The same hazard at the other place a ring is dropped. stream_files frees the previous ring
        before sizing the next one, and the guard it relies on reads the LIBRARY - the copies out of
        that ring are only enqueued, so nothing there can see them."""
        path, _ = self.write_ranges(12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        held_streamer = []
        ring_at_wait = []

        def build_ring(*args, **kwargs):
            return self.recording_ring(chunks)

        with patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build_ring), \
                patch.object(FakeCudaEvent, "synchronize",
                             lambda _event: ring_at_wait.append(held_streamer[0].requests_iterator)):
            with FileStreamer() as fs:
                held_streamer.append(fs)
                fs.stream_files(chunks)
                list(fs.get_chunks())
                first_ring = fs.requests_iterator
                fs.stream_files(chunks)                 # the second stream replaces it
                self.assertIsNot(fs.requests_iterator, first_ring, "the ring was not replaced")
                list(fs.get_chunks())

        # One wait for the replacement and one on the way out. The first is the one under test, and
        # the ring it names is the one being replaced.
        self.assertEqual(len(ring_at_wait), 2,
                         "stream_files replaced the ring without waiting for its copies")
        self.assertIs(ring_at_wait[0], first_ring,
                      "the wait ran after the previous ring had already been dropped")

    def test_moving_to_another_stream_part_way_through_is_refused(self):
        """One event per slot can only mark the copies on one stream. A caller that switched would
        leave the copies on the first stream unwaited for, the slot refilled under them, and the
        tensors it is holding changed - with nothing raised. So it is raised here instead."""
        path, _ = self.write_ranges(12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        moved = {"n": 0}

        def wandering_stream(device):
            """The caller's stream, which changes once the first slot has been released."""
            return FAKE_STREAM(device if moved["n"] == 0 else "elsewhere")

        def build_ring(*args, **kwargs):
            ring = FilesRequestsIteratorWithBuffer(
                RANGE_SIZE * 4, 2, chunks, owned=False,
                slots=device_slots("cpu", FakeCudaEvent, wandering_stream),
            )
            real_release = ring.release

            def move_after_release(request, ready_event=None):
                moved["n"] += 1
                return real_release(request, ready_event)

            ring.release = move_after_release
            return ring

        with patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build_ring):
            with self.assertRaises(ValueError) as caught:
                with FileStreamer() as fs:
                    fs.stream_files(chunks)
                    list(fs.get_chunks())

        self.assertIn("another stream", str(caught.exception))

    def test_a_host_ring_waits_for_nothing(self):
        """Nothing is enqueued out of a host slot: the copy out of it finishes before tensor.to()
        returns, so there is no event to record and nothing to wait for."""
        waits = []
        with patch.object(FakeCudaEvent, "synchronize", lambda event: waits.append(event)), \
                patch.object(FakeStream, "synchronize", lambda stream: waits.append(stream)):
            self.stream(12, HOST_SLOTS)
        self.assertEqual(waits, [], "a host ring waited on its way out")

    def test_a_host_ring_releases_without_an_event(self):
        got, expected = self.stream(12, HOST_SLOTS)
        self.assertEqual({i: text for i, text, _ in got}, expected)
        self.assertTrue(all(e is None for e in self.released))


if __name__ == "__main__":
    unittest.main()
