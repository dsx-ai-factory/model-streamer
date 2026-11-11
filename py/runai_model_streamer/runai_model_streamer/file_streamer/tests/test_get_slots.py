# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from runai_model_streamer.file_streamer import file_streamer as file_streamer_module
from runai_model_streamer.file_streamer.file_streamer import FileStreamer, device_slots
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    FilesRequestsIteratorWithBuffer,
    RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME,
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
)

RANGE_SIZE = 8


class TestGetSlots(unittest.TestCase):
    """Whole slots instead of ranges, which is what the distributed path broadcasts."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def write_ranges(self, name, count):
        """A file of `count` ranges, range i holding "rng{i:05d}", which is RANGE_SIZE bytes.

        Every range says which one it is, so a slot handed over out of turn cannot read as correct."""
        expected = [f"rng{i:05d}".encode() for i in range(count)]
        path = os.path.join(self.temp_dir, name)
        with open(path, "wb") as handle:
            handle.write(b"".join(expected))
        return path, expected

    def collect(self, chunks, count):
        """(bytes of each slot, ranges per slot, the size the ring chose).

        Drives the generator with send(None): a host ring needs no event, and this is where a device
        ring hands one back."""
        slots, shapes = [], []
        with FileStreamer() as fs:
            fs.stream_files(chunks)
            buffer_size = fs.requests_iterator.buffer_size
            generator = fs.get_slots()
            try:
                request, memory = generator.send(None)
                while True:
                    slots.append(bytes(memoryview(memory)))
                    shapes.append(sum(len(fc.sizes) for fc in request.files))
                    request, memory = generator.send(None)
            except StopIteration:
                pass
        self.assertEqual(sum(shapes), count, "some range never reached a slot")
        return slots, shapes, buffer_size

    # Four ranges per slot and two slots, so submissions INTERLEAVE and completion order stops being
    # submission order - which is the ordering get_slots has to impose.
    SMALL_RING = {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: str(RANGE_SIZE * 4 * 2),
                  RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2",
                  RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME: "0"}

    @patch.dict(os.environ, SMALL_RING)
    def test_the_slots_hold_the_file_in_order(self):
        path, expected = self.write_ranges("ordered.bin", 12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        slots, shapes, _ = self.collect(chunks, 12)

        # The bytes, not only the shape: a slot handed over in the wrong order still has the right
        # SIZE, so sizes alone would pass on exactly the failure this ordering exists to prevent.
        self.assertEqual(b"".join(slots), b"".join(expected),
                         "the slots did not reassemble the file in order")
        self.assertGreater(len(slots), 1, "the setup fitted everything in one slot")

    @patch.dict(os.environ, SMALL_RING)
    def test_a_slot_carries_every_range_of_its_request(self):
        path, _ = self.write_ranges("full.bin", 12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        slots, shapes, buffer_size = self.collect(chunks, 12)

        self.assertEqual([len(slot) for slot in slots], [n * RANGE_SIZE for n in shapes],
                         "a slot's bytes are its ranges, with nothing else in them")
        for length in (len(slot) for slot in slots):
            self.assertLessEqual(length, buffer_size)

    @patch.dict(os.environ, SMALL_RING)
    def test_a_stream_shorter_than_the_ring(self):
        """A stream the ring spans rather than recycles, so it picks a different shape."""
        path, expected = self.write_ranges("one.bin", 3)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 3)]
        slots, _, _ = self.collect(chunks, 3)

        self.assertEqual(b"".join(slots), b"".join(expected))

    @patch.dict(os.environ, SMALL_RING)
    def test_several_files(self):
        first, head = self.write_ranges("a.bin", 5)
        second, tail = self.write_ranges("b.bin", 6)
        chunks = [FileChunks.contiguous(0, first, 0, [RANGE_SIZE] * 5),
                  FileChunks.contiguous(1, second, 0, [RANGE_SIZE] * 6)]
        slots, _, _ = self.collect(chunks, 11)

        self.assertEqual(b"".join(slots), b"".join(head) + b"".join(tail))


class FakeStream:
    """A torch CUDA stream is unavailable on a host-only runner."""

    def synchronize(self) -> None:
        pass


FAKE_STREAM = FakeStream()


class FakeEvent:
    """Completes at once. What is under test is that an event reaches release() and the teardown,
    not when it fires."""

    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class TestGetSlotsOnADeviceRing(unittest.TestCase):
    """A ring whose slots need an event to be released.

    The memory is CPU tensors, so every step except the event itself is the real one."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.released = []

    def chunks(self, count=12):
        path = os.path.join(self.temp_dir, "model.bin")
        with open(path, "wb") as handle:
            handle.write(b"".join(f"rng{i:05d}".encode() for i in range(count)))
        return [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * count)]

    def ring(self, chunks):
        slots = device_slots("cpu", FakeEvent, lambda device: FAKE_STREAM)
        ring = FilesRequestsIteratorWithBuffer(
            RANGE_SIZE * 4, 2, chunks, owned=False, slots=slots,
        )
        real_release = ring.release

        def record(request, ready_event=None):
            self.released.append(ready_event)
            real_release(request, ready_event)

        ring.release = record
        return ring

    def streamer(self, chunks):
        return patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode",
                            lambda *a, **k: self.ring(chunks))

    def test_the_event_reaches_release_and_the_teardown(self):
        chunks = self.chunks()
        events = []
        with self.streamer(chunks):
            with FileStreamer() as fs:
                fs.stream_files(chunks)
                generator = fs.get_slots()
                event = None
                try:
                    while True:
                        generator.send(event)
                        event = FakeEvent()
                        events.append(event)
                except StopIteration:
                    pass
                last = fs._clone_event

        self.assertGreater(len(self.released), 1, "the ring never recycled a slot")
        self.assertTrue(all(e is not None for e in self.released[1:]),
                        "every slot after the first was released with the event it was sent")
        # The event made after a successful send is the one the NEXT send passes, so the last one
        # sent is the one that ended the stream.
        self.assertIs(last, events[-1],
                      "the teardown waits on the newest event it was given")

    def test_a_device_slot_cannot_be_released_without_an_event(self):
        """release() refuses None on a ring whose slots need one, and get_slots must not hide it."""
        chunks = self.chunks()
        with self.streamer(chunks):
            with FileStreamer() as fs:
                fs.stream_files(chunks)
                generator = fs.get_slots()
                generator.send(None)
                with self.assertRaisesRegex(ValueError, "needs an event"):
                    generator.send(None)

    def test_a_resume_after_the_context_exited_ends_the_stream(self):
        """The ring is gone, so there is nothing to describe a slot with."""
        chunks = self.chunks()
        with self.streamer(chunks):
            with FileStreamer() as fs:
                fs.stream_files(chunks)
                generator = fs.get_slots()
                generator.send(None)
            with self.assertRaises(StopIteration):
                generator.send(FakeEvent())


class TestAResumeAfterTheContextExited(unittest.TestCase):
    """__exit__ ends the C handle and clears it, and a generator can be resumed after that.

    The same hole in get_chunks is covered in test_file_streamer.py, beside the generator it
    belongs to."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def chunks(self, count=12):
        path = os.path.join(self.temp_dir, "model.bin")
        with open(path, "wb") as handle:
            handle.write(b"".join(f"rng{i:05d}".encode() for i in range(count)))
        return [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * count)]

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: str(RANGE_SIZE * 4 * 2),
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2",
                             RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME: "0"})
    def test_get_slots_ends_instead_of_asking_a_freed_handle(self):
        chunks = self.chunks()
        with FileStreamer() as fs:
            fs.stream_files(chunks)
            generator = fs.get_slots()
            next(generator)
        with self.assertRaises(StopIteration):
            next(generator)


class TestGetSlotsFailures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def chunks(self, count=8):
        path = os.path.join(self.temp_dir, "model.bin")
        with open(path, "wb") as handle:
            handle.write(b"".join(f"rng{i:05d}".encode() for i in range(count)))
        return [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * count)]

    def test_a_failed_range_fails_the_stream(self):
        chunks = self.chunks()
        with FileStreamer() as fs:
            fs.stream_files(chunks)
            with patch.object(file_streamer_module, "runai_response",
                              return_value=(1, next(iter(fs.live_requests)), 0, 0, False)):
                with self.assertRaisesRegex(ValueError, "Could not receive response"):
                    next(fs.get_slots())

    def test_a_response_for_an_unknown_submission_fails_the_stream(self):
        chunks = self.chunks()
        with FileStreamer() as fs:
            fs.stream_files(chunks)
            with patch.object(file_streamer_module, "runai_response",
                              return_value=(0, 9999, 0, 0, False)):
                with self.assertRaisesRegex(ValueError, "unknown submission"):
                    next(fs.get_slots())

    def test_abandoning_the_generator_drains_every_submission(self):
        """range_dsts are raw addresses into the ring, so a range still in flight would write into
        memory the next stream_files has already freed."""
        chunks = self.chunks()
        with FileStreamer() as fs:
            fs.stream_files(chunks)
            generator = fs.get_slots()
            next(generator)
            generator.close()

            self.assertEqual(fs.outstanding, 0)
            self.assertEqual(fs.live_requests, {})


if __name__ == "__main__":
    unittest.main()
