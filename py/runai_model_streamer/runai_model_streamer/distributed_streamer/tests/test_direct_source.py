# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.direct_source import (
    DirectDistributionSource,
)
from runai_model_streamer.distributed_streamer.distribution_source import (
    RANGE_OFFSET_ALIGNMENT,
)
from runai_model_streamer.distributed_streamer.tests.test_distribution_source import (
    DistributionSourceContract,
)
from runai_model_streamer.file_streamer import file_streamer as file_streamer_module
from runai_model_streamer.file_streamer.device_support import DeviceSupport
from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    FilesRequestsIteratorWithBuffer,
    SlotMemory,
    _alloc_slot,
)

RANGE_SIZE = 100

class FakeStream:
    """A torch CUDA stream is unavailable on a host-only runner."""

    def synchronize(self) -> None:
        pass


class FakeEvent:
    """Completes at once. What is under test is that the slot is given back WITH an event, not when
    that event fires."""

    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


# The shape a sent-from ring really has: nothing reads into it directly, and its slots are refilled
# by the library's own stream, so giving one back takes an event.
SENT_SLOTS = SlotMemory(_alloc_slot, direct_io=False, label="TEST", new_event=FakeEvent,
                        current_stream=lambda device=None: FakeStream())

# This rank's numbering -> the caller's. Neither the identity nor order preserving, so a source that
# forgot to convert cannot pass by accident.
ORIGINALS = {
    0: {0: (7, 3, RANGE_SIZE), 1: (7, 1, RANGE_SIZE), 2: (2, 0, RANGE_SIZE)},
    1: {0: (7, 2, RANGE_SIZE), 1: (2, 5, RANGE_SIZE)},
}

CALLER_RANGES = {(7, 3), (7, 1), (2, 0), (7, 2), (2, 5)}


class DirectSourceFixture:
    """A real FileStreamer over a real file, with a ring placed by the rule."""

    RANGES = 5

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.exits = []

    def chunks(self):
        """TWO files, so a request spans a file boundary.

        With one file every range's flat index equals its index within the file, and every lookup
        uses the same id - so an indexing or id bug could not show. Each range holds its own byte,
        so a wrong offset reads as another range's data."""
        files = []
        for index, count in enumerate((3, 2)):
            path = os.path.join(self.temp_dir, f"model{index}.bin")
            with open(path, "wb") as handle:
                handle.write(b"".join(bytes([index * 16 + i + 1]) * RANGE_SIZE
                                      for i in range(count)))
            files.append(FileChunks.contiguous(index, path, 0, [RANGE_SIZE] * count))

        return files

    def streamer(self, alignment=RANGE_OFFSET_ALIGNMENT, buffers=2):
        """A FileStreamer whose ring is built with `alignment`. Closed by the test's cleanup."""
        chunks = self.chunks()

        def build(*args, **kwargs):
            # owned=False: a slot that is sent from is REUSED, so the ring takes it back with an
            # event. An owned pool hands its slots away and refuses one.
            return FilesRequestsIteratorWithBuffer(
                RANGE_OFFSET_ALIGNMENT * 2, buffers, chunks, owned=False,
                slots=SENT_SLOTS, range_alignment=alignment,
            )

        patcher = patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build)
        patcher.start()
        self.addCleanup(patcher.stop)

        streamer = FileStreamer()
        streamer.__enter__()
        self.addCleanup(streamer.__exit__, None, None, None)
        streamer.stream_files(chunks)
        return streamer

    def empty_streamer(self):
        """A rank whose share is empty still builds a ring, so the source is constructible."""
        def build(*args, **kwargs):
            return FilesRequestsIteratorWithBuffer(
                RANGE_OFFSET_ALIGNMENT * 2, 2, [], owned=False,
                slots=SENT_SLOTS, range_alignment=RANGE_OFFSET_ALIGNMENT,
            )

        patcher = patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build)
        patcher.start()
        self.addCleanup(patcher.stop)

        streamer = FileStreamer()
        streamer.__enter__()
        self.addCleanup(streamer.__exit__, None, None, None)
        streamer.stream_files([])
        return streamer


class TestDirectSourceHonoursTheContract(DirectSourceFixture, DistributionSourceContract,
                                         unittest.TestCase):
    def make_source(self):
        return DirectDistributionSource(self.streamer(), ORIGINALS)

    def expected_ranges(self):
        return CALLER_RANGES

    def source_depth(self):
        return DirectDistributionSource.DEPTH

    def release_event(self):
        return FakeEvent()


class TestAnEmptyShareHonoursTheContract(DirectSourceFixture, DistributionSourceContract,
                                         unittest.TestCase):
    """A rank with nothing to read still takes part in the collective, as a receiver."""

    def make_source(self):
        return DirectDistributionSource(self.empty_streamer(), ORIGINALS)

    def expected_ranges(self):
        return set()

    def source_depth(self):
        return DirectDistributionSource.DEPTH

    def release_event(self):
        return FakeEvent()


class TestTheRingStreamFilesBuilds(DirectSourceFixture, unittest.TestCase):
    """The two halves, joined.

    Every other test here builds the ring by hand. This one asks stream_files for a sent-from ring
    and feeds that to the source, so nothing stands between the flag and the thing that checks it."""

    SUPPORT = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeEvent,
                            current_stream=lambda device: FakeStream())

    def test_a_sent_from_ring_is_one_this_source_accepts(self):
        chunks = self.chunks()

        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size",
                             return_value=4096):
            with FileStreamer() as fs:
                # The shape the group agreed. Two of these ranges a slot, once they are spaced.
                fs.stream_files(chunks, device="cpu", sent_from=True,
                                slot_shape=(RANGE_OFFSET_ALIGNMENT * 2, 2))

                source = DirectDistributionSource(fs, ORIGINALS)

                got, seen = b"", []
                while True:
                    filled = source.next_slot()
                    if filled is None:
                        break
                    # A TORCH tensor, where the rings built by hand above are numpy: what a slot is
                    # made of belongs to the path that built it, and a real device ring is torch.
                    memory = filled.memory.numpy().tobytes()
                    for r in filled.ranges:
                        seen.append((r.file_id, r.range_index))
                        got += memory[r.offset:r.offset + r.size]
                    source.release(filled, FakeEvent())

        expected = b""
        for name in ("model0.bin", "model1.bin"):
            with open(os.path.join(self.temp_dir, name), "rb") as handle:
                expected += handle.read()

        self.assertEqual(set(seen), CALLER_RANGES)
        self.assertEqual(got, expected, "the slots did not reassemble the files in order")

    def test_a_ring_stream_files_did_not_mark_as_sent_from_is_refused(self):
        chunks = self.chunks()

        with patch.object(file_streamer_module, "device_support", return_value=self.SUPPORT), \
                patch.object(file_streamer_module, "device_free_memory", return_value=1 << 30), \
                patch.object(file_streamer_module, "runai_probe_direct_block_size",
                             return_value=4096):
            with FileStreamer() as fs:
                fs.stream_files(chunks, device="cpu",
                                slot_shape=(RANGE_OFFSET_ALIGNMENT * 2, 2))

                with self.assertRaisesRegex(ValueError, "wrong offsets"):
                    DirectDistributionSource(fs, ORIGINALS)

                fs.drain_live_submissions()


class TestDirectSource(DirectSourceFixture, unittest.TestCase):
    def drain(self, source):
        """Every slot, as (ranges, a copy of the bytes). Copied because the ring reuses slots."""
        slots = []
        while True:
            filled = source.next_slot()
            if filled is None:
                return slots
            slots.append((list(filled.ranges), bytes(memoryview(filled.memory))))
            source.release(filled, FakeEvent())

    def test_the_offsets_are_the_ones_the_ring_placed(self):
        """Two 100-byte ranges per slot, spaced by the rule, so the slot holds 512 + 100 bytes."""
        source = DirectDistributionSource(self.streamer(), ORIGINALS)
        slots = self.drain(source)

        self.assertEqual([[r.offset for r in ranges] for ranges, _ in slots],
                         [[0, 512], [0, 512], [0]])
        self.assertEqual([len(memory) for _, memory in slots], [612, 612, 100])

    def test_the_ranges_are_in_the_callers_numbering(self):
        source = DirectDistributionSource(self.streamer(), ORIGINALS)

        seen = [(r.file_id, r.range_index) for ranges, _ in self.drain(source) for r in ranges]
        self.assertEqual(seen, [(7, 3), (7, 1), (2, 0), (7, 2), (2, 5)])

    def test_the_bytes_are_the_ones_the_file_holds(self):
        source = DirectDistributionSource(self.streamer(), ORIGINALS)

        expected = b""
        for name in ("model0.bin", "model1.bin"):
            with open(os.path.join(self.temp_dir, name), "rb") as handle:
                expected += handle.read()

        got = b""
        for ranges, memory in self.drain(source):
            for r in ranges:
                got += memory[r.offset:r.offset + r.size]
        self.assertEqual(got, expected, "the slots did not reassemble the file in order")

    def test_a_stream_left_part_way_hands_over_the_last_event(self):
        """The event reaches the streamer at RELEASE, not at the next next_slot().

        A loop that stops when its count is done never asks for another slot. If the event waited
        for that call the streamer would never learn it, and its teardown would free the ring while
        a collective was still reading the slot."""
        streamer = self.streamer()
        source = DirectDistributionSource(streamer, ORIGINALS)

        filled = source.next_slot()
        event = FakeEvent()
        source.release(filled, event)

        self.assertIs(streamer._clone_event, event,
                      "the streamer did not learn the event of the slot just released")

    def test_a_release_without_a_slot_out_is_refused(self):
        source = DirectDistributionSource(self.streamer(), ORIGINALS)

        with self.assertRaisesRegex(RuntimeError, "no slot out"):
            source.release(None, FakeEvent())

        filled = source.next_slot()
        source.release(filled, FakeEvent())
        with self.assertRaisesRegex(RuntimeError, "no slot out"):
            source.release(filled, FakeEvent())

    def test_a_ring_that_needs_an_event_refuses_a_release_without_one(self):
        """Raised where the caller can be named, not out of some later next_slot()."""
        source = DirectDistributionSource(self.streamer(), ORIGINALS)
        filled = source.next_slot()

        with self.assertRaisesRegex(ValueError, "needs an event"):
            source.release(filled)

    def test_a_failure_ends_the_stream_instead_of_reading_as_its_end(self):
        """A generator that raised is closed. Without saying so, the next call would read its
        StopIteration as a clean end of stream and drop every slot still owed."""
        source = DirectDistributionSource(self.streamer(), ORIGINALS)
        filled = source.next_slot()

        with self.assertRaises(ValueError):
            source.release(filled)          # no event: the ring refuses it

        self.assertIsNone(source.next_slot(), "the stream is over, not merely interrupted")

    def test_a_slot_the_source_did_not_hand_out_is_refused(self):
        """Releasing a stale slot would resume the ring for a slot the caller still holds, and give
        the newest slot the event meant for an older one."""
        source = DirectDistributionSource(self.streamer(), ORIGINALS)
        first = source.next_slot()
        source.release(first, FakeEvent())
        second = source.next_slot()

        with self.assertRaisesRegex(RuntimeError, "did not just hand out"):
            source.release(first, FakeEvent())

        source.release(second, FakeEvent())

    def test_a_ring_one_slot_deep(self):
        """Release then refill is the only way to make progress, so nothing can be in flight while
        a slot is out."""
        source = DirectDistributionSource(self.streamer(buffers=1), ORIGINALS)

        seen = 0
        while True:
            filled = source.next_slot()
            if filled is None:
                break
            seen += len(filled.ranges)
            source.release(filled, FakeEvent())

        self.assertEqual(seen, 5)

    def test_a_zero_sized_range_is_described_where_the_ring_put_it(self):
        """An empty tensor in a shard. The ring gives it no pad, so the source must not invent one -
        it reports the layout, it does not compute it."""
        path = os.path.join(self.temp_dir, "zero.bin")
        with open(path, "wb") as handle:
            handle.write(b"\x01" * RANGE_SIZE + b"\x02" * RANGE_SIZE)
        files = [FileChunks(0, path, [0, RANGE_SIZE, RANGE_SIZE], [RANGE_SIZE, 0, RANGE_SIZE])]

        def build(*args, **kwargs):
            return FilesRequestsIteratorWithBuffer(
                RANGE_OFFSET_ALIGNMENT * 2, 2, files, owned=False,
                slots=SENT_SLOTS, range_alignment=RANGE_OFFSET_ALIGNMENT,
            )

        patcher = patch.object(FilesRequestsIteratorWithBuffer, "with_memory_mode", build)
        patcher.start()
        self.addCleanup(patcher.stop)

        streamer = FileStreamer()
        streamer.__enter__()
        self.addCleanup(streamer.__exit__, None, None, None)
        streamer.stream_files(files)

        source = DirectDistributionSource(streamer, {0: {0: (9, 0, RANGE_SIZE),
                                                         1: (9, 1, 0),
                                                         2: (9, 2, RANGE_SIZE)}})
        ranges, _ = self.drain(source)[0]

        self.assertEqual([(r.offset, r.size) for r in ranges],
                         [(0, RANGE_SIZE), (RANGE_SIZE, 0), (512, RANGE_SIZE)])

    def test_a_ring_placed_by_another_rule_is_refused(self):
        """A host ring places its ranges congruent with their file offsets, which a peer computing
        the layout would not find."""
        with self.assertRaisesRegex(ValueError, "wrong offsets"):
            DirectDistributionSource(self.streamer(alignment=1), ORIGINALS)

    def test_a_streamer_with_no_ring_is_refused(self):
        with FileStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "no ring"):
                DirectDistributionSource(streamer, ORIGINALS)

    def test_next_slot_refuses_while_a_slot_is_out(self):
        source = DirectDistributionSource(self.streamer(), ORIGINALS)
        filled = source.next_slot()

        with self.assertRaisesRegex(RuntimeError, "one slot"):
            source.next_slot()

        source.release(filled, FakeEvent())
        self.assertIsNotNone(source.next_slot())


if __name__ == "__main__":
    unittest.main()
