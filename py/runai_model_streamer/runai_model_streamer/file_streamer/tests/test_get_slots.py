# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.prediction import predicted_slots
from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
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
        """A file of `count` ranges, range i holding exactly "rng{i:05d}", which is RANGE_SIZE bytes."""
        expected = [f"rng{i:05d}".encode() for i in range(count)]
        path = os.path.join(self.temp_dir, name)
        with open(path, "wb") as handle:
            handle.write(b"".join(expected))
        return path, expected

    def collect(self, chunks, count):
        """(bytes of each slot, ranges per slot) for a whole stream.

        Drives the generator with send(None): a host ring needs no event, and this is where a device
        ring would hand one back."""
        slots, shapes = [], []
        with FileStreamer() as fs:
            fs.stream_files(chunks)
            # The size the ring CHOSE, not the one the limit suggests: a stream smaller than the
            # limit is spanned instead, so the two differ and the prediction has to be asked about
            # the shape that actually exists.
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

    def assert_matches_prediction(self, chunks, shapes, buffer_size):
        predicted = [len(slot) for slot in predicted_slots(chunks, 1, 0, buffer_size)]
        self.assertEqual(shapes, predicted,
                         "the slots the ring produced are not the ones a peer would predict")

    # Four ranges per slot and two slots, so submissions INTERLEAVE and completion order stops being
    # submission order - which is the ordering get_slots has to impose.
    SMALL_RING = {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: str(RANGE_SIZE * 4 * 2),
                  RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2",
                  RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME: "0"}

    @patch.dict(os.environ, SMALL_RING)
    def test_the_slots_hold_the_file_in_order(self):
        path, expected = self.write_ranges("ordered.bin", 12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        slots, shapes, size = self.collect(chunks, 12)
        self.assert_matches_prediction(chunks, shapes, size)

        # The bytes, not only the shape: a slot handed over in the wrong order still has the right
        # SIZE, so sizes alone would pass on exactly the failure this ordering exists to prevent.
        self.assertEqual(b"".join(slots), b"".join(expected),
                         "the slots did not reassemble the file in order")
        self.assertGreater(len(slots), 1, "the setup fitted everything in one slot")

    @patch.dict(os.environ, SMALL_RING)
    def test_the_slot_boundaries_are_the_ones_the_prediction_expects(self):
        """The whole point: a peer computes these boundaries without being told them."""
        path, _ = self.write_ranges("predicted.bin", 12)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 12)]
        _, shapes, size = self.collect(chunks, 12)
        self.assert_matches_prediction(chunks, shapes, size)
        self.assertGreater(len(shapes), 1, "the setup did not produce a boundary to agree about")

    @patch.dict(os.environ, SMALL_RING)
    def test_a_stream_shorter_than_the_ring(self):
        """A stream the ring spans rather than recycles. It picks a different shape here - the span
        search buys buffers larger than budget // N - so this is also a case where a hard coded
        expectation would be wrong and the prediction still has to agree."""
        path, expected = self.write_ranges("one.bin", 3)
        chunks = [FileChunks.contiguous(0, path, 0, [RANGE_SIZE] * 3)]
        slots, shapes, size = self.collect(chunks, 3)
        self.assert_matches_prediction(chunks, shapes, size)
        self.assertEqual(b"".join(slots), b"".join(expected))

    @patch.dict(os.environ, SMALL_RING)
    def test_several_files(self):
        first, head = self.write_ranges("a.bin", 5)
        second, tail = self.write_ranges("b.bin", 6)
        chunks = [FileChunks.contiguous(0, first, 0, [RANGE_SIZE] * 5),
                  FileChunks.contiguous(1, second, 0, [RANGE_SIZE] * 6)]
        slots, shapes, size = self.collect(chunks, 11)
        self.assert_matches_prediction(chunks, shapes, size)
        self.assertEqual(b"".join(slots), b"".join(head) + b"".join(tail))


if __name__ == "__main__":
    unittest.main()
