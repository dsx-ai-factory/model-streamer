import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.distributed_streamer import DistributedStreamer
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
)

RANGE_SIZE = 8


class TestNoneDeviceIsTheHost(unittest.TestCase):
    """device=None has always meant the host - it reaches tensor.to(None), which returns the tensor
    unchanged - but None does NOT equal "cpu".

    This class covers the entry point where that mattered. stream_files picks the path and decides
    whether the ring is handed over, from `owned and device == "cpu"`, so an unnormalised None turned
    `owned=True` into a recycled ring under tensors the caller had been promised."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write_ranges(self, count: int):
        """A file of `count` ranges, range i holding exactly "rng{i:05d}"."""
        expected = {i: f"rng{i:05d}" for i in range(count)}
        path = os.path.join(self.temp_dir, "none_device.txt")
        with open(path, "w") as handle:
            handle.write("".join(expected[i] for i in range(count)))
        return path, expected

    @patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
                             RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2"})
    def test_a_none_device_still_hands_the_ring_over(self):
        path, expected = self.write_ranges(6)
        chunks = [FileChunks.contiguous(17, path, 0, [RANGE_SIZE] * 6)]

        with DistributedStreamer() as streamer:
            streamer.stream_files(chunks, None, device=None, is_distributed=False, owned=True)
            inner = streamer.file_streamer
            self.assertEqual(inner.device_str, "cpu")
            self.assertTrue(inner.requests_iterator._pool._owned)
            # held, not read: a recycled ring is only visible once the stream has moved past them
            held = {index: buffer for _, index, buffer in streamer.get_chunks()}

        self.assertEqual(
            {index: buffer.numpy().tobytes().decode("utf-8") for index, buffer in held.items()},
            expected,
        )


if __name__ == "__main__":
    unittest.main()
