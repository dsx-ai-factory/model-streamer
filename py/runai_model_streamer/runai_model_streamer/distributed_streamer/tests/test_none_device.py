import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.distributed_streamer.distributed_streamer import DistributedStreamer
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
    RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME,
)

RANGE_SIZE = 8

# Two buffers of one range each against six ranges, so the ring has to recycle.
#
# The pad budget has to go. A slot is buffer_size + direct_block * pads, so under the DEFAULT budget
# a slot would be 512 KiB whatever the limit says, and a recycled slot would write each range at a
# fresh offset inside it. The ring would still recycle, but nothing would ever be overwritten, and
# both checks below would then pass with or without the bug.
TIGHT_RING = {
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2",
    RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME: "0",
}


class TestOwnedFollowsTheDestination(unittest.TestCase):
    """Only a host destination hands its buffers over, and only "cpu" equals "cpu".

    This class covers the entry point where that matters. stream_files decides whether the ring is
    handed over from `owned and device == "cpu"`, so a host written any other way turned `owned=True`
    into a recycled ring under tensors the caller had been promised.

    torch.device("cpu") is the dangerous one: .to() returns the tensor unchanged, so nothing copies
    and the caller is left holding the buffer. The indexed forms copy by accident - torch keeps an
    index the CPU has no use for - which hides the fault rather than fixing it."""

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

    @patch.dict(os.environ, TIGHT_RING)
    def test_every_host_spelling_hands_the_ring_over(self):
        for device in [None, "cpu", "cpu:0", torch.device("cpu"), torch.device("cpu", 0)]:
            with self.subTest(device=repr(device)):
                path, expected = self.write_ranges(6)
                chunks = [FileChunks.contiguous(17, path, 0, [RANGE_SIZE] * 6)]

                with DistributedStreamer() as streamer:
                    streamer.stream_files(chunks, None, device=device, is_distributed=False, owned=True)
                    inner = streamer.file_streamer
                    self.assertEqual(inner.device_str, "cpu")
                    self.assertTrue(inner.requests_iterator._pool._owned)
                    held = {index: buffer for _, index, buffer in streamer.get_chunks()}

                # Every range got memory of its own. Under a recycled ring these six collapse to
                # two addresses holding the last two ranges twice over.
                self.assertEqual(len({b.data_ptr() for b in held.values()}), 6)

                self.assertEqual(
                    {i: b.numpy().tobytes().decode("utf-8") for i, b in held.items()},
                    expected,
                )

    @patch.dict(os.environ, TIGHT_RING)
    def test_a_device_destination_keeps_the_ring(self):
        """The other half of the mapping. A device destination yields tensor.to(device), a fresh
        allocation the caller already owns, so the ring stays ours to reuse however owned is set.

        Asserted on the ring, which needs no GPU: stream_files builds it, and nothing reaches CUDA
        until the first .to() in get_chunks."""
        path, _ = self.write_ranges(6)
        chunks = [FileChunks.contiguous(17, path, 0, [RANGE_SIZE] * 6)]

        with DistributedStreamer() as streamer:
            streamer.stream_files(chunks, None, device="cuda", is_distributed=False, owned=True)
            inner = streamer.file_streamer

            self.assertEqual(inner.device_str, "cuda")
            self.assertFalse(inner.requests_iterator._pool._owned)

            inner.drain_live_submissions()


if __name__ == "__main__":
    unittest.main()
