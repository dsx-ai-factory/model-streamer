"""Real-library CUDA/NCCL integration tests, launched with two torchrun ranks."""

import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from runai_model_streamer import SafetensorsStreamer


class GPUStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        cls.fixture_dir = tempfile.TemporaryDirectory(prefix="streamer-gpu-")
        cls.addClassCleanup(cls.fixture_dir.cleanup)
        # Each rank writes identical data locally. Logical shard IDs and tensor
        # names match on both ranks, so broadcasts have the same reference data.
        cls.expected = {}
        cls.paths = []
        dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.int64)
        for shard in range(3):
            tensors = {}
            for index in range(12):
                dtype = dtypes[index % len(dtypes)]
                values = torch.arange(256 * 256).reshape(256, 256) % 127
                name = f"shard_{shard}_tensor_{index}"
                tensors[name] = (values + shard + index).to(dtype)
            tensors[f"shard_{shard}_empty"] = torch.empty((0, 7))
            tensors[f"shard_{shard}_scalar"] = torch.tensor(shard, dtype=torch.int32)
            path = Path(cls.fixture_dir.name) / f"shard-{shard}.safetensors"
            save_file(tensors, str(path))
            cls.paths.append(str(path))
            cls.expected.update(tensors)
        cls.one_path = str(Path(cls.fixture_dir.name) / "one-tensor.safetensors")
        cls.one_tensor = {"only_tensor": torch.arange(1031, dtype=torch.float32)}
        save_file(cls.one_tensor, cls.one_path)

    def setUp(self):
        # Bound RAM and force multiple submissions/broadcast batches even for
        # small fixtures. Each rank gets half the 4 MiB host-buffer budget.
        settings = patch.dict(os.environ, {
            "RUNAI_STREAMER_MEMORY_LIMIT": str(4 * 1024 * 1024),
            "RUNAI_STREAMER_RING_BUFFERS": "4",
            "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": str(1024 * 1024),
            "RUNAI_STREAMER_DIST": "auto",
            "RUNAI_STREAMER_PARTITION_POLICY": "spans",
        })
        settings.start()
        self.addCleanup(settings.stop)

    def check_stream(self, paths, expected, distributed, copy_from_cpu=False):
        received = {}
        device = "cpu" if copy_from_cpu else str(self.device)
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(paths, device=device, is_distributed=distributed)
            self.assertEqual(streamer.file_streamer.is_distributed, distributed,
                             "Streamer silently changed the requested execution path")
            for name, tensor in streamer.get_tensors():
                self.assertNotIn(name, received, "Duplicate tensor")
                self.assertIn(name, expected, "Unexpected tensor")
                if copy_from_cpu:
                    self.assertEqual(tensor.device.type, "cpu")
                    tensor = tensor.to(self.device)
                self.assertEqual(tensor.device, self.device)
                # The streamer reuses its buffers. Preserve each result before
                # advancing, then validate after every batch has been consumed.
                received[name] = tensor.clone()
            torch.cuda.synchronize(self.device)
        self.assertEqual(set(received), set(expected), "Missing tensors")
        for name, reference in expected.items():
            torch.testing.assert_close(received[name].cpu(), reference, rtol=0, atol=0)

    def test_cpu_to_cuda(self):
        self.check_stream(self.paths, self.expected, False, copy_from_cpu=True)

    def test_cuda_streaming_repeated(self):
        for _ in range(3):
            self.check_stream(self.paths, self.expected, False)

    def test_nccl_chunks(self):
        with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": "chunks"}):
            self.check_stream(self.paths, self.expected, True)

    def test_nccl_files(self):
        with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": "files"}):
            self.check_stream(self.paths, self.expected, True)

    def test_nccl_spans_repeated(self):
        for _ in range(3):
            self.check_stream(self.paths, self.expected, True)

    def test_nccl_rank_without_local_data(self):
        # One tensor across two ranks leaves one rank receiving only.
        self.check_stream([self.one_path], self.one_tensor, True)


def main():
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Launch these tests with torchrun --nproc_per_node=2")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs are required; GPU tests must not skip")
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GPUStreamingTests)
        result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
        # torchrun propagates either rank's failure and terminates its peer.
        return 0 if result.wasSuccessful() and result.testsRun > 0 and not result.skipped else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
