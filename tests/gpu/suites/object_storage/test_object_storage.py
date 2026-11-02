# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Features of the streamer that need an object store, tested end to end against the SeaweedFS service of the environment.

This is the example of a suite that is not about the content of a model: each test is a feature of the streamer, seen from
the outside, with a small file that the test writes (gpu_harness.safetensors_file) and the service that the harness provides
(gpu_harness.object_store). One process, no torchrun, so it is an ordinary unittest module that suite.sh runs with
run_unittest; the results of every test are in the results files like those of any suite.

    stream      a safetensors file in a bucket is streamed to the GPU, and what arrives is what was put there
    list        list_safetensors finds the safetensors files under a prefix, and only those
    pull        pull_files downloads the files of a prefix
    missing     streaming an object that is not there is an error, not an empty model

By hand, with the service running (tests/gpu/run_local.sh --suites object_storage does all of this):

    python3 -m gpu_harness.run_unittest --suite object_storage --run basic tests/gpu/suites/object_storage/test_object_storage.py
"""

import os
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "harness"))  # gpu_harness: what every suite shares
from gpu_harness.object_store import ObjectStore  # noqa: E402
from gpu_harness.safetensors_file import small_model, tensor_bytes, write_safetensors  # noqa: E402
from gpu_harness.tensors import tensor_crc32  # noqa: E402

from runai_model_streamer import SafetensorsStreamer, list_safetensors, pull_files  # noqa: E402
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype  # noqa: E402


class ObjectStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ObjectStore.from_environment()
        cls.store.wait_until_ready(seconds=30)
        # The GPU of the job. No GPU is a failure, like everywhere in these tests; GPU_DEVICE=cpu says that the run is meant to
        # be on the CPU (the object storage is what is tested, and a machine without a GPU can do that).
        cls.device = os.environ.get("GPU_DEVICE", "cuda:0")
        if cls.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU is visible: set GPU_DEVICE=cpu to run this suite on the CPU")

    def setUp(self):
        self.prefix = self.store.new_prefix(self._testMethodName)
        self.addCleanup(self.store.delete_prefix, self.prefix)
        directory = tempfile.TemporaryDirectory(prefix="object-storage-")
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def put_small_model(self, name):
        """A small safetensors file, put under this test's prefix. Returns (its uri, what its tensors are)."""
        path, described = small_model(self.directory, name)
        self.store.put_file(self.prefix + name, path)
        return self.store.uri(self.prefix + name), described, path

    def test_the_service_keeps_what_is_put_in_it(self):
        data = tensor_bytes("round trip", 1 << 20)
        self.store.put_bytes(self.prefix + "object.bin", data)
        self.assertEqual(self.store.get_bytes(self.prefix + "object.bin"), data)
        self.assertEqual(self.store.keys(self.prefix), [self.prefix + "object.bin"])

    def test_stream_a_safetensors_file_from_a_bucket_to_the_gpu(self):
        uri, described, _ = self.put_small_model("model.safetensors")
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(uri, device=self.device)
            arrived = {name: tensor for name, tensor in streamer.get_tensors()}
        self.assertEqual(sorted(arrived), sorted(described))
        for name, (dtype, shape, crc) in described.items():
            tensor = arrived[name]
            self.assertEqual(tensor.dtype, safetensors_to_torch_dtype[dtype], name)
            self.assertEqual(list(tensor.shape), shape, name)
            self.assertEqual(tensor.device.type, torch.device(self.device).type, name)
            self.assertEqual(tensor_crc32(tensor), crc, "%s: the bytes that arrived are not the bytes that were put" % name)

    def test_list_safetensors_finds_the_safetensors_files_under_a_prefix_and_nothing_else(self):
        self.put_small_model("a.safetensors")
        self.put_small_model("b.safetensors")
        self.store.put_bytes(self.prefix + "notes.txt", b"not a model")
        self.store.put_bytes(self.prefix + "config.json", b"{}")
        found = list_safetensors(self.store.uri(self.prefix))
        self.assertEqual(sorted(os.path.basename(path) for path in found), ["a.safetensors", "b.safetensors"])

    def test_pull_files_downloads_the_files_of_a_prefix(self):
        _, _, original = self.put_small_model("model.safetensors")
        self.store.put_bytes(self.prefix + "config.json", b'{"model_type": "test"}')
        destination = os.path.join(self.directory, "pulled")
        pull_files(self.store.uri(self.prefix), destination)
        pulled = {name: os.path.join(base, name) for base, _, names in os.walk(destination) for name in names}
        self.assertEqual(sorted(pulled), ["config.json", "model.safetensors"])
        with open(original, "rb") as handle, open(pulled["model.safetensors"], "rb") as got:
            self.assertEqual(got.read(), handle.read())

    def test_streaming_an_object_that_is_not_there_is_an_error(self):
        """Not an empty model, and not a hang: the error is raised, whichever call it comes from. A ValueError of the streamer
        that says the file could not be accessed: a missing plugin or a wrong endpoint would not say that."""
        with self.assertRaisesRegex(ValueError, "File access error"):
            with SafetensorsStreamer() as streamer:
                streamer.stream_file(self.store.uri(self.prefix + "no-such-model.safetensors"), device=self.device)
                list(streamer.get_tensors())

    def test_a_file_that_is_not_safetensors_is_an_error(self):
        self.store.put_bytes(self.prefix + "garbage.safetensors", b"this is not a safetensors file" * 10)
        with self.assertRaisesRegex(ValueError, "Corrupted File"):
            with SafetensorsStreamer() as streamer:
                streamer.stream_file(self.store.uri(self.prefix + "garbage.safetensors"), device=self.device)
                list(streamer.get_tensors())

    def test_a_file_of_a_single_tensor_streams_too(self):
        data = tensor_bytes("single", 4096 * 3 + 1)  # not a multiple of a page: the end of an object is not aligned
        path = os.path.join(self.directory, "single.safetensors")
        described = write_safetensors(path, [("only", "U8", (len(data),), data)])
        self.store.put_file(self.prefix + "single.safetensors", path)
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(self.store.uri(self.prefix + "single.safetensors"), device=self.device)
            arrived = dict(streamer.get_tensors())
        self.assertEqual(list(arrived), ["only"])
        self.assertEqual(tensor_crc32(arrived["only"]), described["only"][2])


if __name__ == "__main__":
    unittest.main()
