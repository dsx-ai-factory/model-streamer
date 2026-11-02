# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the helpers that suites use to put objects and files somewhere: ObjectStore (with a client that is made
up here, so no service and no boto3) and the safetensors writer, whose files are also read back with the safetensors
library where it is installed.

    python3 -m unittest tests/gpu/harness/test_object_store_and_files.py
"""

import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_harness import object_store, safetensors_file  # noqa: E402


class FakeClient:
    """The few calls of boto3's S3 client that ObjectStore makes."""

    def __init__(self):
        self.objects = {}
        self.head_failures = 0

    def head_bucket(self, Bucket):
        if self.head_failures:
            self.head_failures -= 1
            raise ConnectionError("not up yet")

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_file(self, path, Bucket, Key):
        with open(path, "rb") as handle:
            self.objects[Key] = handle.read()

    def delete_object(self, Bucket, Key):
        del self.objects[Key]

    def get_paginator(self, name):
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in client.objects if k.startswith(Prefix))
                # two pages, to see that a listing follows them
                for chunk in (keys[:1], keys[1:]):
                    yield {"Contents": [{"Key": k} for k in chunk]} if chunk else {}
        return Paginator()


class ObjectStoreTests(unittest.TestCase):
    def store(self):
        store = object_store.ObjectStore("http://seaweedfs:9000", "bucket")
        store._client = FakeClient()
        return store

    def test_the_environment_says_how_to_get_object_storage_when_it_has_none(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("AWS_ENDPOINT_URL", None)
            with self.assertRaises(object_store.ObjectStoreNotAvailable) as raised:
                object_store.ObjectStore.from_environment()
        self.assertIn("object-storage profile", str(raised.exception))

    def test_the_settings_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {"AWS_ENDPOINT_URL": "http://x:9000", "GPU_OBJECT_BUCKET": "b",
                                          "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"}):
            store = object_store.ObjectStore.from_environment()
        self.assertEqual((store.endpoint, store.bucket), ("http://x:9000", "b"))
        self.assertEqual(store._settings["aws_access_key_id"], "k")
        self.assertEqual(store.uri("a/b"), "s3://b/a/b")

    def test_the_bucket_defaults_to_the_one_that_the_service_creates(self):
        with mock.patch.dict(os.environ, {"AWS_ENDPOINT_URL": "http://x:9000"}):
            os.environ.pop("GPU_OBJECT_BUCKET", None)
            self.assertEqual(object_store.ObjectStore.from_environment().bucket, "test-bucket")

    def test_objects_round_trip_and_are_listed_under_a_prefix(self):
        store = self.store()
        store.put_bytes("p/a", b"1")
        store.put_bytes("p/b", b"2")
        store.put_bytes("q/c", b"3")
        self.assertEqual(store.get_bytes("p/a"), b"1")
        self.assertEqual(store.keys("p/"), ["p/a", "p/b"])  # across two pages
        store.delete_prefix("p/")
        self.assertEqual(store.keys(), ["q/c"])

    def test_a_prefix_is_a_test_s_own(self):
        store = self.store()
        first, second = store.new_prefix("a test"), store.new_prefix("a test")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("gpu-tests/a test-") and first.endswith("/"))

    def test_waiting_until_ready_waits_for_the_bucket_and_gives_up(self):
        store = self.store()
        store._client.head_failures = 2
        with mock.patch.object(object_store.time, "sleep"):
            store.wait_until_ready(seconds=30)
        store._client.head_failures = 10 ** 6
        with mock.patch.object(object_store.time, "sleep"), mock.patch.object(
                object_store.time, "monotonic", side_effect=[0, 1, 2, 100]):
            with self.assertRaises(object_store.ObjectStoreNotAvailable) as raised:
                store.wait_until_ready(seconds=5)
        self.assertIn("not up yet", str(raised.exception))


class SafetensorsFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="safetensors-file-")
        self.addCleanup(self.dir.cleanup)

    def read_header(self, path):
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            return length, json.loads(handle.read(length).decode()), handle.read()

    def test_the_bytes_of_a_tensor_are_a_function_of_its_name(self):
        self.assertEqual(safetensors_file.tensor_bytes("a", 100), safetensors_file.tensor_bytes("a", 100))
        self.assertNotEqual(safetensors_file.tensor_bytes("a", 100), safetensors_file.tensor_bytes("b", 100))
        self.assertEqual(len(safetensors_file.tensor_bytes("a", 12345)), 12345)
        self.assertEqual(safetensors_file.tensor_bytes("a", 0), b"")
        self.assertGreater(len(set(safetensors_file.tensor_bytes("a", 1000))), 10, "not all the same byte")

    def test_a_file_has_the_layout_of_safetensors(self):
        path, described = safetensors_file.small_model(self.dir.name)
        length, header, data = self.read_header(path)
        self.assertEqual(length % 8, 0, "the data starts on a multiple of 8")
        self.assertEqual(sorted(header), sorted(described))
        position = 0
        for name, (dtype, shape, crc) in described.items():
            entry = header[name]
            self.assertEqual((entry["dtype"], entry["shape"]), (dtype, shape))
            start, end = entry["data_offsets"]
            self.assertEqual(start, position, "no hole between tensors")
            self.assertEqual(zlib.crc32(data[start:end]), crc, name)
            position = end
        self.assertEqual(position, len(data), "the file ends with the last tensor")

    def test_the_sizes_are_what_the_shapes_and_dtypes_say(self):
        path, _ = safetensors_file.small_model(self.dir.name)
        _, header, _ = self.read_header(path)
        itemsize = {"BF16": 2, "F32": 4, "F16": 2}
        for entry in header.values():
            elements = 1
            for dimension in entry["shape"]:
                elements *= dimension
            start, end = entry["data_offsets"]
            self.assertEqual(end - start, elements * itemsize[entry["dtype"]])

    def test_the_safetensors_library_reads_the_files_where_it_is_installed(self):
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors is not installed")
        path, described = safetensors_file.small_model(self.dir.name)
        with safe_open(path, framework="numpy") as handle:
            self.assertEqual(sorted(handle.keys()), sorted(described))

    def test_a_single_tensor_of_any_size_is_written(self):
        path = os.path.join(self.dir.name, "one", "single.safetensors")
        data = safetensors_file.tensor_bytes("single", 4096 * 3 + 1)
        described = safetensors_file.write_safetensors(path, [("only", "U8", (len(data),), data)])
        _, header, body = self.read_header(path)
        self.assertEqual(body, data)
        self.assertEqual(described["only"], ("U8", [len(data)], zlib.crc32(data)))
        self.assertEqual(header["only"]["data_offsets"], [0, len(data)])


if __name__ == "__main__":
    unittest.main()
