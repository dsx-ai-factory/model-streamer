# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the real model download and its manifest. No network: the source is a local file.

    python3 -m unittest tests/gpu/test_real_model.py
"""

import hashlib
import json
import os
import pathlib
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_model  # noqa: E402


def write_safetensors(path, tensors, metadata=None):
    """tensors: list of (name, dtype, shape, bytes), written in that order."""
    header, offset = {}, 0
    if metadata:
        header["__metadata__"] = metadata
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded)
        for _, _, _, data in tensors:
            handle.write(data)


TENSORS = [
    ("a.weight", "BF16", [2, 3], bytes(range(12))),
    ("b.bias", "F32", [2], bytes(range(100, 108))),
    ("c.empty", "U8", [0], b""),
    ("d.scalar", "U8", [], b"\x07"),
]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="real-model-")
        self.addCleanup(self.dir.cleanup)
        self.source = os.path.join(self.dir.name, "source.safetensors")
        write_safetensors(self.source, TENSORS, {"format": "pt"})
        data = pathlib.Path(self.source).read_bytes()
        self.model = {"repo": "x/y", "revision": "r", "filename": "model.safetensors", "size": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}
        self.url = pathlib.Path(self.source).as_uri()
        self.out = os.path.join(self.dir.name, "out")


class Manifest(Fixture):
    def test_manifest_describes_every_tensor_with_the_crc_of_its_bytes(self):
        real_model.fetch(self.out, self.url, self.model)
        with open(os.path.join(self.out, real_model.MODEL_NAME + ".manifest.json")) as handle:
            manifest = json.load(handle)
        (entry,) = manifest["files"]
        self.assertEqual(entry["path"], "model.safetensors")
        self.assertEqual(manifest["num_tensors"], 4)
        self.assertEqual(manifest["total_data_bytes"], 12 + 8 + 0 + 1)
        self.assertEqual(manifest["largest_tensor_bytes"], 12)
        self.assertEqual(manifest["dtypes"], ["BF16", "F32", "U8"])
        by_name = {t["name"]: t for t in entry["tensors"]}
        for name, dtype, shape, data in TENSORS:
            tensor = by_name[name]
            self.assertEqual((tensor["dtype"], tensor["shape"], tensor["end"] - tensor["start"]),
                             (dtype, shape, len(data)), name)
            self.assertEqual(tensor["crc32"], zlib.crc32(data), name)
        self.assertEqual(entry["data_start"], 8 + entry["header_len"])
        self.assertEqual(entry["metadata"], {"format": "pt"})

    def test_tensors_are_listed_in_file_order(self):
        manifest = real_model.build_manifest(self.dir.name, "source.safetensors")
        starts = [t["start"] for t in manifest["files"][0]["tensors"]]
        self.assertEqual(starts, sorted(starts))

    def test_a_hole_between_tensors_is_refused(self):
        path = os.path.join(self.dir.name, "hole.safetensors")
        header = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                  "b": {"dtype": "U8", "shape": [2], "data_offsets": [3, 5]}}
        encoded = json.dumps(header).encode()
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)) + encoded + bytes(5))
        with self.assertRaises(real_model.RealModelError):
            real_model.build_manifest(self.dir.name, "hole.safetensors")

    def test_bytes_after_the_last_tensor_are_refused(self):
        with open(self.source, "ab") as handle:
            handle.write(b"x")
        with self.assertRaises(real_model.RealModelError):
            real_model.build_manifest(self.dir.name, "source.safetensors")


class Download(Fixture):
    def test_download_is_timed_and_verified(self):
        real_model.fetch(self.out, self.url, self.model)
        with open(os.path.join(self.out, "timing-real.jsonl")) as handle:
            phases = [json.loads(line)["phase"] for line in handle]
        self.assertEqual(phases, ["real_model.download", "real_model.manifest"])
        self.assertEqual(pathlib.Path(self.out, real_model.MODEL_NAME, "model.safetensors").read_bytes(),
                         pathlib.Path(self.source).read_bytes())
        self.assertFalse(os.path.exists(os.path.join(self.out, real_model.MODEL_NAME, "model.safetensors.part")))

    def test_a_changed_file_is_refused_and_leaves_nothing_behind(self):
        self.model["sha256"] = "0" * 64
        with self.assertRaises(real_model.RealModelError):
            real_model.download(self.url, os.path.join(self.dir.name, "x.safetensors"), self.model["size"],
                                self.model["sha256"], attempts=1)
        self.assertEqual([n for n in os.listdir(self.dir.name) if n.startswith("x.")], [])

    def test_a_wrong_size_is_refused(self):
        with self.assertRaises(real_model.RealModelError):
            real_model.download(self.url, os.path.join(self.dir.name, "x.safetensors"), self.model["size"] + 1,
                                self.model["sha256"], attempts=1)

    def test_a_missing_source_is_refused(self):
        with self.assertRaises(real_model.RealModelError):
            real_model.download(pathlib.Path(self.dir.name, "nothing").as_uri(),
                                os.path.join(self.dir.name, "x.safetensors"), 1, "0" * 64, attempts=1)

    def test_the_command_line_reports_a_failure_with_a_nonzero_status(self):
        self.assertEqual(real_model.main(["fetch", "--out", self.out, "--url",
                                          pathlib.Path(self.dir.name, "nothing").as_uri()]), 1)


class Pin(unittest.TestCase):
    def test_the_pinned_model_names_a_revision_a_size_and_a_hash(self):
        self.assertRegex(real_model.MODEL["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(real_model.MODEL["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(real_model.MODEL["size"], 0)
        self.assertIn(real_model.MODEL["revision"], real_model.default_url())


if __name__ == "__main__":
    unittest.main()
