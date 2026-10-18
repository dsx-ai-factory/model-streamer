# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the fixture generator. They need no GPU and no streamer, only numpy.

    python3 -m unittest tests/gpu/test_gen_models.py
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_models as gen  # noqa: E402

SCALE = 0.05
FILES = 40


def generate(out, seed, **overrides):
    argv = ["generate", "--out", out, "--seed", str(seed), "--scale", str(overrides.get("scale", SCALE)),
            "--files", str(FILES)]
    if overrides.get("no_verify"):
        argv.append("--no-verify")
    return gen.main(argv)


class GenerateModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory(prefix="gen-models-")
        cls.out = cls.dir.name
        assert generate(cls.out, 11) == 0
        cls.manifests = {m: gen.load_manifest(cls.out, m) for m in gen.MODEL_NAMES}

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def tensors(self, model):
        return [(f, t) for f in self.manifests[model]["files"] for t in f["tensors"]]

    def test_verify_passes_on_untouched_fixtures(self):
        self.assertEqual(gen.main(["verify", "--out", self.out]), 0)

    def test_same_seed_same_bytes_different_seed_different_bytes(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 11, no_verify=True), 0)
            again = gen.load_manifest(other, "A")
        crcs = lambda manifest: [t["crc32"] for f in manifest["files"] for t in f["tensors"]]  # noqa: E731
        self.assertEqual(crcs(again), crcs(self.manifests["A"]))
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 12, no_verify=True), 0)
            different = gen.load_manifest(other, "A")
        self.assertNotEqual(crcs(different), crcs(self.manifests["A"]))

    def test_a_corrupted_byte_is_caught(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 11, no_verify=True), 0)
            entry = gen.load_manifest(other, "C")["files"][0]
            path = os.path.join(other, "C", entry["path"])
            with open(path, "r+b") as handle:
                handle.seek(entry["size"] - 1)
                last = handle.read(1)
                handle.seek(entry["size"] - 1)
                handle.write(bytes([last[0] ^ 0xFF]))
            with self.assertRaises(gen.VerifyError):
                gen.verify_model(other, "C")

    def test_a_truncated_file_is_caught(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 11, no_verify=True), 0)
            entry = gen.load_manifest(other, "A")["files"][1]
            os.truncate(os.path.join(other, "A", entry["path"]), entry["size"] - 5)
            with self.assertRaises(gen.VerifyError):
                gen.verify_model(other, "A")

    def test_a_budget_too_small_for_the_mandatory_set_fails_loudly(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 11, scale=0.001), 1)

    def test_every_dtype_rank_and_edge_size_is_present(self):
        gen.check_coverage(list(self.manifests.values()))

    def test_empty_tensors_only_with_the_flag(self):
        empty = [t["name"] for m in gen.MODEL_NAMES for _, t in self.tensors(m) if t["end"] == t["start"]]
        if gen.EMPTY_TENSORS:
            self.assertTrue(empty, "EMPTY_TENSORS is set but no tensor is empty")
        else:
            self.assertEqual(empty, [], "empty tensors are left out until issue #260 is closed")

    def test_model_shapes(self):
        a, b, c, d = (self.manifests[m] for m in gen.MODEL_NAMES)
        self.assertEqual(a["num_files"], 4)
        self.assertEqual(b["num_files"], 1)
        self.assertGreaterEqual(b["num_tensors"], 5000)
        self.assertEqual(c["num_files"], 2)
        self.assertEqual(d["num_files"], FILES)
        self.assertLessEqual(a["largest_tensor_bytes"], gen.A_MAX_TENSOR)
        self.assertEqual(max(t["end"] - t["start"] for f, t in self.tensors("C")),
                         c["largest_tensor_bytes"])

    def test_alignment_classes_all_occur(self):
        starts = [(f["data_start"] + t["start"], t["dtype"]) for f, t in self.tensors("A")]
        size = lambda dtype: gen.DTYPES[dtype][0]  # noqa: E731
        self.assertTrue(any(s % 4096 == 0 for s, _ in starts), "no tensor starts on a 4096 boundary")
        self.assertTrue(any(s % 4096 != 0 for s, _ in starts), "every tensor starts on a 4096 boundary")
        self.assertTrue(any(s % size(d) != 0 for s, d in starts), "no tensor is misaligned for its dtype")

    def test_model_d_has_header_only_and_single_byte_files(self):
        files = self.manifests["D"]["files"]
        self.assertTrue(any(not f["tensors"] for f in files), "no header-only file")
        self.assertTrue(any(len(f["tensors"]) == 1 and f["tensors"][0]["end"] - f["tensors"][0]["start"] == 1
                            for f in files), "no file with a single 1-byte tensor")

    def test_bool_tensors_hold_only_zero_and_one(self):
        for f, t in self.tensors("A"):
            if t["dtype"] != "BOOL" or t["end"] == t["start"]:
                continue
            with open(os.path.join(self.out, "A", f["path"]), "rb") as handle:
                handle.seek(f["data_start"] + t["start"])
                self.assertLessEqual(set(handle.read(t["end"] - t["start"])), {0, 1}, t["name"])

    def test_tensor_names_are_unique_within_a_model(self):
        for model in gen.MODEL_NAMES:
            names = [t["name"] for _, t in self.tensors(model)]
            self.assertEqual(len(names), len(set(names)), model)

    def test_manifest_is_json_round_trippable(self):
        for manifest in self.manifests.values():
            self.assertEqual(json.loads(json.dumps(manifest)), manifest)


if __name__ == "__main__":
    unittest.main()
