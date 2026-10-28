# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the fixture generator. They need no GPU and no streamer, only numpy.

    python3 -m unittest tests/gpu/test_gen_models.py
"""

import copy
import json
import os
import struct
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
            again = gen.load_manifest(other, "mixed")
        crcs = lambda manifest: [t["crc32"] for f in manifest["files"] for t in f["tensors"]]  # noqa: E731
        self.assertEqual(crcs(again), crcs(self.manifests["mixed"]))
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 12, no_verify=True), 0)
            different = gen.load_manifest(other, "mixed")
        self.assertNotEqual(crcs(different), crcs(self.manifests["mixed"]))

    def test_a_corrupted_byte_is_caught(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 11, no_verify=True), 0)
            entry = gen.load_manifest(other, "giant_tensor")["files"][0]
            path = os.path.join(other, "giant_tensor", entry["path"])
            with open(path, "r+b") as handle:
                handle.seek(entry["size"] - 1)
                last = handle.read(1)
                handle.seek(entry["size"] - 1)
                handle.write(bytes([last[0] ^ 0xFF]))
            with self.assertRaises(gen.VerifyError):
                gen.verify_model(other, "giant_tensor")

    def test_a_truncated_file_is_caught(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 11, no_verify=True), 0)
            entry = gen.load_manifest(other, "mixed")["files"][1]
            os.truncate(os.path.join(other, "mixed", entry["path"]), entry["size"] - 5)
            with self.assertRaises(gen.VerifyError):
                gen.verify_model(other, "mixed")

    def test_a_budget_too_small_for_the_mandatory_set_fails_loudly(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(generate(other, 11, scale=0.001), 1)

    def test_every_dtype_rank_and_edge_size_is_present(self):
        gen.check_coverage(list(self.manifests.values()))

    def test_every_known_dtype_is_in_the_default_models(self):
        self.assertEqual(self.manifests["mixed"]["dtypes"], gen.DTYPE_NAMES)
        self.assertEqual({t["dtype"] for _, t in self.tensors("mixed")}, set(gen.DTYPE_NAMES))

    def test_dtypes_option_narrows_the_mixed_model_and_the_coverage_check(self):
        narrow = ["F32", "U16", "C64"]
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(gen.main(["generate", "--out", other, "--seed", "11", "--scale", str(SCALE),
                                       "--files", str(FILES), "--dtypes", ",".join(narrow)]), 0)
            mixed = gen.load_manifest(other, "mixed")
        self.assertEqual(mixed["dtypes"], sorted(narrow))
        # the mixed model's tensors of the random draw and the mandatory ones, plus the U8 edge sizes
        self.assertEqual({t["dtype"] for f in mixed["files"] for t in f["tensors"]} - {"U8"}, set(narrow))

    def test_unknown_dtype_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="gen-models-") as other:
            self.assertEqual(gen.main(["generate", "--out", other, "--seed", "11", "--dtypes", "F32,F9"]), 1)

    def test_packed_dtype_shapes_count_values_and_bytes_count_pairs(self):
        f4 = [t for _, t in self.tensors("mixed") if t["dtype"] == "F4"]
        self.assertTrue(f4)
        for tensor in f4:
            values = 1
            for dim in tensor["shape"]:
                values *= dim
            self.assertTrue(tensor["shape"], "an F4 tensor needs a last dimension, so it has no scalar")
            self.assertEqual(tensor["shape"][-1] % 2, 0, tensor["name"])
            self.assertEqual(tensor["end"] - tensor["start"], values // 2, tensor["name"])
        special = next(t for t in f4 if t["name"] == "mandatory.f4.special")
        self.assertEqual((special["shape"], special["end"] - special["start"]), ([512], 256))

    def test_packed_shape_halves_only_the_last_dimension_of_a_packed_dtype(self):
        self.assertEqual(gen.packed_shape("F4", (2, 4)), [2, 2])
        self.assertEqual(gen.packed_shape("F4", (6,)), [3])
        self.assertEqual(gen.packed_shape("U8", (2, 4)), [2, 4])
        self.assertEqual(gen.packed_shape("F32", ()), [])

    def test_the_header_of_an_f4_tensor_records_the_shape_in_values(self):
        entry = next(f for f, t in self.tensors("mixed") if t["name"] == "mandatory.f4.special")
        path = os.path.join(self.out, "mixed", entry["path"])
        with open(path, "rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len).decode("utf-8"))
        self.assertEqual(header["mandatory.f4.special"]["shape"], [512])
        start, end = header["mandatory.f4.special"]["data_offsets"]
        self.assertEqual(end - start, 256)

    def test_empty_tensors_only_with_the_flag(self):
        empty = [t["name"] for m in gen.MODEL_NAMES for _, t in self.tensors(m) if t["end"] == t["start"]]
        if gen.EMPTY_TENSORS:
            self.assertTrue(empty, "EMPTY_TENSORS is set but no tensor is empty")
        else:
            self.assertEqual(empty, [], "EMPTY_TENSORS = False leaves empty tensors out")

    def test_model_shapes(self):
        mixed, many_tensors, giant_tensor, many_files = (self.manifests[m] for m in gen.MODEL_NAMES)
        self.assertEqual(mixed["num_files"], 4)
        self.assertEqual(many_tensors["num_files"], 1)
        self.assertGreaterEqual(many_tensors["num_tensors"], 5000)
        self.assertEqual(giant_tensor["num_files"], 2)
        self.assertEqual(many_files["num_files"], FILES)
        self.assertLessEqual(mixed["largest_tensor_bytes"], gen.MIXED_MAX_TENSOR)
        self.assertEqual(max(t["end"] - t["start"] for f, t in self.tensors("giant_tensor")),
                         giant_tensor["largest_tensor_bytes"])

    def test_alignment_classes_all_occur(self):
        starts = [(f["data_start"] + t["start"], t["dtype"]) for f, t in self.tensors("mixed")]
        size = lambda dtype: gen.DTYPES[dtype][0]  # noqa: E731
        self.assertTrue(any(s % 4096 == 0 for s, _ in starts), "no tensor starts on a 4096 boundary")
        self.assertTrue(any(s % 4096 != 0 for s, _ in starts), "every tensor starts on a 4096 boundary")
        self.assertTrue(any(s % size(d) != 0 for s, d in starts), "no tensor is misaligned for its dtype")

    def test_header_modes_put_the_data_where_they_say(self):
        """aligned: the data starts on a 4096 boundary of the file; pad8: on a multiple of 8; odd: 3 past one."""
        seen = set()
        for model in gen.MODEL_NAMES:
            for entry in self.manifests[model]["files"]:
                mode, start = entry["header_mode"], entry["data_start"]
                seen.add(mode)
                if mode == "aligned":
                    self.assertEqual(start % 4096, 0, (model, entry["path"]))
                elif mode == "pad8":
                    self.assertEqual(start % 8, 0, (model, entry["path"]))
                else:
                    self.assertEqual((mode, start % 8), ("odd", 3), (model, entry["path"]))
        self.assertEqual(seen, {"aligned", "pad8", "odd"})

    def test_the_regeneration_check_notices_a_tensor_that_does_not_come_from_the_seed(self):
        manifest = copy.deepcopy(self.manifests["mixed"])
        self.assertEqual(gen.check_determinism(manifest), [])
        sampled = [t for f in manifest["files"] for t in f["tensors"] if 0 < t["end"] - t["start"] <= 8 * gen.MIB]
        sampled[0]["crc32"] ^= 1  # the first one that it regenerates
        problems = gen.check_determinism(manifest)
        self.assertEqual(len(problems), 1)
        self.assertIn(sampled[0]["name"], problems[0])

    def test_the_regeneration_check_notices_another_seed(self):
        manifest = copy.deepcopy(self.manifests["mixed"])
        manifest["seed"] += 1
        self.assertEqual(len(gen.check_determinism(manifest)), 3)

    def test_the_many_files_model_has_the_default_number_of_files(self):
        """Not the 40 of these tests: what a run generates, below the soft limit of 1024 open files."""
        self.assertEqual(gen.DEFAULT_FILES, 500)
        self.assertLess(gen.DEFAULT_FILES, 1024)

    def test_many_files_has_header_only_and_single_byte_files(self):
        files = self.manifests["many_files"]["files"]
        self.assertTrue(any(not f["tensors"] for f in files), "no header-only file")
        self.assertTrue(any(len(f["tensors"]) == 1 and f["tensors"][0]["end"] - f["tensors"][0]["start"] == 1
                            for f in files), "no file with a single 1-byte tensor")

    def test_bool_tensors_hold_only_zero_and_one(self):
        for f, t in self.tensors("mixed"):
            if t["dtype"] != "BOOL" or t["end"] == t["start"]:
                continue
            with open(os.path.join(self.out, "mixed", f["path"]), "rb") as handle:
                handle.seek(f["data_start"] + t["start"])
                self.assertLessEqual(set(handle.read(t["end"] - t["start"])), {0, 1}, t["name"])

    def test_tensor_names_are_unique_within_a_model(self):
        for model in gen.MODEL_NAMES:
            names = [t["name"] for _, t in self.tensors(model)]
            self.assertEqual(len(names), len(set(names)), model)

    def test_unknown_model_name_is_refused(self):
        self.assertEqual(gen.main(["verify", "--out", self.out, "--models", "mixed,nonsense"]), 1)
        self.assertEqual(gen.parse_models("giant_tensor, many_files"), ["giant_tensor", "many_files"])

    def test_manifest_is_json_round_trippable(self):
        for manifest in self.manifests.values():
            self.assertEqual(json.loads(json.dumps(manifest)), manifest)


if __name__ == "__main__":
    unittest.main()
