# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the rows and memory limits of the GPU test.

    python3 -m unittest tests/gpu/test_matrix.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matrix  # noqa: E402

MIB = 1 << 20

# The full size models (generated with --scale 1): what the memory limit is computed from.
FULL_SIZE = {
    "mixed": {"total_data_bytes": 1073741824, "largest_tensor_bytes": 24595200, "header_len": 120_000},
    "many_tensors": {"total_data_bytes": 105_000_000, "largest_tensor_bytes": 4 * MIB, "header_len": 10_900_000},
    "giant_tensor": {"total_data_bytes": 4_250_000_000, "largest_tensor_bytes": 2 ** 31, "header_len": 1_000},
    "many_files": {"total_data_bytes": 10_770_000, "largest_tensor_bytes": 2_950_000, "header_len": 4_088},
}


def manifest(model):
    facts = FULL_SIZE[model]
    return {"total_data_bytes": facts["total_data_bytes"], "largest_tensor_bytes": facts["largest_tensor_bytes"],
            "files": [{"header_len": facts["header_len"]}]}


def ring(limit, manifest_):
    """(buffer size, buffers) the streamer derives from a limit, as _ring_sizing does for a binding limit."""
    largest = manifest_["largest_tensor_bytes"]
    budget = min(limit, manifest_["total_data_bytes"])
    buffer_size = max(largest, budget // matrix.RING_BUFFERS)
    return buffer_size, min(matrix.RING_BUFFERS, budget // buffer_size)


class Rows(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [row.id for row in matrix.full_rows()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_row_is_valid(self):
        for row in matrix.full_rows():
            self.assertTrue(matrix.is_valid(row.model, row.dist, row.policy, row.owned, row.device, row.limit),
                            row.id)
            if row.dist:
                self.assertEqual(row.device, "cuda", row.id)
            if row.policy == "files":
                self.assertNotEqual(row.model, "many_tensors", row.id)
            if row.limit == "deep":
                self.assertIn(row.model, matrix.DEEP_MODELS, row.id)

    def test_every_option_is_in_the_full_set(self):
        rows = matrix.full_rows()
        for model in matrix.MODEL_NAMES:
            for field, values in (("policy", (None, "spans", "chunks")), ("owned", (True, False)),
                                  ("device", ("cuda", "cpu_to")), ("limit", ("tight", "default"))):
                for value in values:
                    self.assertTrue(any(r.model == model and getattr(r, field) == value for r in rows),
                                    (model, field, value))
        self.assertTrue(any(r.limit == "deep" for r in rows))
        self.assertTrue(any(r.policy == "files" for r in rows))

    def test_counts(self):
        counts = {model: sum(1 for r in matrix.full_rows() if r.model == model) for model in matrix.MODEL_NAMES}
        self.assertEqual(counts, {"mixed": 30, "many_tensors": 24, "giant_tensor": 20, "many_files": 20})

    def test_core_is_part_of_full(self):
        full = {row.id for row in matrix.full_rows()}
        core = matrix.core_rows()
        self.assertEqual(len(core), len(matrix.CORE_IDS))
        self.assertLessEqual({row.id for row in core}, full)
        for model in matrix.MODEL_NAMES:
            self.assertTrue(any(r.model == model and not r.dist for r in core), model)
            self.assertTrue(any(r.model == model and r.dist for r in core), model)

    def test_smoke_rows_load_the_real_model_with_the_default_limit(self):
        rows = matrix.smoke_rows()
        self.assertEqual({row.model for row in rows}, {matrix.REAL_MODEL})
        self.assertEqual({row.limit for row in rows}, {"default"})
        self.assertEqual(len({row.id for row in rows}), len(rows))
        self.assertTrue(any(r.dist for r in rows) and any(not r.dist for r in rows))
        self.assertTrue(any(r.device == "cpu_to" for r in rows))
        for row in rows:
            self.assertTrue(matrix.is_valid(row.model, row.dist, row.policy, row.owned, row.device, row.limit),
                            row.id)
        # the real model is not one of the synthetic models, so it is never in the full or core sets
        self.assertFalse({row.id for row in rows} & {row.id for row in matrix.full_rows()})

    def test_rows_for_refuses_an_unknown_mode(self):
        self.assertEqual(matrix.rows_for("core"), matrix.core_rows())
        self.assertEqual(matrix.rows_for("full"), matrix.full_rows())
        self.assertEqual(matrix.rows_for("smoke"), matrix.smoke_rows())
        with self.assertRaises(ValueError):
            matrix.rows_for("everything")


class MemoryLimit(unittest.TestCase):
    def test_default_leaves_the_variable_unset(self):
        self.assertIsNone(matrix.memory_limit(manifest("mixed"), False, "default"))

    def test_tight_covers_two_of_the_largest_range_per_rank(self):
        for model in matrix.MODEL_NAMES:
            m = manifest(model)
            largest = max(m["largest_tensor_bytes"], m["files"][0]["header_len"])
            self.assertGreaterEqual(matrix.memory_limit(m, False, "tight"), 2 * largest, model)
            self.assertGreaterEqual(matrix.memory_limit(m, True, "tight"), 4 * largest, model)

    def test_tight_recycles_with_a_small_ring(self):
        # many_tensors has five buffers, not two: its 10 MB header sets the floor of the limit
        for model, expected in (("mixed", 2), ("many_tensors", 5)):
            m = manifest(model)
            buffer_size, buffers = ring(matrix.memory_limit(m, False, "tight"), m)
            self.assertEqual(buffers, expected, model)
            self.assertLess(buffer_size * buffers, m["total_data_bytes"], model)

    def test_deep_has_eighteen_buffers_and_still_recycles(self):
        for model in matrix.DEEP_MODELS:
            m = manifest(model)
            for distributed in (False, True):
                ranks = 2 if distributed else 1
                limit = matrix.memory_limit(m, distributed, "deep")
                buffer_size, buffers = ring(limit // ranks, m)
                self.assertEqual(buffers, matrix.RING_BUFFERS, (model, distributed))
                self.assertLess(buffer_size * buffers, m["total_data_bytes"], (model, distributed))

    def test_deep_is_refused_where_nothing_would_be_reused(self):
        for model in set(matrix.MODEL_NAMES) - set(matrix.DEEP_MODELS):
            with self.assertRaises(ValueError):
                matrix.memory_limit(manifest(model), False, "deep")

    def test_unknown_level_is_refused(self):
        with self.assertRaises(ValueError):
            matrix.memory_limit(manifest("mixed"), False, "huge")


if __name__ == "__main__":
    unittest.main()
