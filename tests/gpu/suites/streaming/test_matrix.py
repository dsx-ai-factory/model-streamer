# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the rows and memory limits of the GPU test.

    python3 -m unittest tests/gpu/suites/streaming/test_matrix.py
"""

import copy
import functools
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_models  # noqa: E402
import matrix  # noqa: E402

MIB = 1 << 20
SEEDS = (1, 2, 3)  # of the full size models that the limits are computed for


@functools.lru_cache(maxsize=None)
def _facts(model, seed):
    gen_models.use_dtypes(gen_models.DTYPE_NAMES)  # other tests narrow them
    return gen_models.plan_facts(gen_models.make_plan(model, seed, 1.0, gen_models.DEFAULT_FILES))


def manifest(model, seed=1):
    """What memory_limit is computed from, of the full size model of a seed: from the generator's plan, so that nothing
    is written and a change of the generator is seen here (about 2 seconds for many_tensors)."""
    return copy.deepcopy(_facts(model, seed))


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
            if row.limit == "roomy":
                self.assertIn(row.model, matrix.ROOMY_MODELS, row.id)

    def test_every_option_is_in_the_full_set(self):
        rows = matrix.full_rows()
        for model in matrix.MODEL_NAMES:
            for field, values in (("policy", (None, "spans", "chunks")), ("owned", (True, False)),
                                  ("device", ("cuda", "cpu_to")), ("limit", ("tight", "default"))):
                for value in values:
                    self.assertTrue(any(r.model == model and getattr(r, field) == value for r in rows),
                                    (model, field, value))
        self.assertTrue(any(r.limit == "roomy" for r in rows))
        self.assertTrue(any(r.policy == "files" for r in rows))

    def test_counts(self):
        counts = {model: sum(1 for r in matrix.full_rows() if r.model == model) for model in matrix.MODEL_NAMES}
        self.assertEqual(counts, {"mixed": 30, "many_tensors": 24, "giant_tensor": 20, "many_files": 30})

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

    def test_tight_is_the_largest_tensor_for_each_rank(self):
        for model in matrix.MODEL_NAMES:
            largest = manifest(model)["largest_tensor_bytes"]
            self.assertEqual(matrix.memory_limit(manifest(model), False, "tight"), largest, model)
            self.assertEqual(matrix.memory_limit(manifest(model), True, "tight"), matrix.WORLD * largest, model)

    def test_roomy_is_below_the_model_and_a_few_times_the_tight_limit(self):
        for model in matrix.ROOMY_MODELS:
            m = manifest(model)
            for distributed in (False, True):
                roomy = matrix.memory_limit(m, distributed, "roomy")
                self.assertLess(roomy, m["total_data_bytes"], (model, distributed))
                self.assertGreaterEqual(roomy, 2 * matrix.memory_limit(m, distributed, "tight"), (model, distributed))

    def test_the_limits_can_be_set_for_the_full_size_models_of_several_seeds(self):
        """What the GPU run does for every row, with the models that it generates: a limit that cannot be set for a model
        (a header larger than the largest tensor, a roomy limit that the model fits in) would fail the row there, so
        a change of the generator that causes it is found here. The models are planned, not written."""
        for seed in SEEDS:
            for model in matrix.MODEL_NAMES:
                m = manifest(model, seed)
                for distributed in (False, True):
                    matrix.memory_limit(m, distributed, "tight")
                    if model in matrix.ROOMY_MODELS:
                        roomy = matrix.memory_limit(m, distributed, "roomy")
                        self.assertLess(roomy, m["total_data_bytes"], (model, seed, distributed))

    def test_the_facts_of_a_plan_are_those_of_the_manifest_that_is_written(self):
        """plan_facts of a plan is what generate writes: the total, the largest tensor and the header of every file."""
        import tempfile
        with tempfile.TemporaryDirectory(prefix="plan-facts-") as out:
            gen_models.use_dtypes(gen_models.DTYPE_NAMES)
            for model in ("mixed", "many_files"):
                plan = gen_models.make_plan(model, 5, 0.05, 20)
                shard_facts = gen_models.plan_facts(plan)
                written = [gen_models.write_shard(os.path.join(out, "%s-%d" % (model, i)), shard, 5, model)
                           for i, shard in enumerate(plan)]
                self.assertEqual(shard_facts["total_data_bytes"],
                                 sum(e["end"] - e["start"] for f in written for e in f["tensors"]), model)
                self.assertEqual([f["header_len"] for f in shard_facts["files"]], [f["header_len"] for f in written],
                                 model)

    def test_roomy_is_refused_where_the_model_fits_in_it(self):
        """giant_tensor: twice a 2 GiB tensor, for each of two ranks, is more than the 4.5 GB of the model."""
        with self.assertRaises(ValueError):
            matrix.memory_limit(manifest("giant_tensor"), True, "roomy")
        self.assertNotIn("giant_tensor", matrix.ROOMY_MODELS)

    def test_a_header_larger_than_the_largest_tensor_is_refused(self):
        """The streamer reads a header as one range, so the tight limit would have to be sized by it."""
        m = manifest("many_tensors")
        m["files"] = [{"header_len": m["largest_tensor_bytes"] + 1}]
        for level in ("tight", "roomy"):
            with self.assertRaises(ValueError):
                matrix.memory_limit(m, False, level)

    def test_every_model_has_a_largest_tensor_that_is_not_below_its_header(self):
        for seed in SEEDS:
            for model in matrix.MODEL_NAMES:
                m = manifest(model, seed)
                self.assertGreaterEqual(m["largest_tensor_bytes"], max(f["header_len"] for f in m["files"]),
                                        (model, seed))

    def test_unknown_level_is_refused(self):
        with self.assertRaises(ValueError):
            matrix.memory_limit(manifest("mixed"), False, "huge")


class OneGpu(unittest.TestCase):
    def test_the_world_is_two_unless_it_is_one(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("GPU_WORLD", None)
            self.assertEqual(matrix.gpu_world(), matrix.WORLD)
            os.environ["GPU_WORLD"] = "1"
            self.assertEqual(matrix.gpu_world(), 1)
        self.assertEqual(matrix.gpu_world(""), matrix.WORLD)
        self.assertEqual(matrix.gpu_world("2"), 2)
        self.assertEqual(matrix.gpu_world("1"), 1)
        for text in ("0", "3", "two", "1.0"):
            with self.assertRaises(ValueError):
                matrix.gpu_world(text)

    def test_two_ranks_run_every_row(self):
        for mode in ("core", "full", "smoke"):
            self.assertEqual(matrix.rows_for(mode, 2), matrix.rows_for(mode), mode)
            self.assertEqual(matrix.skipped_rows(mode, 2), [], mode)

    def test_one_rank_runs_the_rows_that_are_not_distributed_and_leaves_out_the_others(self):
        for mode in ("core", "full", "smoke"):
            run, skipped = matrix.rows_for(mode, 1), matrix.skipped_rows(mode, 1)
            self.assertTrue(run, mode)
            self.assertTrue(all(not row.dist for row in run), mode)
            self.assertTrue(all(row.dist for row in skipped), mode)
            self.assertEqual(len(run) + len(skipped), len(matrix.rows_for(mode)), mode)

    def test_the_counts_of_one_gpu(self):
        self.assertEqual((len(matrix.rows_for("full", 1)), len(matrix.skipped_rows("full", 1))), (44, 60))
        self.assertEqual((len(matrix.rows_for("core", 1)), len(matrix.skipped_rows("core", 1))), (8, 6))
        self.assertEqual((len(matrix.rows_for("smoke", 1)), len(matrix.skipped_rows("smoke", 1))), (2, 1))

    def test_every_model_and_every_limit_still_has_a_row_on_one_gpu(self):
        rows = matrix.rows_for("full", 1)
        for model in matrix.MODEL_NAMES:
            self.assertTrue(any(r.model == model for r in rows), model)
        for limit in matrix.LIMITS:
            self.assertTrue(any(r.limit == limit for r in rows), limit)


class FilterRows(unittest.TestCase):
    def test_a_prefix_selects_the_rows_whose_id_starts_with_it(self):
        picked = matrix.filter_rows(matrix.full_rows(), "mixed.solo")
        self.assertTrue(picked)
        self.assertTrue(all(row.id.startswith("mixed.solo") for row in picked))
        self.assertEqual(len(picked), sum(1 for r in matrix.full_rows() if r.id.startswith("mixed.solo")))

    def test_several_prefixes_and_the_spaces_around_them(self):
        both = matrix.filter_rows(matrix.full_rows(), "giant_tensor.dist.chunks, many_files.solo")
        self.assertEqual({r.model for r in both}, {"giant_tensor", "many_files"})
        self.assertEqual(both, matrix.filter_rows(matrix.full_rows(), "giant_tensor.dist.chunks,many_files.solo"))

    def test_an_empty_entry_does_not_select_every_row(self):
        self.assertEqual(matrix.filter_rows(matrix.full_rows(), "mixed.solo,"), matrix.filter_rows(
            matrix.full_rows(), "mixed.solo"))
        self.assertEqual(matrix.row_prefixes("a,, b ,"), ("a", "b"))

    def test_no_prefix_selects_all_the_rows(self):
        everything = matrix.full_rows()
        for text in (None, "", " ", " , "):
            self.assertEqual(matrix.filter_rows(everything, text), everything, repr(text))

    def test_a_prefix_that_no_row_has_selects_none(self):
        self.assertEqual(matrix.filter_rows(matrix.core_rows(), "giant_tensor.dist.chunks"), [])


class UnmatchedPrefixes(unittest.TestCase):
    def test_a_prefix_that_no_row_has_is_named_even_when_another_prefix_selects_rows(self):
        rows = matrix.core_rows()
        text = "mixed.solo, mixd.dist,giant_tensor.dist.chunks"
        self.assertTrue(matrix.filter_rows(rows, text))
        self.assertEqual(matrix.unmatched_prefixes(rows, text), ["mixd.dist", "giant_tensor.dist.chunks"])

    def test_nothing_is_unmatched_when_every_prefix_selects_rows_or_there_is_none(self):
        rows = matrix.full_rows()
        self.assertEqual(matrix.unmatched_prefixes(rows, "mixed.solo,many_files"), [])
        for text in (None, "", " , "):
            self.assertEqual(matrix.unmatched_prefixes(rows, text), [], repr(text))


class NoRowsOutcome(unittest.TestCase):
    def test_an_empty_selection_without_a_filter_is_a_failure(self):
        for allow in (True, False):
            self.assertEqual(matrix.no_rows_outcome(None, allow, "core")[0], 1)
            self.assertEqual(matrix.no_rows_outcome(" , ", allow, "core")[0], 1)

    def test_a_filter_that_matches_nothing_passes_only_where_it_is_allowed(self):
        status, message = matrix.no_rows_outcome("giant_tensor.dist.chunks", True, "core")
        self.assertEqual(status, 0)
        self.assertIn("nothing to run on this leg", message)
        status, message = matrix.no_rows_outcome("mixd.solo", False, "full")
        self.assertEqual(status, 1)
        self.assertIn("typo", message)
        self.assertIn("mixd.solo", message)


if __name__ == "__main__":
    unittest.main()
