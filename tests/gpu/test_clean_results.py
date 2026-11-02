# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of clean_results.sh, which removes the results of an earlier run before a new run starts.

    python3 -m unittest tests/gpu/test_clean_results.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "clean_results.sh")

sys.path.insert(0, HERE)
import rows_summary  # noqa: E402


def touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


class CleanResults(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="clean-results-")
        self.addCleanup(self.dir.cleanup)
        self.results = os.path.join(self.dir.name, "gpu")

    def clean(self):
        return subprocess.run(["bash", SCRIPT, self.results], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)

    def files(self):
        found = []
        for base, _, names in os.walk(self.results):
            found += [os.path.relpath(os.path.join(base, name), self.results) for name in names]
        return sorted(found)

    def test_the_results_of_an_earlier_run_are_removed_and_the_log_is_not(self):
        for name in ("rows-sync_buffered-rank0.json", "rows-sync_buffered-rank1.json", "timing-sync_buffered-rank0.jsonl",
                     "timing-summary.txt", "env-check.txt", "timing-setup.jsonl",
                     "self-check/rows-sync_buffered-rank0.json", "smoke/rows-default-rank1.json",
                     "manifests/mixed.manifest.json.gz"):
            touch(os.path.join(self.results, name))
        touch(os.path.join(self.results, "run.log"), "the log that the workflow is writing")
        self.assertEqual(self.clean().returncode, 0)
        self.assertEqual(self.files(), ["run.log"])
        with open(os.path.join(self.results, "run.log")) as handle:
            self.assertEqual(handle.read(), "the log that the workflow is writing")

    def test_a_directory_that_is_not_there_is_not_an_error(self):
        self.assertEqual(self.clean().returncode, 0)
        self.assertFalse(os.path.exists(self.results))

    def test_a_wrong_usage_is_refused(self):
        done = subprocess.run(["bash", SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        self.assertEqual(done.returncode, 2)

    def test_a_run_of_one_gpu_after_a_run_of_two_is_not_counted_as_two_ranks(self):
        """The failure that this prevents: the second rank's file of the old run next to the new run's first rank."""
        def write(name, world):
            touch(os.path.join(self.results, name), json.dumps(
                {"strategy": "x", "expected_rows": 1, "world": world, "skipped_rows": 0 if world == 2 else 60,
                 "rows": [{"row": "a", "strategy": "x", "ok": True}]}))
        write("rows-sync_buffered-rank0.json", 2)
        write("rows-sync_buffered-rank1.json", 2)
        self.clean()
        write("rows-sync_buffered-rank0.json", 1)  # the new run: one rank
        lines, incomplete = rows_summary.summarize(self.results, False, ["sync_buffered"])
        self.assertFalse(incomplete, lines)
        self.assertEqual([line for line in lines if "rank1" in line], [])
        self.assertIn("ONE GPU: 60 distributed rows were not run", lines[-1])


if __name__ == "__main__":
    unittest.main()
