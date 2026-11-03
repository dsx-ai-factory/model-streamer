# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of clean_results.sh, which removes the results of an earlier run before a new run starts.

    python3 -m unittest tests/gpu/harness/test_clean_results.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "clean_results.sh")

sys.path.insert(0, HERE)
import summarize  # noqa: E402
from gpu_harness.results import write_results  # noqa: E402


def touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


class CleanResults(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="clean-results-")
        self.addCleanup(self.dir.cleanup)
        self.results = os.path.join(self.dir.name, "gpu")

    def clean(self, *suites):
        return subprocess.run(["bash", SCRIPT, self.results, *suites], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True)

    def files(self):
        found = []
        for base, _, names in os.walk(self.results):
            found += [os.path.relpath(os.path.join(base, name), self.results) for name in names]
        return sorted(found)

    def test_the_results_of_an_earlier_run_are_removed_and_the_log_is_not(self):
        for name in ("streaming/results-sync_buffered-rank0.json", "streaming/results-sync_buffered-rank1.json",
                     "streaming/timing-sync_buffered-rank0.jsonl", "timing-summary.txt", "env-check.txt",
                     "timing-setup.jsonl", "streaming/self-check/results-sync_buffered-rank0.json",
                     "streaming/smoke/results-default-rank1.json"):
            touch(os.path.join(self.results, name))
        touch(os.path.join(self.results, "run.log"), "the log that the workflow is writing")
        self.assertEqual(self.clean().returncode, 0)
        self.assertEqual(self.files(), ["run.log"])
        with open(os.path.join(self.results, "run.log")) as handle:
            self.assertEqual(handle.read(), "the log that the workflow is writing")

    def test_the_directory_of_a_suite_that_is_named_goes_with_what_it_kept_there(self):
        touch(os.path.join(self.results, "streaming", "manifests", "mixed.manifest.json.gz"))
        touch(os.path.join(self.results, "other", "kept.txt"))
        touch(os.path.join(self.results, "run.log"))
        self.assertEqual(self.clean("streaming").returncode, 0)
        self.assertEqual(self.files(), ["other/kept.txt", "run.log"])

    def test_what_is_not_a_result_is_left_alone(self):
        touch(os.path.join(self.results, "notes.txt"))
        self.assertEqual(self.clean().returncode, 0)
        self.assertEqual(self.files(), ["notes.txt"])

    def test_a_directory_that_is_not_there_is_not_an_error(self):
        self.assertEqual(self.clean().returncode, 0)
        self.assertFalse(os.path.exists(self.results))

    def test_a_wrong_usage_is_refused(self):
        done = subprocess.run(["bash", SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        self.assertEqual(done.returncode, 2)

    def test_a_run_of_one_gpu_after_a_run_of_two_is_not_counted_as_two_ranks(self):
        """The failure that this prevents: the second rank's file of the old run next to the new run's first rank."""
        directory = os.path.join(self.results, "streaming")

        def write(rank, world):
            write_results(directory, "streaming", "sync_buffered", rank, [{"case": "a", "ok": True}], 1, world=world,
                          skipped_cases=0 if world == 2 else 60,
                          skipped_note="" if world == 2 else "ONE GPU: 60 distributed rows were not run")
        write(0, 2)
        write(1, 2)
        self.clean("streaming")
        write(0, 1)  # the new run: one rank
        lines, incomplete = summarize.summarize(self.results, ["streaming/results-sync_buffered-rank*.json"])
        self.assertFalse(incomplete, lines)
        self.assertEqual([line for line in lines if "rank1" in line], [])
        self.assertIn("ONE GPU: 60 distributed rows were not run", lines[-1])


if __name__ == "__main__":
    unittest.main()
