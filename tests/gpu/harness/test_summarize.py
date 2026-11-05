# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the summary of the cases that ran, with made-up results files.

    python3 -m unittest tests/gpu/harness/test_summarize.py
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize  # noqa: E402
from gpu_harness.results import results_filename, write_results  # noqa: E402
from gpu_harness.world import DEFAULT_WORLD  # noqa: E402

RUNS = ["sync_buffered", "libaio_direct"]


def case(name, ok, **extra):
    entry = {"case": name, "ok": ok}
    entry.update(extra)
    return entry


class Summary(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="summarize-")
        self.addCleanup(self.dir.cleanup)

    def write(self, run, rank, cases, expected, subdirectory=None, world=2, skipped=0, note=""):
        directory = os.path.join(self.dir.name, subdirectory) if subdirectory else self.dir.name
        write_results(directory, "suite", run, rank, cases, expected, world=world, skipped_cases=skipped,
                      skipped_note=note)

    def write_all(self, cases, expected):
        for run in RUNS:
            for rank in (0, 1):
                self.write(run, rank, cases, expected)

    def summarize(self, *expected):
        return summarize.summarize(self.dir.name, expected or ["results-%s-rank*.json" % run for run in RUNS])

    def test_every_case_finished_is_complete(self):
        self.write_all([case("a", True), case("b", True)], 2)
        lines, incomplete = self.summarize()
        self.assertFalse(incomplete)
        self.assertEqual(len(lines), 4)
        self.assertIn("2 of 2 cases finished, 0 failed", lines[0])

    def test_failed_cases_are_counted_but_do_not_make_the_run_incomplete(self):
        self.write_all([case("a", True), case("b", False)], 2)
        lines, incomplete = self.summarize()
        self.assertFalse(incomplete)
        self.assertIn("2 of 2 cases finished, 1 failed; failed: b", lines[0])

    def test_the_failed_cases_are_named_up_to_a_limit(self):
        count = summarize.MAX_FAILED_LISTED + 3
        self.write_all([case("c%d" % i, False) for i in range(count)], count)
        lines, _ = self.summarize()
        listed = ", ".join("c%d" % i for i in range(summarize.MAX_FAILED_LISTED))
        self.assertIn("failed: %s, and 3 more" % listed, lines[0])

    def test_no_failed_cases_are_named_when_none_failed(self):
        self.write_all([case("a", True)], 1)
        lines, _ = self.summarize()
        self.assertNotIn("failed:", lines[0])

    def test_a_case_with_a_fault_injected_on_purpose_is_not_counted_as_a_failure(self):
        """The self-checks inject faults, and the case has errors by design: only the other case is a failure."""
        self.write_all([case("a", False, fault="corrupt"), case("b", False)], 2)
        lines, _ = self.summarize()
        self.assertIn("2 of 2 cases finished, 1 failed", lines[0])

    def test_a_case_left_started_names_the_case_where_the_run_ended(self):
        self.write_all([case("a", True), case("b", None)], 5)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertIn("1 of 5 cases finished", lines[0])
        self.assertIn("the run ended inside case b", lines[0])
        self.assertIn("INCOMPLETE", lines[0])

    def test_fewer_cases_than_expected_is_incomplete(self):
        self.write_all([case("a", True)], 3)
        self.assertTrue(self.summarize()[1])

    def test_an_expected_result_that_is_not_there_is_incomplete(self):
        for rank in (0, 1):
            self.write("sync_buffered", rank, [case("a", True)], 1)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertIn("results-libaio_direct-rank*.json: no results at all", lines[-1])

    def test_a_suite_that_wrote_nothing_is_missed(self):
        """The reason for the expected globs: nothing is left of a suite that did not run, to complain about."""
        self.write_all([case("a", True)], 1)
        lines, incomplete = self.summarize("results-sync_buffered-rank*.json", "other/results-*-rank*.json")
        self.assertTrue(incomplete)
        self.assertIn("other/results-*-rank*.json: no results at all", lines[-1])

    def test_a_rank_without_a_results_file_is_incomplete(self):
        """A rank that died before its first write leaves the file of the other rank, so the run has results."""
        for run in RUNS:
            self.write(run, 0, [case("a", True)], 1)
        self.write("sync_buffered", 1, [case("a", True)], 1)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertEqual([line for line in lines if "no results file" in line],
                         ["results-libaio_direct-rank*.json: no results file of rank 1  <-- INCOMPLETE"])

    def test_the_first_rank_missing_is_found_too(self):
        for run in RUNS:
            self.write(run, 1, [case("a", True)], 1)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertEqual(sum("no results file of rank 0" in line for line in lines), 2)

    def test_results_in_subdirectories_need_both_ranks_too_and_are_listed_with_their_path(self):
        self.write_all([case("a", True)], 1)
        self.write("default", 0, [case("s", True)], 1, subdirectory="smoke")
        self.write("sync_buffered", 1, [case("c", False, fault="leak")], 1, subdirectory="self-check")
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        missing = [line.split(":")[0] for line in lines if "no results file" in line]
        self.assertEqual(missing, [os.path.join("self-check", "results-sync_buffered-rank*.json"),
                                   os.path.join("smoke", "results-default-rank*.json")])
        self.assertTrue(any(line.startswith(os.path.join("self-check", "results-sync_buffered-rank1.json"))
                            for line in lines), lines)

    def test_a_run_of_one_rank_needs_one_results_file_and_says_what_it_left_out(self):
        for run in RUNS:
            self.write(run, 0, [case("a", True)], 1, world=1, skipped=60, note="ONE GPU: 60 distributed rows were not run")
        lines, incomplete = self.summarize()
        self.assertFalse(incomplete, lines)
        self.assertEqual(lines[-1], "ONE GPU: 60 distributed rows were not run")
        self.assertEqual(lines.count("ONE GPU: 60 distributed rows were not run"), 1)

    def test_a_run_of_one_rank_that_has_no_results_file_is_still_incomplete(self):
        self.write("sync_buffered", 0, [case("a", True)], 1, world=1)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertIn("results-libaio_direct-rank*.json: no results at all", lines[-1])

    def test_a_run_of_two_ranks_still_needs_both_files_when_the_file_says_so(self):
        for run in RUNS:
            self.write(run, 0, [case("a", True)], 1, world=2)
        self.assertTrue(self.summarize()[1])

    def test_a_note_without_skipped_cases_is_not_printed(self):
        self.write_all([case("a", True)], 1)
        self.write("sync_buffered", 0, [case("a", True)], 1, world=1, skipped=0, note="stale note")
        self.assertNotIn("stale note", self.summarize()[0])

    def test_the_ranks_are_those_of_the_harness(self):
        self.assertEqual(summarize.DEFAULT_WORLD, DEFAULT_WORLD)

    def test_a_run_with_no_case_to_run_is_complete(self):
        self.write_all([], 0)
        lines, incomplete = self.summarize()
        self.assertFalse(incomplete)
        self.assertIn("0 of 0 cases finished", lines[0])

    def test_the_file_names_are_those_that_the_summary_reads(self):
        self.assertRegex(results_filename("sync_buffered", 1), summarize._FILE.pattern)
        self.assertEqual(results_filename("a,b/c", 0), "results-a+b_c-rank0.json")
        self.assertEqual(results_filename("", 0), "results-default-rank0.json")


class CommandLine(unittest.TestCase):
    def test_exit_status_follows_the_summary(self):
        with tempfile.TemporaryDirectory(prefix="summarize-") as directory:
            for rank in (0, 1):
                write_results(directory, "s", "sync_buffered", rank, [case("a", True)], 1, world=2)
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(summarize.main([directory, "results-sync_buffered-rank*.json"]), 0)
                self.assertEqual(summarize.main([directory, "results-sync_buffered-rank*.json",
                                                 "results-libaio_direct-rank*.json"]), 1)
            self.assertIn("1 of 1 cases finished", out.getvalue())

    def test_a_wrong_usage_is_refused(self):
        self.assertEqual(summarize.main([]), 2)

    def test_the_files_are_json(self):
        with tempfile.TemporaryDirectory(prefix="summarize-") as directory:
            write_results(directory, "s", "r", 0, [case("a", True)], 1)
            with open(os.path.join(directory, "results-r-rank0.json")) as handle:
                data = json.load(handle)
        self.assertEqual((data["suite"], data["run"], data["expected_cases"], data["world"]), ("s", "r", 1, 1))


if __name__ == "__main__":
    unittest.main()
