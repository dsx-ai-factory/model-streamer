# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the summary of the rows that ran, with made-up results files.

    python3 -m unittest tests/gpu/test_rows_summary.py
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rows_summary  # noqa: E402

STRATEGIES = ["sync_buffered", "libaio_direct"]


def row(name, ok, **extra):
    entry = {"row": name, "strategy": "x", "ok": ok}
    entry.update(extra)
    return entry


class Summary(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="rows-summary-")
        self.addCleanup(self.dir.cleanup)

    def write(self, name, rows, expected, subdirectory=None):
        directory = os.path.join(self.dir.name, subdirectory) if subdirectory else self.dir.name
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, name), "w") as handle:
            json.dump({"strategy": "x", "expected_rows": expected, "rows": rows}, handle)

    def write_all(self, rows, expected):
        for strategy in STRATEGIES:
            for rank in (0, 1):
                self.write("rows-%s-rank%d.json" % (strategy, rank), rows, expected)

    def summarize(self, smoke=False):
        return rows_summary.summarize(self.dir.name, smoke, STRATEGIES)

    def test_every_row_finished_is_complete(self):
        self.write_all([row("a", True), row("b", True)], 2)
        lines, incomplete = self.summarize()
        self.assertFalse(incomplete)
        self.assertEqual(len(lines), 4)
        self.assertIn("2 of 2 rows finished, 0 failed", lines[0])

    def test_failed_rows_are_counted_but_do_not_make_the_run_incomplete(self):
        self.write_all([row("a", True), row("b", False)], 2)
        lines, incomplete = self.summarize()
        self.assertFalse(incomplete)
        self.assertIn("2 of 2 rows finished, 1 failed", lines[0])

    def test_a_row_that_failed_on_purpose_is_not_a_failure(self):
        self.write_all([row("a", False, drill="corrupt", drill_caught=True), row("b", False)], 2)
        lines, _ = self.summarize()
        self.assertIn("1 failed", lines[0])

    def test_a_row_with_an_injected_fault_is_not_a_failure_either(self):
        """The self-check injects a fault without declaring a drill, to prove that a failing row fails the run."""
        self.write_all([row("a", False, drill=None, fault="corrupt"), row("b", False)], 2)
        lines, _ = self.summarize()
        self.assertIn("1 failed", lines[0])

    def test_a_row_left_started_names_the_row_where_the_run_ended(self):
        self.write_all([row("a", True), row("b", None)], 5)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertIn("1 of 5 rows finished", lines[0])
        self.assertIn("the run ended inside row b", lines[0])
        self.assertIn("INCOMPLETE", lines[0])

    def test_fewer_rows_than_expected_is_incomplete(self):
        self.write_all([row("a", True)], 3)
        _, incomplete = self.summarize()
        self.assertTrue(incomplete)

    def test_a_strategy_without_any_results_is_incomplete(self):
        for rank in (0, 1):
            self.write("rows-sync_buffered-rank%d.json" % rank, [row("a", True)], 1)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertIn("libaio_direct: no results at all", lines[-1])

    def test_a_rank_without_a_results_file_is_incomplete(self):
        """A rank that died before its first write leaves the file of the other rank, so the strategy has results."""
        for strategy in STRATEGIES:
            self.write("rows-%s-rank0.json" % strategy, [row("a", True)], 1)
        self.write("rows-sync_buffered-rank1.json", [row("a", True)], 1)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertEqual([line for line in lines if "no results file" in line],
                         ["rows-libaio_direct-rank*.json: no results file of rank 1  <-- INCOMPLETE"])

    def test_the_first_rank_missing_is_found_too(self):
        for strategy in STRATEGIES:
            self.write("rows-%s-rank1.json" % strategy, [row("a", True)], 1)
        lines, incomplete = self.summarize()
        self.assertTrue(incomplete)
        self.assertEqual(sum("no results file of rank 0" in line for line in lines), 2)

    def test_the_smoke_and_self_check_results_need_both_ranks_too(self):
        self.write_all([row("a", True)], 1)
        self.write("rows-default-rank0.json", [row("s", True)], 1, subdirectory="smoke")
        self.write("rows-sync_buffered-rank1.json", [row("c", False, drill="leak")], 1, subdirectory="self-check")
        lines, incomplete = self.summarize(smoke=True)
        self.assertTrue(incomplete)
        missing = [line.split(":")[0] for line in lines if "no results file" in line]
        self.assertEqual(missing, [os.path.join("self-check", "rows-sync_buffered-rank*.json"),
                                   os.path.join("smoke", "rows-default-rank*.json")])

    def test_the_ranks_are_those_of_the_test(self):
        import matrix
        self.assertEqual(rows_summary.RANKS, matrix.WORLD)

    def test_a_leg_with_no_row_to_run_is_complete(self):
        self.write_all([], 0)
        lines, incomplete = self.summarize()
        self.assertFalse(incomplete)
        self.assertIn("0 of 0 rows finished", lines[0])

    def test_the_smoke_test_is_expected_only_when_it_was_to_run(self):
        self.write_all([row("a", True)], 1)
        self.assertFalse(self.summarize(smoke=False)[1])
        lines, incomplete = self.summarize(smoke=True)
        self.assertTrue(incomplete)
        self.assertIn("smoke: no results at all", lines[-1])
        for rank in (0, 1):
            self.write("rows-default-rank%d.json" % rank, [row("s", True)], 1, subdirectory="smoke")
        self.assertFalse(self.summarize(smoke=True)[1])

    def test_results_in_subdirectories_are_listed_with_their_path(self):
        self.write_all([row("a", True)], 1)
        for rank in (0, 1):
            self.write("rows-sync_buffered-rank%d.json" % rank, [row("c", False, drill="leak")], 1,
                       subdirectory="self-check")
        lines, _ = self.summarize()
        self.assertTrue(any(line.startswith(os.path.join("self-check", "rows-sync_buffered-rank0.json"))
                            for line in lines), lines)


class CommandLine(unittest.TestCase):
    def test_exit_status_follows_the_summary(self):
        with tempfile.TemporaryDirectory(prefix="rows-summary-") as directory:
            for rank in (0, 1):
                with open(os.path.join(directory, "rows-sync_buffered-rank%d.json" % rank), "w") as handle:
                    json.dump({"strategy": "x", "expected_rows": 1, "rows": [row("a", True)]}, handle)
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(rows_summary.main([directory, "false", "sync_buffered"]), 0)
                self.assertEqual(rows_summary.main([directory, "false", "sync_buffered", "libaio_direct"]), 1)
            self.assertIn("1 of 1 rows finished", out.getvalue())

    def test_a_wrong_usage_is_refused(self):
        self.assertEqual(rows_summary.main(["only-a-directory"]), 2)
        self.assertEqual(rows_summary.main(["directory", "maybe"]), 2)


if __name__ == "__main__":
    unittest.main()
