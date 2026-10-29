# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the check that a row failed as an ordinary row, with made-up results files.

    python3 -m unittest tests/gpu/test_expect_failure.py
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expect_failure  # noqa: E402

ROW = "mixed.solo.owned.cuda.tight"
CRC = "model.layers.3.q_proj.weight (BF16 [4, 8], 64 bytes): crc32 0badf00d, expected 12345678"


def row(ok, errors=(), drill=None):
    return {"row": ROW, "strategy": "sync_buffered", "ok": ok, "drill": drill, "errors": list(errors)}


class Check(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="expect-failure-")
        self.addCleanup(self.dir.cleanup)

    def write(self, rank, entry):
        with open(os.path.join(self.dir.name, "rows-sync_buffered-rank%d.json" % rank), "w") as handle:
            json.dump({"strategy": "sync_buffered", "expected_rows": 1, "rows": [] if entry is None else [entry]},
                      handle)

    def both(self, entry):
        for rank in (0, 1):
            self.write(rank, entry)

    def check(self, text="crc32"):
        return expect_failure.check(self.dir.name, "sync_buffered", ROW, text)

    def test_a_failed_ordinary_row_on_every_rank_is_what_is_expected(self):
        self.both(row(False, [CRC]))
        ok, message = self.check()
        self.assertTrue(ok, message)

    def test_a_row_that_was_ok_is_not(self):
        self.both(row(True))
        self.assertFalse(self.check()[0])

    def test_a_row_that_never_finished_is_not(self):
        self.both(row(None))
        self.assertIn("never finished", self.check()[1])

    def test_a_drilled_row_is_not_an_ordinary_one(self):
        self.both(row(False, [CRC], drill="corrupt"))
        self.assertIn("was a drill", self.check()[1])

    def test_a_failure_with_another_error_is_not_the_injected_one(self):
        """A crash or any other failure also makes a run exit with a failure: only the injected fault counts."""
        self.both(row(False, ["RuntimeError: NCCL communicator was aborted"]))
        self.assertFalse(self.check()[0])

    def test_one_rank_missing_is_not(self):
        self.write(0, row(False, [CRC]))
        self.assertIn("no results for rank 1", self.check()[1])

    def test_the_row_missing_from_a_rank_is_not(self):
        self.write(0, row(False, [CRC]))
        self.write(1, None)
        self.assertIn("rank 1 has no row", self.check()[1])

    def test_one_rank_with_an_ok_row_is_not(self):
        self.write(0, row(False, [CRC]))
        self.write(1, row(True))
        self.assertFalse(self.check()[0])


class CommandLine(unittest.TestCase):
    def test_exit_status_and_usage(self):
        with tempfile.TemporaryDirectory(prefix="expect-failure-") as directory:
            for rank in (0, 1):
                with open(os.path.join(directory, "rows-sync_buffered-rank%d.json" % rank), "w") as handle:
                    json.dump({"rows": [row(False, [CRC])]}, handle)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(expect_failure.main([directory, "sync_buffered", ROW, "crc32"]), 0)
                self.assertEqual(expect_failure.main([directory, "sync_buffered", ROW, "no such text"]), 1)
            with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                self.assertEqual(expect_failure.main([directory]), 2)

    def test_the_ranks_are_those_of_the_test(self):
        import matrix
        self.assertEqual(expect_failure.RANKS, matrix.WORLD)


if __name__ == "__main__":
    unittest.main()
