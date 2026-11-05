# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the table of phases that a run prints.

    python3 -m unittest tests/gpu/harness/test_phase_timer.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_harness.phase_timer import report  # noqa: E402


class Report(unittest.TestCase):
    def test_the_columns_line_up_for_a_phase_name_longer_than_the_usual_one(self):
        records = [
            {"phase": "run.sync_buffered", "seconds": 100.0},
            {"phase": "many_tensors.dist.chunks.borrowed.cuda.default.load", "seconds": 20.0, "bytes": 2**30},
        ]
        lines = report(records).split("\n")
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({len(line) for line in lines}), 1, "\n" + "\n".join(lines))
        share_end = lines[0].index("share") + len("share")
        for line in lines[1:]:
            self.assertEqual(line.index("%") + 1, share_end, "\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
