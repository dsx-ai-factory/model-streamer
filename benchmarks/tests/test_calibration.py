# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The calibration helpers that need no Linux, GPU or storage."""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibration  # noqa: E402
from calibration import memory  # noqa: E402


class CalibrationTest(unittest.TestCase):
    def test_slabs_cover_the_region_once(self):
        for size, parts in ((10 << 20, 3), (64 << 20, 8), (2 << 20, 32), (5 << 20, 1)):
            pieces = memory.slabs(size, parts)
            self.assertEqual(pieces[0][0], 0)
            self.assertEqual(sum(n for _, n in pieces), size)
            for (off, n), (nxt, _) in zip(pieces, pieces[1:]):
                self.assertEqual(off + n, nxt)

    def test_copy_moves_the_bytes(self):
        src, dst = memory.Region(4 << 20), memory.Region(4 << 20)
        src.view[:4] = b"abcd"
        src.view[-4:] = b"wxyz"
        memory.memcpy(dst, src, 2)
        self.assertEqual((bytes(dst.view[:4]), bytes(dst.view[-4:])), (b"abcd", b"wxyz"))

    def test_meter_keeps_every_run_and_the_median_is_the_ceiling(self):
        meter = calibration.Meter(repeats=3)
        with contextlib.redirect_stdout(io.StringIO()):
            meter.record("h2d_pinned", [4.0, 6.0, 5.0])
            meter.rate("host_memcpy_1t", lambda: None, 1 << 30)
        self.assertEqual(meter.raw["h2d_pinned"], [4.0, 6.0, 5.0])
        self.assertEqual(calibration.median_of(meter.raw["h2d_pinned"]), 5.0)
        self.assertEqual(len(meter.raw["host_memcpy_1t"]), 3)
        self.assertIsNone(calibration.median_of([]))


if __name__ == "__main__":
    unittest.main()
