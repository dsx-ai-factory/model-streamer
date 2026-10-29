# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of what the self-checks of the GPU test accept as the report of an injected fault.

    python3 -m unittest tests/gpu/test_drills.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drills  # noqa: E402

MIB = 1 << 20
TENSOR = "model.layers.3.q_proj.weight"
CRC_ERROR = TENSOR + " (BF16 [4, 8], 64 bytes): crc32 0badf00d, expected 12345678"
LEAK_ERROR = "GPU memory leak: 32.0 MiB still allocated after the row, 0.0 MiB before the run"


class CrcError(unittest.TestCase):
    def test_the_crc_error_of_the_tensor_is_the_report(self):
        self.assertTrue(drills.is_crc_error_of(CRC_ERROR, TENSOR))

    def test_a_crc_error_of_another_tensor_is_not(self):
        self.assertFalse(drills.is_crc_error_of(CRC_ERROR, "model.layers.9.q_proj.weight"))

    def test_a_tensor_whose_name_starts_with_the_corrupted_one_is_not_it(self):
        error = TENSOR + ".bias (BF16 [4], 8 bytes): crc32 00000001, expected 00000002"
        self.assertFalse(drills.is_crc_error_of(error, TENSOR))

    def test_another_error_of_the_same_tensor_is_not(self):
        self.assertFalse(drills.is_crc_error_of(TENSOR + ": shape [4, 9], expected [4, 8]", TENSOR))
        self.assertFalse(drills.is_crc_error_of(TENSOR + ": dtype torch.float16, expected torch.bfloat16", TENSOR))

    def test_nothing_was_corrupted_so_nothing_can_be_reported(self):
        self.assertFalse(drills.is_crc_error_of(CRC_ERROR, None))

    def test_the_error_that_the_row_check_produces_is_recognised(self):
        """The format of RowRun.check, which this reads: name, then the details in parentheses, then crc32."""
        error = "%s (%s %s, %d bytes): crc32 %08x, expected %08x" % (TENSOR, "BF16", [4, 8], 64, 0xBADF00D, 0x12345678)
        self.assertTrue(drills.is_crc_error_of(error, TENSOR))


class LeakError(unittest.TestCase):
    def test_the_memory_error_is_recognised(self):
        self.assertTrue(drills.is_leak_error(LEAK_ERROR))

    def test_other_errors_are_not(self):
        self.assertFalse(drills.is_leak_error(CRC_ERROR))
        self.assertFalse(drills.is_leak_error("RuntimeError: NCCL communicator was aborted"))

    def test_the_leak_of_the_drill_is_32_mib(self):
        self.assertEqual(drills.LEAK_DRILL_BYTES, 32 * MIB)


if __name__ == "__main__":
    unittest.main()
