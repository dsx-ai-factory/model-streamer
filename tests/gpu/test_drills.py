# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the fault injection of the GPU test: which errors count as the report of an injected fault.

    python3 -m unittest tests/gpu/test_drills.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drills  # noqa: E402

MIB = 1 << 20
CRC_ERROR = "model.layers.3.q_proj.weight (BF16 [4, 8], 64 bytes): crc32 0badf00d, expected 12345678"
LEAK_ERROR = "GPU memory leak: 32.0 MiB still allocated after the row, 0.0 MiB before the run"


class ParseDrills(unittest.TestCase):
    def test_kinds_and_rows(self):
        self.assertEqual(drills.parse_drills("corrupt:a.b,leak:c.d"), {"a.b": "corrupt", "c.d": "leak"})

    def test_spaces_and_empty_entries_are_ignored(self):
        self.assertEqual(drills.parse_drills(" corrupt:a , ,leak:b,"), {"a": "corrupt", "b": "leak"})
        self.assertEqual(drills.parse_drills(""), {})

    def test_an_unknown_kind_or_a_missing_row_is_refused(self):
        for text in ("raise:a", "corrupt", "corrupt:", ":a"):
            with self.assertRaises(RuntimeError, msg=text):
                drills.parse_drills(text)


class CorruptDrill(unittest.TestCase):
    def test_the_crc_error_of_the_corrupted_tensor_is_the_report(self):
        self.assertTrue(drills.drill_caught("corrupt", [CRC_ERROR], "model.layers.3.q_proj.weight"))

    def test_a_crc_error_of_another_tensor_is_not(self):
        self.assertFalse(drills.drill_caught("corrupt", [CRC_ERROR], "model.layers.9.q_proj.weight"))

    def test_nothing_was_corrupted_so_nothing_can_be_reported(self):
        self.assertFalse(drills.drill_caught("corrupt", [CRC_ERROR], None))

    def test_another_error_of_the_same_tensor_is_not(self):
        shape = "model.layers.3.q_proj.weight: shape [4, 9], expected [4, 8]"
        self.assertFalse(drills.drill_caught("corrupt", [shape], "model.layers.3.q_proj.weight"))

    def test_a_tensor_whose_name_starts_with_the_corrupted_one_is_not_it(self):
        error = "model.layers.3.q_proj.weight.bias (BF16 [4], 8 bytes): crc32 00000001, expected 00000002"
        self.assertFalse(drills.drill_caught("corrupt", [error], "model.layers.3.q_proj.weight"))

    def test_no_errors(self):
        self.assertFalse(drills.drill_caught("corrupt", [], "x"))


class LeakDrill(unittest.TestCase):
    def test_the_memory_error_with_the_leaked_bytes_is_the_report(self):
        self.assertTrue(drills.drill_caught("leak", [LEAK_ERROR], leak_bytes=32 * MIB))
        self.assertTrue(drills.drill_caught("leak", [LEAK_ERROR], leak_bytes=40 * MIB))

    def test_a_smaller_excess_means_the_leak_did_not_happen(self):
        self.assertFalse(drills.drill_caught("leak", [LEAK_ERROR], leak_bytes=12 * MIB))

    def test_no_memory_error_is_no_report_whatever_the_excess(self):
        self.assertFalse(drills.drill_caught("leak", [CRC_ERROR], leak_bytes=64 * MIB))

    def test_leak_bytes_are_the_size_of_the_drill(self):
        self.assertEqual(drills.LEAK_DRILL_BYTES, 32 * MIB)


class UnexpectedErrors(unittest.TestCase):
    """The other problems of a drilled row, which the drill must not hide."""

    def test_the_report_alone_leaves_nothing_unexpected(self):
        self.assertEqual(drills.unexpected_errors("corrupt", [CRC_ERROR], "model.layers.3.q_proj.weight"), [])
        self.assertEqual(drills.unexpected_errors("leak", [LEAK_ERROR]), [])

    def test_another_error_in_the_row_is_unexpected(self):
        other = "model.layers.9.q_proj.weight: dtype torch.float16, expected torch.bfloat16"
        self.assertEqual(drills.unexpected_errors("corrupt", [CRC_ERROR, other], "model.layers.3.q_proj.weight"),
                         [other])
        exception = "RuntimeError: NCCL communicator was aborted"
        self.assertEqual(drills.unexpected_errors("leak", [LEAK_ERROR, exception]), [exception])

    def test_a_crc_error_of_another_tensor_is_unexpected(self):
        other_crc = "model.layers.9.q_proj.weight (BF16 [4], 8 bytes): crc32 00000001, expected 00000002"
        self.assertEqual(drills.unexpected_errors("corrupt", [CRC_ERROR, other_crc], "model.layers.3.q_proj.weight"),
                         [other_crc])

    def test_errors_that_were_not_recorded_are_unexpected(self):
        unexpected = drills.unexpected_errors("leak", [LEAK_ERROR], error_count=45)
        self.assertEqual(len(unexpected), 1)
        self.assertIn("44 more", unexpected[0])
        self.assertEqual(drills.unexpected_errors("leak", [LEAK_ERROR], error_count=1), [])

    def test_a_corrupt_drill_that_corrupted_nothing_has_every_error_unexpected(self):
        self.assertEqual(drills.unexpected_errors("corrupt", [CRC_ERROR], None), [CRC_ERROR])

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            drills.unexpected_errors("raise", ["x"])


class RunFailed(unittest.TestCase):
    """What turns a run red, row by row. The cases of an ordinary row and of a drilled one are opposites."""

    def test_an_ordinary_row_that_is_not_ok_fails_the_run(self):
        self.assertTrue(drills.run_failed(False, ok=False, drill=None, caught=None, unexpected=[]))

    def test_an_ordinary_row_that_is_ok_does_not(self):
        self.assertFalse(drills.run_failed(False, ok=True, drill=None, caught=None, unexpected=[]))

    def test_a_drilled_row_that_failed_as_it_must_does_not(self):
        self.assertFalse(drills.run_failed(False, ok=False, drill="corrupt", caught=True, unexpected=[]))
        self.assertFalse(drills.run_failed(False, ok=False, drill="leak", caught=True, unexpected=[]))

    def test_a_drilled_row_whose_fault_was_not_caught_fails_the_run(self):
        self.assertTrue(drills.run_failed(False, ok=False, drill="corrupt", caught=False, unexpected=[]))
        self.assertTrue(drills.run_failed(False, ok=True, drill="leak", caught=False, unexpected=[]))

    def test_a_drilled_row_with_another_error_fails_the_run(self):
        self.assertTrue(drills.run_failed(False, ok=False, drill="corrupt", caught=True, unexpected=["RuntimeError: x"]))

    def test_once_failed_the_run_stays_failed(self):
        for ok, drill, caught, unexpected in ((True, None, None, []), (False, "leak", True, []), (True, None, None, [])):
            self.assertTrue(drills.run_failed(True, ok, drill, caught, unexpected))

    def test_a_row_that_is_ok_and_drilled_with_nothing_caught_is_the_drill_not_doing_its_job(self):
        self.assertTrue(drills.run_failed(False, ok=True, drill="corrupt", caught=False, unexpected=[]))


class UnknownKind(unittest.TestCase):
    def test_refused(self):
        with self.assertRaises(ValueError):
            drills.drill_caught("raise", [])


if __name__ == "__main__":
    unittest.main()
