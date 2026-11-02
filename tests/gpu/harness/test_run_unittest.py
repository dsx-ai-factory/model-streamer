# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of gpu_harness.run_unittest, which runs the test modules of a suite in one process and writes their
results the way every suite does.

    python3 -m unittest tests/gpu/harness/test_run_unittest.py
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize  # noqa: E402


class RunUnittest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="run-unittest-")
        self.addCleanup(self.dir.cleanup)
        self.results = os.path.join(self.dir.name, "results")

    def module(self, name, body):
        path = os.path.join(self.dir.name, "test_%s.py" % name)
        with open(path, "w") as handle:
            handle.write(textwrap.dedent(body))
        return path

    def run_tests(self, *paths, extra=()):
        env = dict(os.environ, PYTHONPATH=HERE, GPU_RESULTS=self.results)
        done = subprocess.run([sys.executable, "-m", "gpu_harness.run_unittest", "--suite", "demo", "--run", "basic",
                               *extra, *paths], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        return done.returncode, done.stdout

    def results_file(self):
        with open(os.path.join(self.results, "results-basic-rank0.json")) as handle:
            return json.load(handle)

    def test_passing_tests_are_cases_that_passed(self):
        path = self.module("pass", """
            import unittest
            class T(unittest.TestCase):
                def test_a(self): pass
                def test_b(self): self.assertTrue(True)
            """)
        status, output = self.run_tests(path)
        self.assertEqual(status, 0, output)
        data = self.results_file()
        self.assertEqual((data["suite"], data["run"], data["world"], data["expected_cases"]), ("demo", "basic", 1, 2))
        self.assertEqual([c["ok"] for c in data["cases"]], [True, True])
        self.assertEqual(sorted(c["case"] for c in data["cases"]), ["test_pass.T.test_a", "test_pass.T.test_b"])
        lines, incomplete = summarize.summarize(self.results, ["results-basic-rank*.json"])
        self.assertFalse(incomplete, lines)

    def test_a_failure_and_an_error_fail_the_run_with_their_text(self):
        path = self.module("fail", """
            import unittest
            class T(unittest.TestCase):
                def test_fails(self): self.assertEqual(1, 2)
                def test_errors(self): raise RuntimeError("boom")
                def test_passes(self): pass
            """)
        status, output = self.run_tests(path)
        self.assertEqual(status, 1, output)
        by_name = {c["case"].split(".")[-1]: c for c in self.results_file()["cases"]}
        self.assertEqual({k: v["ok"] for k, v in by_name.items()}, {"test_fails": False, "test_errors": False,
                                                                     "test_passes": True})
        self.assertIn("AssertionError", by_name["test_fails"]["errors"][0])
        self.assertIn("boom", by_name["test_errors"]["errors"][0])

    def test_a_skipped_test_is_a_failure_unless_it_is_allowed(self):
        path = self.module("skip", """
            import unittest
            class T(unittest.TestCase):
                @unittest.skip("no gpu")
                def test_skipped(self): pass
            """)
        status, output = self.run_tests(path)
        self.assertEqual(status, 1, output)
        case = self.results_file()["cases"][0]
        self.assertFalse(case["ok"])
        self.assertIn("a GPU test that cannot run is a failure", case["errors"][0])
        status, output = self.run_tests(path, extra=["--allow-skips"])
        self.assertEqual(status, 0, output)
        self.assertTrue(self.results_file()["cases"][0]["ok"])

    def test_a_test_whose_only_failures_are_those_of_its_subtests_fails(self):
        path = self.module("sub", """
            import unittest
            class T(unittest.TestCase):
                def test_rows(self):
                    for row in ("a", "b", "c"):
                        with self.subTest(row=row):
                            self.assertNotEqual(row, "b")
            """)
        status, output = self.run_tests(path)
        self.assertEqual(status, 1, output)
        case = self.results_file()["cases"][0]
        self.assertFalse(case["ok"])
        self.assertIn("row='b'", case["errors"][0])

    def test_a_module_that_has_no_test_fails_the_run(self):
        path = self.module("empty", """
            import unittest
            class NotATest: pass
            """)
        status, output = self.run_tests(path)
        self.assertEqual(status, 1, output)
        self.assertIn("no test was found", output)

    def test_a_run_that_is_killed_in_the_middle_of_a_test_names_it(self):
        """The reason for writing a case as started: the file of a run that never ended says where it was."""
        path = self.module("kill", """
            import os, unittest
            class T(unittest.TestCase):
                def test_a(self): pass
                def test_b_dies(self): os._exit(9)
                def test_c(self): pass
            """)
        status, _ = self.run_tests(path)
        self.assertEqual(status, 9)
        data = self.results_file()
        self.assertEqual(data["expected_cases"], 3)
        self.assertEqual([(c["case"].split(".")[-1], c["ok"]) for c in data["cases"]],
                         [("test_a", True), ("test_b_dies", None)])
        lines, incomplete = summarize.summarize(self.results, ["results-basic-rank*.json"])
        self.assertTrue(incomplete)
        self.assertIn("the run ended inside case test_kill.T.test_b_dies", lines[0])

    def test_dotted_names_select_a_class_or_a_test(self):
        self.module("dotted", """
            import unittest
            class A(unittest.TestCase):
                def test_one(self): pass
                def test_two(self): pass
            class B(unittest.TestCase):
                def test_three(self): pass
            """)
        env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + self.dir.name, GPU_RESULTS=self.results)
        done = subprocess.run([sys.executable, "-m", "gpu_harness.run_unittest", "--suite", "demo", "--run", "basic",
                               "test_dotted.A.test_two", "test_dotted.B"], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertEqual(sorted(c["case"] for c in self.results_file()["cases"]),
                         ["test_dotted.A.test_two", "test_dotted.B.test_three"])

    def test_the_timing_of_every_case_is_written_for_the_table_of_the_run(self):
        path = self.module("timed", """
            import unittest
            class T(unittest.TestCase):
                def test_a(self): pass
            """)
        self.run_tests(path)
        with open(os.path.join(self.results, "timing-basic-rank0.jsonl")) as handle:
            phases = [json.loads(line)["phase"] for line in handle if line.strip()]
        self.assertEqual(phases, ["demo.test_timed.T.test_a"])


if __name__ == "__main__":
    unittest.main()
