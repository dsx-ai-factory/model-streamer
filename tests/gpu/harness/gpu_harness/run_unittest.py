# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ordinary unittest modules in one process, and write their results the way every suite does (results.py), so that
the summary of the run sees them: how many cases finished, which failed, and where a killed run ended.

    python3 -m gpu_harness.run_unittest --suite NAME --run RUN [--results DIR] [--allow-skips] TEST [TEST ...]

TEST is a test file (tests/gpu/suites/NAME/test_x.py) or a dotted name (module, module.Class, module.Class.test_method).
Every test is a case, named by its unittest id. A test that is skipped fails the run, as everywhere in these tests (a GPU
test that cannot run is a failure, not a pass), unless --allow-skips. Running nothing fails too. This is the way to run a
test of one process; a test whose ranks agree on its outcome uses a Session instead (session.py) and torchrun.
"""

import argparse
import os
import sys
import time
import unittest

from .phase_timer import dump_jsonl
from .results import timing_filename, write_results


class RecordingResult(unittest.TextTestResult):
    """Text results, and a results file that is rewritten when a test starts and when it ends."""

    def __init__(self, stream, descriptions, verbosity, recorder):
        super().__init__(stream, descriptions, verbosity)
        self.recorder = recorder

    def startTest(self, test):
        super().startTest(test)
        self.recorder.start(test.id())

    def addSuccess(self, test):
        super().addSuccess(test)
        self.recorder.finish(test.id(), self.recorder.current_ok(True))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.recorder.finish(test.id(), False, self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.recorder.finish(test.id(), False, self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.recorder.allow_skips:
            self.recorder.finish(test.id(), True, note="skipped: %s" % reason)
        else:
            self.recorder.finish(test.id(), False, "skipped (%s): a GPU test that cannot run is a failure" % reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.recorder.finish(test.id(), True, note="expected failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.recorder.finish(test.id(), False, "passed, and was expected to fail")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.recorder.add_error(test.id(), "%s: %s" % (subtest.id(), self._exc_info_to_string(err, test)))

    def stopTest(self, test):
        super().stopTest(test)
        # A test whose only failures are those of its subtests gets no result of its own from unittest.
        self.recorder.close(test.id())


class Recorder:
    """The cases of this run, and the files they are written to."""

    def __init__(self, suite, run, results_dir, expected, allow_skips):
        self.suite, self.run, self.results_dir = suite, run, results_dir
        self.expected, self.allow_skips = expected, allow_skips
        self.cases = []
        self.timing = []
        self._started = {}
        self._subtest_errors = {}

    def start(self, case):
        self.cases.append({"case": case, "ok": None})
        self._started[case] = time.perf_counter()
        self.write()

    def add_error(self, case, text):
        self._subtest_errors.setdefault(case, []).append(text)

    def current_ok(self, ok):
        """A test whose subtests failed is not a success, whatever the test itself reports."""
        return ok and not any(self._subtest_errors.values())

    def close(self, case):
        """The test has ended: if no result was recorded for it (only subtests failed, or it ended in a way that unittest
        does not report), record one now."""
        if self.entry(case)["ok"] is None:
            self.finish(case, True)

    def entry(self, case):
        """The case that started. A failure or a skip in setUpClass, tearDownClass or setUpModule is reported by unittest
        without any test having started (against a holder, named like "setUpClass (module.Class)"): it is a case of its
        own, added when it is reported, and it fails the run."""
        for found in reversed(self.cases):
            if found["case"] == case:
                return found
        created = {"case": case, "ok": None}
        self.cases.append(created)
        return created

    def finish(self, case, ok, error=None, note=None):
        entry = self.entry(case)
        seconds = time.perf_counter() - self._started.pop(case, time.perf_counter())
        errors = list(self._subtest_errors.pop(case, [])) + ([error] if error else [])
        entry.update({"ok": ok and not errors, "seconds": seconds})
        if errors:
            entry["errors"] = errors
        if note:
            entry["note"] = note
        self.timing.append({"phase": "%s.%s" % (self.suite, case), "seconds": seconds})
        self.write()

    def write(self):
        write_results(self.results_dir, self.suite, self.run, 0, self.cases, self.expected, world=1)
        dump_jsonl(self.timing, os.path.join(self.results_dir, timing_filename(self.run, 0)))


def load(names):
    """The tests that the names say: a file of tests, or a dotted name."""
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for name in names:
        if name.endswith(".py") and os.path.exists(name):
            directory, module = os.path.split(os.path.abspath(name))
            if directory not in sys.path:
                sys.path.insert(0, directory)
            name = module[:-3]
        suite.addTests(loader.loadTestsFromName(name))
    return suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--results", default=os.environ.get("GPU_RESULTS", "test-results/gpu"))
    parser.add_argument("--allow-skips", action="store_true")
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args(argv)

    tests = load(args.tests)
    expected = tests.countTestCases()
    if expected == 0:
        print("no test was found in %s" % " ".join(args.tests), file=sys.stderr)
        return 1
    recorder = Recorder(args.suite, args.run, args.results, expected, args.allow_skips)
    recorder.write()  # a run killed before its first test still has a file, and says how many it was to run
    runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=2,
                                     resultclass=lambda stream, descriptions, verbosity: RecordingResult(
                                         stream, descriptions, verbosity, recorder))
    outcome = runner.run(tests)
    failed = not outcome.wasSuccessful() or any(case["ok"] is not True for case in recorder.cases)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
