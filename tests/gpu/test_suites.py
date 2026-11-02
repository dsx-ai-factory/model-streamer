# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of how the harness runs suites, and of what every suite of tests/gpu/suites/ must be (the contract that
tests/gpu/README.md, "Adding a test", describes): they fail when a suite is added in a way that the runner cannot run.

    python3 -m unittest tests/gpu/test_suites.py

The runner is exercised with suites of its own, made up here, and no GPU: bash, python3 and timeout are what it needs.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SUITES = os.path.join(HERE, "suites")
HARNESS = os.path.join(HERE, "harness")


def suite_names():
    return sorted(name for name in os.listdir(SUITES) if os.path.isdir(os.path.join(SUITES, name))
                  and not name.startswith((".", "_")))


def declared(name):
    """What suite.sh of a suite defines, by sourcing it (it only defines variables and functions): the description, the
    names of its functions and its unit tests."""
    script = textwrap.dedent("""
        set -u
        source "%s/suite.sh"
        echo "description=${suite_description:-}"
        for f in suite_prepare suite_run suite_expect suite_collect suite_timing; do
            if declare -F "$f" >/dev/null; then echo "function=$f"; fi
        done
        for t in ${suite_unit_tests[@]+"${suite_unit_tests[@]}"}; do echo "unit_test=$t"; done
    """ % os.path.join(SUITES, name))
    done = subprocess.run(["bash", "-c", script], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if done.returncode != 0:
        raise AssertionError("suite.sh of %s does not source: %s" % (name, done.stderr))
    found = {"description": "", "functions": [], "unit_tests": []}
    for line in done.stdout.splitlines():
        key, _, value = line.partition("=")
        if key == "description":
            found["description"] = value
        elif key == "function":
            found["functions"].append(value)
        elif key == "unit_test":
            found["unit_tests"].append(value)
    return found


class Contract(unittest.TestCase):
    """What a suite is. Every directory under tests/gpu/suites/ is checked, so a new suite is checked from its first day."""

    def test_there_is_a_suite_and_the_streaming_suite_is_one_of_them(self):
        self.assertIn("streaming", suite_names())

    def test_every_suite_has_a_suite_sh_a_readme_and_a_test(self):
        for name in suite_names():
            directory = os.path.join(SUITES, name)
            self.assertTrue(os.path.isfile(os.path.join(directory, "suite.sh")), name)
            self.assertTrue(os.path.isfile(os.path.join(directory, "README.md")), name + ": a README.md says what it tests")
            tests = [f for f in os.listdir(directory) if f.startswith("test_") and f.endswith(".py")]
            self.assertTrue(tests, name + ": no test_*.py")

    def test_suite_sh_is_valid_bash_that_describes_the_suite_and_defines_suite_run(self):
        for name in suite_names():
            done = subprocess.run(["bash", "-n", os.path.join(SUITES, name, "suite.sh")], stderr=subprocess.PIPE,
                                  universal_newlines=True)
            self.assertEqual(done.returncode, 0, "%s: %s" % (name, done.stderr))
            found = declared(name)
            self.assertTrue(found["description"], name + ": suite_description is empty")
            self.assertIn("suite_run", found["functions"], name)
            # Without it a suite whose tests wrote nothing (a script that exits 0 and does not run its tests) is green.
            self.assertIn("suite_expect", found["functions"], name + ": suite_expect says what the suite must have written")
            self.assertLess(len(found["description"]), 160, name + ": a description is one line")

    def test_the_unit_tests_that_a_suite_names_exist_and_are_its_own(self):
        for name in suite_names():
            for path in declared(name)["unit_tests"]:
                self.assertTrue(os.path.isfile(os.path.join(REPO, path)), "%s: %s" % (name, path))
                self.assertTrue(path.startswith("tests/gpu/suites/%s/" % name), "%s names a test that is not its own: %s" % (
                    name, path))

    def test_every_profile_of_the_default_environment_is_the_name_of_a_suite(self):
        """The services of a suite have a profile of the suite's name, which run.sh switches on for the suites that are
        selected: a profile with another name would never start."""
        with open(os.path.join(HERE, "compose.yaml")) as handle:
            compose = handle.read()
        profiles = set()
        for group in re.findall(r"profiles:\s*\[([^\]]*)\]", compose):
            profiles.update(p.strip() for p in group.split(",") if p.strip())
        self.assertTrue(profiles, "compose.yaml has no profile: the object_storage suite has no service")
        self.assertLessEqual(profiles, set(suite_names()), "a profile of compose.yaml is not the name of a suite")
        self.assertIn("object_storage", profiles)

    def test_the_harness_does_not_know_the_suites_and_a_suite_does_not_know_another(self):
        for base, _, files in os.walk(HARNESS):
            for filename in files:
                if filename.endswith((".py", ".sh")) and not filename.startswith("test_"):
                    with open(os.path.join(base, filename)) as handle:
                        text = handle.read()
                    for name in suite_names():
                        self.assertNotIn("suites/%s" % name, text, "the harness names the suite %s in %s" % (name, filename))
        for name in suite_names():
            others = [other for other in suite_names() if other != name]
            for base, _, files in os.walk(os.path.join(SUITES, name)):
                for filename in files:
                    if filename.endswith((".py", ".sh")):
                        with open(os.path.join(base, filename)) as handle:
                            text = handle.read()
                        for other in others:
                            self.assertNotIn("suites/%s/" % other, text, "%s uses the suite %s (%s)" % (name, other, filename))

    def test_every_python_file_of_a_suite_that_is_run_by_a_script_names_its_path_from_the_root(self):
        """What suite.sh runs is a path from the root of the checkout, which is where the entry runs it."""
        for name in suite_names():
            with open(os.path.join(SUITES, name, "suite.sh")) as handle:
                text = handle.read()
            for path in re.findall(r"tests/gpu/suites/%s/[\w./-]*\w" % name, text):
                self.assertTrue(os.path.exists(os.path.join(REPO, path)), "%s: suite.sh names %s, which is not there" % (
                    name, path))


@unittest.skipUnless(shutil.which("timeout") and shutil.which("bash"), "the runner needs bash and timeout")
class Runner(unittest.TestCase):
    """The runner (harness/common.sh and suites.sh), with suites made up here: what it does with a suite that passes,
    one that fails, one that cannot be prepared, one that writes nothing, and a name that is not a suite."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="suites-")
        self.addCleanup(self.dir.cleanup)
        self.suites = os.path.join(self.dir.name, "suites")
        self.results = os.path.join(self.dir.name, "results")

    def make_suite(self, name, body, tests=None):
        directory = os.path.join(self.suites, name)
        os.makedirs(directory)
        with open(os.path.join(directory, "suite.sh"), "w") as handle:
            handle.write(textwrap.dedent(body))
        if tests is not None:
            with open(os.path.join(directory, "test_%s.py" % name), "w") as handle:
                handle.write(textwrap.dedent(tests))
        return directory

    def run_runner(self, suites, env=None):
        script = textwrap.dedent("""
            set -euo pipefail
            cd "%(repo)s"
            export WORLD=1 RESULTS="%(results)s" FIXTURES="%(fixtures)s" GPU_SEED=1
            export PYTHONPATH="%(harness)s${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
            mkdir -p "$RESULTS" "$FIXTURES"
            SETUP_TIMING="$RESULTS/timing-setup.jsonl"; : > "$SETUP_TIMING"
            . tests/gpu/harness/common.sh
            . tests/gpu/harness/suites.sh
            SUITES_DIR="%(suites)s"
            select_suites || exit 2
            run_suites
            report_suites
            finished=true
            exit "$status"
        """ % {"repo": REPO, "results": self.results, "fixtures": os.path.join(self.dir.name, "fixtures"),
               "harness": HARNESS, "suites": self.suites})
        # Not the GPU_* variables of whatever runs this test (the container of a run has its own GPU_SUITES, GPU_ROWS ...)
        environment = {k: v for k, v in os.environ.items() if not k.startswith("GPU_")}
        environment["GPU_SUITES"] = suites
        environment.update(env or {})
        done = subprocess.run(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, env=environment)
        return done.returncode, done.stdout

    PASSING = """
        import unittest
        class Cases(unittest.TestCase):
            def test_a(self): pass
            def test_b(self): self.assertEqual(1, 1)
        """
    FAILING = """
        import unittest
        class Cases(unittest.TestCase):
            def test_a(self): pass
            def test_b(self): self.assertEqual(1, 2)
        """

    def unittest_suite(self, name, tests):
        return self.make_suite(name, """
            suite_description="a suite of unittest modules"
            suite_run() {
                run_unittest cases 1 "%s"
            }
            suite_expect() { echo "results-cases-rank0.json"; }
            """ % os.path.join(self.suites, name, "test_%s.py" % name), tests)

    def test_a_suite_that_passes_gives_a_green_run_with_its_cases_in_the_summary(self):
        self.unittest_suite("good", self.PASSING)
        status, output = self.run_runner("good")
        self.assertEqual(status, 0, output)
        self.assertIn("results-cases-rank0.json: 2 of 2 cases finished, 0 failed", output)
        self.assertNotIn("THE RUN ENDED EARLY", output)
        self.assertTrue(os.path.exists(os.path.join(self.results, "good", "results-cases-rank0.json")))

    def test_a_failing_test_fails_the_run_and_is_named(self):
        self.unittest_suite("bad", self.FAILING)
        status, output = self.run_runner("bad")
        self.assertEqual(status, 1, output)
        self.assertIn("2 of 2 cases finished, 1 failed", output)

    def test_every_suite_runs_even_when_an_earlier_one_fails(self):
        self.unittest_suite("bad", self.FAILING)
        self.unittest_suite("good", self.PASSING)
        status, output = self.run_runner("bad,good")
        self.assertEqual(status, 1, output)
        self.assertIn("bad/results-cases-rank0.json: 2 of 2 cases finished, 1 failed", output)
        self.assertIn("good/results-cases-rank0.json: 2 of 2 cases finished, 0 failed", output)

    def test_a_suite_that_cannot_be_prepared_is_not_run_and_fails_the_run(self):
        directory = self.make_suite("unprepared", """
            suite_description="fails to prepare"
            suite_prepare() { echo "no fixtures"; return 1; }
            suite_run() { touch "%s/ran"; }
            suite_expect() { echo "results-cases-rank0.json"; }
            """ % self.dir.name)
        status, output = self.run_runner("unprepared")
        self.assertEqual(status, 1, output)
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, "ran")), "the tests of a suite that failed to prepare ran")
        self.assertIn("could not be prepared", output)
        self.assertIn("unprepared/results-cases-rank0.json: no results at all", output)
        del directory

    def test_a_suite_that_writes_nothing_that_it_should_is_missed(self):
        self.make_suite("silent", """
            suite_description="runs and writes no results"
            suite_run() { run_step nothing 1 true; }
            suite_expect() { echo "results-nothing-rank0.json"; }
            """)
        status, output = self.run_runner("silent")
        self.assertEqual(status, 1, output)
        self.assertIn("silent/results-nothing-rank0.json: no results at all", output)

    def test_a_step_that_fails_fails_the_run_and_the_next_step_runs(self):
        self.make_suite("steps", """
            suite_description="two steps, the first fails"
            suite_run() {
                run_step first 1 false
                run_step second 1 touch "%s/second-ran"
            }
            """ % self.dir.name)
        status, output = self.run_runner("steps")
        self.assertEqual(status, 1, output)
        self.assertIn("FAILED run.first", output)
        self.assertTrue(os.path.exists(os.path.join(self.dir.name, "second-ran")))

    def test_a_cap_is_cut_to_what_is_left_of_the_budget_and_is_never_less_than_a_minute(self):
        script = textwrap.dedent("""
            set -euo pipefail
            cd "%s"
            export WORLD=1 RESULTS="%s" SETUP_TIMING="%s/t.jsonl"
            mkdir -p "$RESULTS"
            . tests/gpu/harness/common.sh
            finished=true
            BUDGET_SECONDS=100000; echo "plenty $(cap 5)"
            BUDGET_SECONDS=1;      echo "none $(cap 5)"
            BUDGET_SECONDS=$(( $(date +%%s) - START_EPOCH + 300 + 200 )); echo "left $(cap 5)"
        """ % (REPO, self.results, self.dir.name))
        done = subprocess.run(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        self.assertEqual(done.returncode, 0, done.stdout)
        lines = dict(line.split(" ", 1) for line in done.stdout.split("\n") if line)
        self.assertEqual(lines["plenty"], "300")  # its own cap: 5 minutes
        self.assertEqual(lines["none"], "60")     # nothing is left, and a step is never given less than a minute
        self.assertLessEqual(int(lines["left"]), 200)  # what is left of the budget, less 5 minutes for the report
        self.assertGreaterEqual(int(lines["left"]), 60)

    def test_the_environment_of_a_suite_is_its_own(self):
        self.make_suite("env", """
            suite_description="what the runner sets for a suite"
            suite_run() {
                run_step env 1 bash -c 'echo "suite=$GPU_SUITE results=$GPU_RESULTS fixtures=$GPU_FIXTURES world=$WORLD"'
            }
            """)
        _, output = self.run_runner("env")
        self.assertIn("suite=env results=%s fixtures=%s world=1" % (os.path.join(self.results, "env"),
                                                                    os.path.join(self.dir.name, "fixtures", "env")), output)

    def test_what_one_suite_defines_does_not_leak_into_the_next(self):
        self.make_suite("first", """
            suite_description="defines a collect and an expectation"
            suite_run() { true; }
            suite_collect() { touch "%s/collected"; }
            suite_expect() { echo "results-x-rank0.json"; }
            """ % self.dir.name)
        self.make_suite("second", """
            suite_description="defines only suite_run"
            suite_run() { true; }
            """)
        status, output = self.run_runner("first,second")
        self.assertEqual(output.count("first/results-x-rank0.json: no results at all"), 1, output)
        self.assertNotIn("second/results-x-rank0.json", output)
        self.assertEqual(status, 1)

    @unittest.skipUnless(os.path.isdir("/proc/self"), "the runner reads /proc")
    def test_a_worker_left_behind_is_killed_and_what_runs_the_runner_is_not(self):
        """A process that runs a file of a suite is a stray after its step ended. The unit test that runs the runner has a
        suite's file on its own command line, and must survive."""
        stray = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "tests/gpu/suites/x/worker.py"])
        self.addCleanup(stray.kill)
        script = textwrap.dedent("""
            set -euo pipefail
            cd "%s"
            export WORLD=1 RESULTS="%s" SETUP_TIMING="%s/t.jsonl"
            mkdir -p "$RESULTS"
            . tests/gpu/harness/common.sh
            finished=true
            stop_stray_workers
        """ % (REPO, self.results, self.dir.name))
        # this very process has "tests/gpu/suites/..." on its command line, like a run of this test by path does
        runner = subprocess.run(["bash", "-c", script, "bash", "tests/gpu/suites/x/ancestor.py"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(runner.returncode, 0, runner.stdout)
        self.assertIn("killing a worker left behind (pid %d)" % stray.pid, runner.stdout)
        self.assertEqual(stray.wait(timeout=10), -9)

    def test_a_name_that_is_not_a_suite_is_refused_with_the_suites_that_there_are(self):
        self.unittest_suite("good", self.PASSING)
        status, output = self.run_runner("typo")
        self.assertEqual(status, 2, output)
        self.assertIn("there is no suite 'typo'", output)
        self.assertIn("good", output)

    def test_a_suite_without_suite_run_fails_the_run(self):
        self.make_suite("empty", """
            suite_description="forgot to define suite_run"
            """)
        status, output = self.run_runner("empty")
        self.assertEqual(status, 1, output)
        self.assertIn("does not define suite_run", output)

    def test_the_unit_tests_of_a_suite_run_before_it_and_fail_the_run(self):
        directory = self.make_suite("unit", """
            suite_description="has a unit test that fails"
            suite_unit_tests=(test_unit)
            suite_run() { touch "%s/ran"; }
            """ % self.dir.name)
        with open(os.path.join(directory, "test_unit.py"), "w") as handle:
            handle.write(textwrap.dedent(self.FAILING))
        # a dotted name, found on PYTHONPATH: the suite of a real run names a path from the root of the checkout
        status, output = self.run_runner("unit", env={"EXTRA_PYTHONPATH": directory})
        self.assertEqual(status, 1, output)
        self.assertTrue(os.path.exists(os.path.join(self.dir.name, "ran")), "the suite did not run after its unit tests failed")
        self.assertIn("FAILED unit.unit_tests", output)


if __name__ == "__main__":
    unittest.main()
