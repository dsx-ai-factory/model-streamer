# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of run.sh, with --dry-run: what it refuses, and the settings that it hands to the container.
No docker and no GPU: the machine's GPUs are given by GPU_LOCAL_GPU_COUNT.

    python3 -m unittest tests/gpu/test_run.py
"""

import os
import re
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "run.sh")


class Run(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="run-")
        self.addCleanup(self.dir.cleanup)
        self.wheel = os.path.join(self.dir.name, "runai_model_streamer-0.0.0-py3-none-any.whl")
        with open(self.wheel, "wb") as handle:
            handle.write(b"not a wheel")

    def run_script(self, *args, gpus="2", **env):
        # Not the environment of whatever runs this test: in the container of a run it holds that run's own GPU_SUITES, GPU_ROWS,
        # GPU_STREAMER_SPEC ..., which are what these tests set for themselves.
        inherited = {k: v for k, v in os.environ.items() if not k.startswith("GPU_") and k != "CUDA_VISIBLE_DEVICES"}
        environment = dict(inherited, GPU_LOCAL_GPU_COUNT=gpus, TMPDIR=self.dir.name)
        environment.update(env)
        done = subprocess.run(["bash", SCRIPT, "--dry-run", *args], env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True)
        return done.returncode, done.stdout

    def settings(self, output):
        """The variables that the dry run prints, as a dict."""
        found = {}
        for line in output.splitlines():
            if line.startswith("  ") and "=" in line:
                name, _, value = line.strip().partition("=")
                found[name] = value
        return found

    def test_a_wheel_on_two_gpus_is_the_ci_setup(self):
        status, output = self.run_script("--wheel", self.wheel, "--matrix", "full", "--seed", "7")
        self.assertEqual(status, 0, output)
        found = self.settings(output)
        self.assertEqual((found["GPU_WORLD"], found["CUDA_VISIBLE_DEVICES"]), ("2", "0,1"))
        self.assertEqual((found["GPU_MATRIX"], found["GPU_SEED"], found["GPU_SELF_CHECK"]), ("full", "7", "false"))
        self.assertNotIn("GPU_STREAMER_SPEC", found)
        self.assertEqual(found["GPU_SUITES"], "streaming")
        self.assertNotIn("one GPU:", output)
        self.assertNotIn("services of the suites", output)

    def test_the_wheel_is_copied_alone_into_the_directory_that_is_mounted(self):
        status, output = self.run_script("--wheel", self.wheel)
        self.assertEqual(status, 0, output)
        # the directory is removed when the script ends, so what it held is in the text: the wheel, by its name
        self.assertIn("the wheel runai_model_streamer-0.0.0-py3-none-any.whl", output)
        self.assertIn("GPU_WHEEL_HOST_DIR=", output)

    def test_one_gpu_says_that_what_needs_two_is_not_run(self):
        status, output = self.run_script("--wheel", self.wheel, "--gpus", "1")
        self.assertEqual(status, 0, output)
        found = self.settings(output)
        self.assertEqual((found["GPU_WORLD"], found["CUDA_VISIBLE_DEVICES"]), ("1", "0"))
        self.assertIn("one GPU: what needs two is not run", output)

    def test_every_suite_that_is_asked_for_is_a_profile_of_its_own_name(self):
        status, output = self.run_script("--wheel", self.wheel, "--suites", "streaming,object_storage")
        self.assertEqual(status, 0, output)
        self.assertEqual(self.settings(output)["GPU_SUITES"], "streaming,object_storage")
        self.assertIn("--profile streaming --profile object_storage run --rm -T tests", output)
        status, output = self.run_script("--wheel", self.wheel)
        self.assertIn("--profile streaming run --rm -T tests", output)
        self.assertNotIn("object_storage", output.replace("GPU_SUITES", ""))

    def test_a_wheel_directory_is_mounted_as_it_is(self):
        directory = os.path.join(self.dir.name, "downloaded")
        os.makedirs(directory)
        status, output = self.run_script("--wheel-dir", directory)
        self.assertEqual(status, 0, output)
        self.assertEqual(os.path.realpath(self.settings(output)["GPU_WHEEL_HOST_DIR"]), os.path.realpath(directory))
        self.assertIn("the wheels in", output)
        status, output = self.run_script("--wheel-dir", os.path.join(self.dir.name, "missing"))
        self.assertEqual(status, 2)
        self.assertIn("is not a directory", output)

    def test_the_streamer_comes_from_one_of_wheels_a_directory_or_a_specifier(self):
        directory = os.path.join(self.dir.name, "d")
        os.makedirs(directory)
        status, output = self.run_script("--wheel", self.wheel, "--wheel-dir", directory)
        self.assertEqual((status, "not more" in output), (2, True), output)
        status, output = self.run_script("--wheel-dir", directory, "--streamer-spec", "runai-model-streamer")
        self.assertEqual((status, "not more" in output), (2, True), output)

    def test_the_compose_project_is_named_for_the_cleanup_that_ci_does(self):
        status, output = self.run_script("--wheel", self.wheel, "--project", "gpu-tests-newest-1-1")
        self.assertEqual(status, 0, output)
        self.assertIn("docker compose -p gpu-tests-newest-1-1 -f", output)
        status, output = self.run_script("--wheel", self.wheel)
        self.assertNotIn(" -p ", output)

    def test_no_chown_is_an_option_of_ci_and_changes_nothing_that_is_handed_on(self):
        status, output = self.run_script("--wheel", self.wheel, "--no-chown", "--gpus", "2")
        self.assertEqual(status, 0, output)

    def test_a_number_of_gpus_that_ci_gives_is_not_adapted_to_the_machine(self):
        """CI gives --gpus 2 explicitly, so that a runner with one GPU fails the job instead of running a subset."""
        status, output = self.run_script("--wheel", self.wheel, "--gpus", "2", gpus="1")
        self.assertEqual(status, 2)
        self.assertIn("the machine has 1", output)

    def test_a_suite_that_does_not_exist_is_refused_with_the_ones_that_do(self):
        status, output = self.run_script("--wheel", self.wheel, "--suites", "streaming,typo")
        self.assertEqual(status, 2)
        self.assertIn("there is no suite 'typo'", output)
        self.assertIn("streaming", output)

    def test_more_than_one_wheel_goes_into_the_directory_that_is_mounted(self):
        plugin = os.path.join(self.dir.name, "runai_model_streamer_s3-0.0.0-py3-none-any.whl")
        with open(plugin, "wb") as handle:
            handle.write(b"x")
        status, output = self.run_script("--wheel", self.wheel, "--wheel", plugin)
        self.assertEqual(status, 0, output)
        self.assertIn("the wheel runai_model_streamer-0.0.0-py3-none-any.whl, the wheel "
                      "runai_model_streamer_s3-0.0.0-py3-none-any.whl", output)

    def test_another_environment_is_another_compose_file(self):
        other = os.path.join(self.dir.name, "other.yaml")
        with open(other, "w") as handle:
            handle.write("services:\n  tests:\n    image: x\n")
        status, output = self.run_script("--wheel", self.wheel, "--compose-file", other)
        self.assertEqual(status, 0, output)
        self.assertIn("docker compose -f %s --profile streaming run --rm -T tests" % other, output)
        status, output = self.run_script("--wheel", self.wheel, "--compose-file", os.path.join(self.dir.name, "none.yaml"))
        self.assertEqual(status, 2)
        self.assertIn("was not found", output)

    def test_the_container_has_every_variable_that_run_sh_hands_on(self):
        """run.sh hands its options over as GPU_* variables, which reach the container only if compose.yaml lists them (or
        uses them itself, in an image or a mount): one that is not listed is an option that does nothing."""
        with open(os.path.join(HERE, "compose.yaml")) as handle:
            compose = handle.read()
        listed = set(re.findall(r"^\s+- ([A-Z][A-Z0-9_]*)(?:=.*)?$", compose, flags=re.M))
        used = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)[:?-]", compose))
        handed = set()
        for args in (("--wheel", self.wheel, "--rows", "x", "--self-check", "--torch-spec", "t", "--numpy-spec", "n",
                      "--s3-spec", "s", "--image", "i", "--seed", "1"),
                     ("--streamer-spec", "runai-model-streamer")):
            status, output = self.run_script(*args)
            self.assertEqual(status, 0, output)
            handed.update(self.settings(output))
        handed -= {"GPU_LOCAL_GPU_COUNT"}  # of the tests of run.sh
        missing = sorted(name for name in handed if name not in listed and name not in used)
        self.assertEqual(missing, [], "run.sh hands these on, and compose.yaml does not pass them to the container")

    def test_a_variable_of_the_run_that_runs_these_tests_does_not_change_them(self):
        status, output = self.run_script("--wheel", self.wheel, GPU_SUITES="object_storage", GPU_ROWS="x",
                                         GPU_STREAMER_SPEC="y", GPU_COMPOSE_FILE="/nonexistent.yaml")
        found = self.settings(output)
        self.assertEqual((status, found["GPU_SUITES"]), (0, "streaming"), output)
        self.assertNotIn("GPU_ROWS", found)
        self.assertNotIn("GPU_STREAMER_SPEC", found)

    def test_an_exported_variable_does_not_override_an_option_that_was_not_given(self):
        """run_script puts these in the environment that run.sh starts with, as a developer's shell would: run.sh must not hand
        them on, or --wheel would be installed over by a streamer spec exported days ago."""
        status, output = self.run_script("--wheel", self.wheel, GPU_STREAMER_SPEC="runai-model-streamer==0.1", GPU_ROWS="zz",
                                         GPU_S3_SPEC="s", GPU_TORCH_SPEC="t", GPU_NUMPY_SPEC="n", GPU_TEST_IMAGE="i")
        self.assertEqual(status, 0, output)
        found = self.settings(output)
        for name in ("GPU_STREAMER_SPEC", "GPU_ROWS", "GPU_S3_SPEC", "GPU_TORCH_SPEC", "GPU_NUMPY_SPEC", "GPU_TEST_IMAGE"):
            self.assertNotIn(name, found, name)

    def test_variables_that_have_no_option_pass_through(self):
        status, output = self.run_script("--wheel", self.wheel, GPU_REAL_SMOKE="false", GPU_BUDGET_MINUTES="120")
        found = self.settings(output)
        self.assertEqual((found["GPU_REAL_SMOKE"], found["GPU_BUDGET_MINUTES"]), ("false", "120"))

    def test_help_ends_with_the_options_and_not_with_code(self):
        done = subprocess.run(["bash", SCRIPT, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        self.assertEqual(done.returncode, 0)
        self.assertIn("--wheel PATH", done.stdout)
        self.assertNotIn("set -euo pipefail", done.stdout)
        self.assertNotIn("here=", done.stdout)

    def test_the_s3_plugin_specifier_is_handed_on(self):
        status, output = self.run_script("--streamer-spec", "runai-model-streamer==0.17.0", "--s3-spec",
                                         "runai-model-streamer-s3==0.17.0")
        self.assertEqual(status, 0, output)
        self.assertEqual(self.settings(output)["GPU_S3_SPEC"], "runai-model-streamer-s3==0.17.0")

    def test_the_default_is_the_gpus_that_the_machine_has_up_to_two(self):
        for available, expected in (("1", "1"), ("2", "2"), ("8", "2")):
            status, output = self.run_script("--wheel", self.wheel, gpus=available)
            self.assertEqual(status, 0, output)
            self.assertEqual(self.settings(output)["GPU_WORLD"], expected, available)

    def test_more_gpus_than_the_machine_has_is_refused(self):
        status, output = self.run_script("--wheel", self.wheel, "--gpus", "2", gpus="1")
        self.assertEqual(status, 2)
        self.assertIn("the machine has 1", output)

    def test_no_gpu_is_refused(self):
        status, output = self.run_script("--wheel", self.wheel, gpus="0")
        self.assertEqual(status, 2)
        self.assertIn("no GPU", output)

    def test_a_requirement_specifier_instead_of_a_wheel(self):
        status, output = self.run_script("--streamer-spec", "runai-model-streamer==0.17.0")
        self.assertEqual(status, 0, output)
        self.assertEqual(self.settings(output)["GPU_STREAMER_SPEC"], "runai-model-streamer==0.17.0")

    def test_the_streamer_comes_from_exactly_one_place(self):
        status, output = self.run_script("--wheel", self.wheel, "--streamer-spec", "runai-model-streamer")
        self.assertEqual((status, "not more" in output), (2, True), output)
        status, output = self.run_script()
        self.assertEqual((status, "the streamer to test" in output), (2, True), output)

    def test_a_wheel_that_is_not_one_file_is_refused(self):
        status, output = self.run_script("--wheel", os.path.join(self.dir.name, "missing.whl"))
        self.assertEqual(status, 2)
        self.assertIn("exactly one file", output)
        other = os.path.join(self.dir.name, "runai_model_streamer-0.0.1-py3-none-any.whl")
        with open(other, "wb") as handle:
            handle.write(b"x")
        status, output = self.run_script("--wheel", os.path.join(self.dir.name, "runai_model_streamer-*.whl"))
        self.assertEqual(status, 2)
        self.assertIn("matches 2", output)

    def test_a_file_that_is_not_a_wheel_is_refused(self):
        other = os.path.join(self.dir.name, "streamer.tar.gz")
        with open(other, "wb") as handle:
            handle.write(b"x")
        status, output = self.run_script("--wheel", other)
        self.assertEqual(status, 2)
        self.assertIn(".whl", output)

    def test_the_options_are_handed_on(self):
        status, output = self.run_script("--wheel", self.wheel, "--rows", "mixed.solo,many_files", "--torch-spec",
                                         "torch==2.0.0", "--numpy-spec", "numpy<2", "--self-check", "--image", "x/y:1")
        self.assertEqual(status, 0, output)
        found = self.settings(output)
        self.assertEqual(found["GPU_ROWS"], "mixed.solo,many_files")
        self.assertEqual((found["GPU_TORCH_SPEC"], found["GPU_NUMPY_SPEC"]), ("torch==2.0.0", "numpy<2"))
        self.assertEqual((found["GPU_SELF_CHECK"], found["GPU_TEST_IMAGE"]), ("true", "x/y:1"))

    def test_an_unknown_matrix_or_option_is_refused(self):
        self.assertEqual(self.run_script("--wheel", self.wheel, "--matrix", "everything")[0], 2)
        self.assertEqual(self.run_script("--wheel", self.wheel, "--nonsense")[0], 2)
        self.assertEqual(self.run_script("--wheel", self.wheel, "--gpus", "3")[0], 2)


if __name__ == "__main__":
    unittest.main()
