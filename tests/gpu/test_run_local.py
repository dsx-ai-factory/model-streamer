# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of run_local.sh, with --dry-run: what it refuses, and the settings that it hands to the container.
No docker and no GPU: the machine's GPUs are given by GPU_LOCAL_GPU_COUNT.

    python3 -m unittest tests/gpu/test_run_local.py
"""

import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "run_local.sh")


class RunLocal(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="run-local-")
        self.addCleanup(self.dir.cleanup)
        self.wheel = os.path.join(self.dir.name, "runai_model_streamer-0.0.0-py3-none-any.whl")
        with open(self.wheel, "wb") as handle:
            handle.write(b"not a wheel")

    def run_script(self, *args, gpus="2", **env):
        environment = dict(os.environ, GPU_LOCAL_GPU_COUNT=gpus, TMPDIR=self.dir.name)
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

    def test_the_suites_that_are_asked_for_are_handed_on_with_the_services_that_they_need(self):
        status, output = self.run_script("--wheel", self.wheel, "--suites", "streaming,object_storage")
        self.assertEqual(status, 0, output)
        self.assertEqual(self.settings(output)["GPU_SUITES"], "streaming,object_storage")
        self.assertIn("services of the suites: object-storage", output)
        self.assertIn("--profile object-storage run --rm tests", output)

    def test_a_suite_with_no_service_asks_for_no_profile(self):
        status, output = self.run_script("--wheel", self.wheel, "--suites", "streaming")
        self.assertEqual(status, 0, output)
        self.assertNotIn("--profile", output)

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
        self.assertIn("docker compose -f %s run --rm tests" % other, output)
        status, output = self.run_script("--wheel", self.wheel, "--compose-file", os.path.join(self.dir.name, "none.yaml"))
        self.assertEqual(status, 2)
        self.assertIn("was not found", output)

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
        self.assertEqual((status, "not both" in output), (2, True), output)
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
