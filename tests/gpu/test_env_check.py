# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the environment check of the newest-torch leg. No network: PyPI's answer is made up here.

    python3 -m unittest tests/gpu/test_env_check.py
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
import env_check  # noqa: E402


def pypi(*releases):
    """releases: (version, requires_python) or (version, requires_python, yanked)."""
    return {"releases": {r[0]: [{"requires_python": r[1], "yanked": len(r) > 2 and r[2]}] for r in releases}}


PY310 = (3, 10, 12)


class PythonAllows(unittest.TestCase):
    def test_lower_and_upper_bounds(self):
        self.assertTrue(env_check.python_allows(">=3.10", PY310))
        self.assertFalse(env_check.python_allows(">=3.11", PY310))
        self.assertTrue(env_check.python_allows(">=3.10,<3.14", PY310))
        self.assertFalse(env_check.python_allows(">=3.8,<3.10", PY310))
        self.assertTrue(env_check.python_allows(">3.9", PY310))
        self.assertTrue(env_check.python_allows("<=3.11", PY310))

    def test_wildcards_and_exclusions(self):
        self.assertTrue(env_check.python_allows("!=3.9.*", PY310))
        self.assertFalse(env_check.python_allows("!=3.10.*", PY310))
        self.assertTrue(env_check.python_allows("==3.10.*", PY310))
        self.assertTrue(env_check.python_allows("~=3.8", PY310))
        self.assertFalse(env_check.python_allows("~=3.11", PY310))

    def test_nothing_and_what_is_not_understood_allow(self):
        self.assertTrue(env_check.python_allows("", PY310))
        self.assertTrue(env_check.python_allows(None, PY310))
        self.assertTrue(env_check.python_allows(">=3.10, !!!", PY310))


class VersionTuple(unittest.TestCase):
    def test_only_plain_releases_have_a_version(self):
        self.assertEqual(env_check.version_tuple("2.14.1"), (2, 14, 1))
        for text in ("2.15.0rc1", "2.14.1+cu128", "2.15.0.dev20260101", "2.15.0a1"):
            self.assertIsNone(env_check.version_tuple(text), text)

    def test_versions_sort_as_numbers(self):
        self.assertGreater(env_check.version_tuple("2.10.0"), env_check.version_tuple("2.9.1"))


class Assess(unittest.TestCase):
    def test_the_newest_torch_installed_is_fine(self):
        data = pypi(("2.13.0", ">=3.10"), ("2.14.1", ">=3.10"))
        limited, lines = env_check.assess("2.14.1+cu128", data, PY310)
        self.assertFalse(limited)
        self.assertIn("OK", lines[-1])
        self.assertNotIn("still run", "\n".join(lines))

    def test_a_newest_torch_that_needs_a_newer_python_is_flagged_with_the_reason(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0", ">=3.11"))
        limited, lines = env_check.assess("2.14.1", data, PY310)
        self.assertTrue(limited)
        text = "\n".join(lines)
        self.assertIn("2.15.0", text)
        self.assertIn("Python >=3.11", text)
        self.assertIn("Python 3.10.12", text)
        self.assertIn("Change the base container", text)
        # the reader is told that the tests still run, and that failing tests need a look whatever the environment is
        self.assertIn("still run, with torch 2.14.1", lines[-1])
        self.assertIn("separate problem from the test results", lines[-1])
        self.assertIn("looked into regardless", lines[-1])

    def test_an_older_torch_that_the_python_could_have_upgraded_is_flagged_differently(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0", ">=3.10"))
        limited, lines = env_check.assess("2.14.1", data, PY310)
        self.assertTrue(limited)
        self.assertIn("held torch back", lines[-2])
        self.assertNotIn("Change the base container", lines[-2])
        self.assertIn("still run", lines[-1])

    def test_pre_releases_and_yanked_releases_do_not_count_as_the_newest(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0rc1", ">=3.11"), ("2.15.0", ">=3.11", True))
        limited, _ = env_check.assess("2.14.1", data, PY310)
        self.assertFalse(limited)

    def test_a_newer_python_can_use_the_newest(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0", ">=3.11"))
        limited, _ = env_check.assess("2.15.0", data, (3, 12, 3))
        self.assertFalse(limited)


class NewestTorch(unittest.TestCase):
    """What the check does with the answer of PyPI, however it looks (the network call is replaced)."""

    def ask(self, answer, installed="2.14.1"):
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(answer if isinstance(answer, bytes) else json.dumps(answer).encode())
        response.__exit__.return_value = False
        with mock.patch.object(env_check.urllib.request, "urlopen", return_value=response), \
                mock.patch.object(env_check.importlib.metadata, "version", return_value=installed), \
                mock.patch.object(env_check.sys, "version_info", (3, 10, 12, "final", 0)):
            return env_check.newest_torch()

    def test_a_good_answer_is_assessed(self):
        limited, lines = self.ask(pypi(("2.14.1", ">=3.10"), ("2.15.0", ">=99.0")))
        self.assertTrue(limited)
        self.assertIn("ERROR", "\n".join(lines))

    def test_an_answer_without_the_releases_key_is_not_verified_and_not_a_failure(self):
        limited, lines = self.ask({"info": {"version": "2.15.0"}})
        self.assertFalse(limited)
        self.assertIn("not verified", lines[0])
        self.assertIn("KeyError", lines[0])

    def test_an_answer_with_no_release_is_not_verified_and_not_a_failure(self):
        limited, lines = self.ask({"releases": {}})
        self.assertFalse(limited)
        self.assertIn("not verified", lines[0])

    def test_an_answer_of_another_shape_is_not_verified_and_not_a_failure(self):
        for answer in ({"releases": []}, {"releases": {"2.15.0": None}}, {"releases": {"2.15.0": [None]}}):
            limited, lines = self.ask(answer)
            self.assertFalse(limited, answer)
            self.assertIn("not verified", lines[0])

    def test_an_answer_that_is_not_json_is_not_verified_and_not_a_failure(self):
        limited, lines = self.ask(b"<html>bad gateway</html>")
        self.assertFalse(limited)
        self.assertIn("not verified", lines[0])

    def test_a_network_error_is_not_verified_and_not_a_failure(self):
        with mock.patch.object(env_check.urllib.request, "urlopen", side_effect=OSError("no route to host")):
            limited, lines = env_check.newest_torch()
        self.assertFalse(limited)
        self.assertIn("not verified", lines[0])
        self.assertIn("no route to host", lines[0])


class CommandLine(unittest.TestCase):
    def test_the_status_is_one_only_when_the_container_limits_torch(self):
        with mock.patch.object(env_check, "newest_torch", return_value=(True, ["ERROR: x"])), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(env_check.main(["newest-torch"]), 1)
        with mock.patch.object(env_check, "newest_torch", return_value=(False, ["not verified"])), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(env_check.main(["newest-torch"]), 0)

    def test_the_result_is_written_where_the_job_summary_reads_it(self):
        with tempfile.TemporaryDirectory(prefix="env-check-") as directory, \
                mock.patch.dict(os.environ, {"GPU_RESULTS": directory}), \
                mock.patch.object(env_check, "newest_torch", return_value=(True, ["ERROR: x", "second line"])), \
                redirect_stdout(io.StringIO()):
            env_check.main(["newest-torch"])
            with open(os.path.join(directory, "env-check.txt")) as handle:
                self.assertEqual(handle.read(), "ERROR: x\nsecond line\n")

    def test_a_wrong_usage_is_refused(self):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(env_check.main([]), 2)
            self.assertEqual(env_check.main(["something-else"]), 2)


if __name__ == "__main__":
    unittest.main()
