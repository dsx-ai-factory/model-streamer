# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the environment check of the newest-torch leg. No network: PyPI's answer is made up here.

    python3 -m unittest tests/gpu/test_env_check.py
"""

import os
import sys
import unittest

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

    def test_a_newest_torch_that_needs_a_newer_python_is_flagged_with_the_reason(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0", ">=3.11"))
        limited, lines = env_check.assess("2.14.1", data, PY310)
        self.assertTrue(limited)
        text = "\n".join(lines)
        self.assertIn("2.15.0", text)
        self.assertIn("Python >=3.11", text)
        self.assertIn("Python 3.10.12", text)
        self.assertIn("Change the base container", text)

    def test_an_older_torch_that_the_python_could_have_upgraded_is_flagged_differently(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0", ">=3.10"))
        limited, lines = env_check.assess("2.14.1", data, PY310)
        self.assertTrue(limited)
        self.assertIn("held torch back", lines[-1])
        self.assertNotIn("Change the base container", lines[-1])

    def test_pre_releases_and_yanked_releases_do_not_count_as_the_newest(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0rc1", ">=3.11"), ("2.15.0", ">=3.11", True))
        limited, _ = env_check.assess("2.14.1", data, PY310)
        self.assertFalse(limited)

    def test_a_newer_python_can_use_the_newest(self):
        data = pypi(("2.14.1", ">=3.10"), ("2.15.0", ">=3.11"))
        limited, _ = env_check.assess("2.15.0", data, (3, 12, 3))
        self.assertFalse(limited)


if __name__ == "__main__":
    unittest.main()
