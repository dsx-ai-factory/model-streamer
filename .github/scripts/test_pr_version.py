# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify branch-specific PR version selection and input validation."""

import unittest
from unittest.mock import patch

import pr_version


class PRVersionTests(unittest.TestCase):
    def version(self, tags, branch="master", number=348, commit="abc1234" + "0" * 33):
        return pr_version.candidate_version(tags, branch, number, commit)

    def test_master_uses_next_minor_and_resets_patch(self):
        self.assertEqual(self.version(["0.17.2", "0.16.9"]),
                         "0.18.0.dev348+gabc1234")

    def test_release_uses_next_patch_in_target_line(self):
        self.assertEqual(self.version(["0.17.0", "0.17.2", "0.18.0", "1.0.0"],
                                      "release/0.17"), "0.17.3.dev348+gabc1234")

    def test_tags_are_sorted_numerically(self):
        self.assertEqual(self.version(["0.9.9", "0.10.2", "0.10.10"], "release/0.10"),
                         "0.10.11.dev348+gabc1234")

    def test_prerelease_and_noncanonical_tags_are_ignored(self):
        self.assertEqual(self.version(["0.17.2", "1.0.0rc1", "v2.0.0", "03.0.0"]),
                         "0.18.0.dev348+gabc1234")

    def test_missing_matching_stable_tag_fails(self):
        for tags, branch in (([], "master"), (["0.18.0"], "release/0.17")):
            with self.subTest(branch=branch):
                with self.assertRaisesRegex(ValueError, "No stable release tags"):
                    self.version(tags, branch)

    def test_unsupported_branches_fail(self):
        for branch in ("main", "release/0.17.2", "release/00.17"):
            with self.subTest(branch=branch):
                with self.assertRaisesRegex(ValueError, "Unsupported PR target"):
                    self.version(["0.17.2"], branch)

    def test_invalid_pr_number_and_commit_fail(self):
        for number, commit in ((0, "a" * 40), (348, "abc1234"), (348, "x" * 40)):
            with self.subTest(number=number, commit=commit):
                with self.assertRaises(ValueError):
                    self.version(["0.17.2"], number=number, commit=commit)

    def test_cli_only_reads_tags_reachable_from_target_commit(self):
        base = "b" * 40
        args = ["pr_version.py", "--base-branch", "master", "--base-sha", base,
                "--commit", "a" * 40, "--pr-number", "348"]
        with patch("sys.argv", args), patch("builtins.print") as output:
            with patch.object(pr_version.subprocess, "check_output", return_value="0.17.2\n") as git:
                pr_version.main()
        git.assert_called_once_with(["git", "tag", "--merged", base], text=True)
        output.assert_called_once_with("PACKAGE_VERSION=0.18.0.dev348+gaaaaaaa")


if __name__ == "__main__":
    unittest.main()
