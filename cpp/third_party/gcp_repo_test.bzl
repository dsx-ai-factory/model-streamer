# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for prebuilt SDK input validation."""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load(":gcp_repo.bzl", "sdk_build_input_matches")

def _build_input_matches_test_impl(ctx):
    env = unittest.begin(ctx)
    recorded = "# SDK flags\nbuild --compilation_mode=opt\n"
    for newline in ["\n", "\r\n", "\r"]:
        actual = recorded.replace("\n", newline)
        asserts.true(env, sdk_build_input_matches(actual, recorded))
        # Newline normalization must not hide substantive or whitespace changes.
        for before, after in [("opt", "dbg"), ("SDK flags", "changed comment"), ("build ", "build  ")]:
            asserts.false(env, sdk_build_input_matches(actual.replace(before, after), recorded))
    asserts.false(env, sdk_build_input_matches(recorded.rstrip("\n"), recorded))
    return unittest.end(env)

_build_input_matches_test = unittest.make(_build_input_matches_test_impl)

def gcp_repo_test_suite(name):
    """Test the same comparison used by the SDK repository rule."""
    unittest.suite(name, _build_input_matches_test)
