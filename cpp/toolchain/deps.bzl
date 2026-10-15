# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loads dependencies for the C++ toolchain."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:gcp_repo.bzl", "load_protobuf_repo")

def load_toolchain_deps():
    """load_toolchain_deps defines dependent repositories used for configuring the toolchain."""

    # rules_cc loads protobuf's Starlark rules even for prebuilt C++ libraries.
    load_protobuf_repo()

    # These were previously supplied by the Google SDK workspace. Keep them
    # explicit so normal builds no longer need to initialize its source graph.
    http_archive(
        name = "platforms",
        urls = ["https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz"],
        sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
    )

    http_archive(
        name = "bazel_skylib",
        urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz"],
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    )

    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.1.1/rules_cc-0.1.1.tar.gz"],
        sha256 = "712d77868b3152dd618c4d64faaddefcc5965f90f5de6e6dd1d5ddcd0be82d42",
        strip_prefix = "rules_cc-0.1.1",
    )
