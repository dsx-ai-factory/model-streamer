# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loads dependencies for the C++ toolchain."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_toolchain_deps():
    """load_toolchain_deps defines dependent repositories used for configuring the toolchain."""

    http_archive(
        name = "com_google_protobuf",
        sha256 = "da288bf1daa6c04d03a9051781caa52aceb9163586bff9aa6cfb12f69b9395aa",
        strip_prefix = "protobuf-27.0",
        url = "https://github.com/protocolbuffers/protobuf/releases/download/v27.0/protobuf-27.0.tar.gz",
    )

    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.1.1/rules_cc-0.1.1.tar.gz"],
        sha256 = "712d77868b3152dd618c4d64faaddefcc5965f90f5de6e6dd1d5ddcd0be82d42",
        strip_prefix = "rules_cc-0.1.1",
    )
