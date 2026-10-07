# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_gcp_repo():
    # Fetch the Google Cloud C++ libraries.
    # NOTE: Update this version and SHA256 as needed.
    http_archive(
        name = "google_cloud_cpp",
        sha256 = "10867580483cb338e7d50920c2383698f3572cc6b4c7d072e38d5f43755cbd80",
        strip_prefix = "google-cloud-cpp-2.37.0",
        url = "https://github.com/googleapis/google-cloud-cpp/archive/v2.37.0.tar.gz",
    )
