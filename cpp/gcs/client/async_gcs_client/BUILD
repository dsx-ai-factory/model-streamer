# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library")

runai_cc_auto_library(
    # Keep SDK header paths inside Bazel's external repository for dependency checks.
    copts = ["-fno-canonical-system-headers"],
    name = "async_gcs_client",
    deps = [
        "@google_cloud_cpp//:storage",
        "@google_cloud_cpp//:storage_grpc",
        "//utils/threadpool",
    ],
)
