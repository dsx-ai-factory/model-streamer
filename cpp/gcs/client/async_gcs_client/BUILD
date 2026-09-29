# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library")

runai_cc_auto_library(
    name = "async_gcs_client",
    deps = [
        "@google_cloud_cpp//:storage",
        "@google_cloud_cpp//:storage_grpc",
        "//utils/threadpool",
    ],
)
