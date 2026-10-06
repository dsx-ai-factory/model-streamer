# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "client_mgr",
    deps = [
        "//common/backend_api/object_storage",
    ]
)
runai_cc_test(
    name = "client_mgr_test",
    srcs = ["client_mgr_test.cc"],
    deps = [
        ":client_mgr",
        "//common/backend_api/object_storage",
        "//common/storage_uri",
        "//common/s3_credentials",
        "//common/s3_wrapper",
        "//utils/random",
        ],
)
