# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "client_configuration",
    deps = [
        "@google_cloud_cpp//:storage",
        "@google_cloud_cpp//:storage_grpc",
        "//common/exception",
        "//common/response_code",
        "//common/storage_uri",
        "//utils/env",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "client_configuration_test",
    srcs = ["client_configuration_test.cc"],
    deps = [
        ":client_configuration",
        "//utils/temp/env",
    ],
)
