# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "client_configuration",
    deps = [
        "@aws",
        "//common/storage_uri",
        "//common/backend_api/object_storage",
        "//utils/env",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "client_configuration_test",
    srcs = ["client_configuration_test.cc"],
    deps = [
        ":client_configuration",
        "//s3/s3_init",
        "//utils/temp/env",
    ],
)
