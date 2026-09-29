# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "s3_init",
    deps = [
        "@aws",
        "//utils/env",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "s3_init_test",
    srcs = ["s3_init_test.cc"],
    deps = [":s3_init"],
)