# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "scratch_pool",
    deps = [
        "//common/exception",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "scratch_pool_test",
    srcs = ["scratch_pool_test.cc"],
    deps = [
        ":scratch_pool",
    ],
)
