# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test", "runai_cc_binary")

runai_cc_auto_library(
    name = "logging",
)

runai_cc_binary(
    name = "logging_test_helper",
    srcs = ["logging_test_helper.cc"],
    deps = [
        ":logging",
    ],
)

runai_cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    deps = [
        ":logging",
        "//utils/fd",
        "//utils/random",
        "//utils/temp/env",
        "//utils/temp/file",
        "//utils/scope_guard",
    ],
    data = [
        ":logging_test_helper",
    ],
)
