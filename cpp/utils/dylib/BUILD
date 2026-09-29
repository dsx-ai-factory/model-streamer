# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test", "runai_cc_so")

runai_cc_auto_library(
    name = "dylib",
    deps = [
        "//utils/logging",
        "//utils/scope_guard",
    ],
    linkopts = ["-ldl"],
)

runai_cc_so(
    name = "dylib_test_helper",
    srcs = ["dylib_test_helper.c"],
)

runai_cc_test(
    name = "dylib_test",
    srcs = ["dylib_test.cc"],
    deps = [
        ":dylib",
        "//utils/temp/file",
        "//utils/random",
    ],
    data = [":dylib_test_helper"],
)
