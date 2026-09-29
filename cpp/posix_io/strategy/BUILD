# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "strategy",
    deps = [
        "//common/exception",
        "//common/response_code",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "strategy_test",
    srcs = ["strategy_test.cc"],
    deps = [
        ":strategy",
    ],
)
