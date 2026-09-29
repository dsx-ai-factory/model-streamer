# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "scope_guard",
)

runai_cc_test(
    name = "scope_guard_test",
    srcs = ["scope_guard_test.cc"],
    deps = [
        ":scope_guard",
    ],
)
