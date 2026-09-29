# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "fd",
    deps = [
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "fd_test",
    srcs = ["fd_test.cc"],
    deps = [
        ":fd",
        "//utils/random",
        "//utils/temp/file",
    ],
)
