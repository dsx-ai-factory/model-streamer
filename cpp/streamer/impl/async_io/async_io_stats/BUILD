# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "async_io_stats",
    deps = [
        "//posix_io/strategy",
        "//common/submission",
    ],
)

runai_cc_test(
    name = "async_io_stats_test",
    srcs = ["async_io_stats_test.cc"],
    deps = [
        ":async_io_stats",
    ],
)
