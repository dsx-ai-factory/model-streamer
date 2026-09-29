# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "threadpool",
    deps = [
        "//utils/thread",
        "//utils/semaphore",
    ]
)

runai_cc_test(
    name = "threadpool_test",
    srcs = ["threadpool_test.cc"],
    deps = [
        ":threadpool",
        "//utils/random",
    ],
)
