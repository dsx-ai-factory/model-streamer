# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "fs_queue_depth",
    deps = [
        "//common/exception",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "fs_queue_depth_test",
    srcs = ["fs_queue_depth_test.cc"],
    deps = [
        ":fs_queue_depth",
        "//utils/random",
    ],
)
