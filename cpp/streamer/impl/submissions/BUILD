# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "submissions_mgr",
    deps = [
        "//common/submission",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "submissions_mgr_test",
    srcs = ["submissions_mgr_test.cc"],
    deps = [
        ":submissions_mgr",
        "//utils/random",
    ],
)
