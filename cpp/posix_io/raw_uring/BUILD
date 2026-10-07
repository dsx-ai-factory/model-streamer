# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library")

# TEST ONLY, and its own package so that no production glob can reach it - the tests of the probe and
# of the engine both ask the kernel this question, and must ask it the same way.
runai_cc_auto_library(
    name = "raw_uring",
    deps = [
        "@uring//:uring",
    ],
    testonly = True,
)
