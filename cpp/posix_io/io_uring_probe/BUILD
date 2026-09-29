# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "io_uring_probe",
    deps = [
        "//common/response_code",
        "//utils/logging",
        "@uring//:uring",
    ],
)

runai_cc_test(
    name = "io_uring_probe_test",
    srcs = ["io_uring_probe_test.cc"],
    deps = [
        ":io_uring_probe",
    ],
)
