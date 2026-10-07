# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "io_uring_engine",
    deps = [
        "//common/exception",
        "//posix_io/alignment",
        "//posix_io/io_engine",
        "//posix_io/io_uring_probe",
        "//common/response_code",
        "//utils/logging",
        "@uring//:uring",
    ],
)

runai_cc_test(
    name = "io_uring_engine_test",
    srcs = ["io_uring_engine_test.cc"],
    deps = [
        ":io_uring_engine",
        "//posix_io/io_uring_probe",
        "//posix_io/alignment",
        "//posix_io/raw_uring",
        "//utils/random",
        "//utils/scope_guard",
        "//utils/temp/env",
        "//utils/temp/file",
    ],
)
