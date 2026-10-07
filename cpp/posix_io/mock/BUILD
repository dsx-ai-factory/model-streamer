# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

# TEST ONLY, and its own package so that no production glob can reach it - a mock in the shipped
# library is a defect. Nothing under //streamer may depend on this.
runai_cc_auto_library(
    name = "mock",
    deps = [
        "//posix_io/io_engine",
        "//utils/logging",
    ],
    testonly = True,
)

runai_cc_test(
    name = "mock_io_engine_test",
    srcs = ["mock_io_engine_test.cc"],
    deps = [
        ":mock",
    ],
)
