# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "alignment",
    deps = [
        "//posix_io/io_engine",
        "//utils/env",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "alignment_test",
    srcs = ["alignment_test.cc"],
    deps = [
        ":alignment",
        "//utils/temp/env",
        "//common/exception",
    ],
)
