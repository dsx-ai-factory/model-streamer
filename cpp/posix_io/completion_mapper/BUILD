# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "completion_mapper",
    deps = [
        "//posix_io/alignment",
        "//posix_io/io_engine",
        "//common/response_code",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "completion_mapper_test",
    srcs = ["completion_mapper_test.cc"],
    deps = [
        ":completion_mapper",
    ],
)
