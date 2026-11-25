# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "response_code",
    deps = [
        "//streamer/api",
        "//utils/logging",
    ]
)

runai_cc_test(
    name = "response_code_test",
    srcs = ["response_code_test.cc"],
    deps = [
        ":response_code",
        "//utils/random",
    ]
)
