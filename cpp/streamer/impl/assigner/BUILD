# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "assigner",
    deps = [
        "//streamer/impl/assigner/file_read_task",
        "//streamer/impl/request",
        "//common/exception",
        "//common/storage_uri",
        "//streamer/impl/config/config",
        "//utils/logging"
    ],
)

runai_cc_test(
    name = "assigner_test",
    srcs = ["assigner_test.cc"],
    deps = [
        ":assigner",
        "//utils/random",
        "//common/response_code",
        "//common/exception",
        "//utils/logging",
    ],
)
