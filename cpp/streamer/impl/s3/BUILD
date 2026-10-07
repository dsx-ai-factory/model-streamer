# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library")

runai_cc_auto_library(
    name = "s3",
    deps = [
        "//common/s3_wrapper",
        "//common/exception",
        "//streamer/impl/reader",
        "//streamer/impl/config/config",
        "//utils/logging",
    ],
)
