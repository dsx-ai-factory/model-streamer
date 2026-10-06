# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "config",
    deps = [
        "//posix_io/alignment",
        "//streamer/impl/config/fs_queue_depth",
        "//utils/env",
        "//utils/logging",
        "//common/s3_wrapper",
    ]
)

runai_cc_test(
    name = "config_test",
    srcs = ["config_test.cc"],
    deps = [":config",
            "//utils/random",
            "//utils/temp/env",
    ],
)
