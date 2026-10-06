# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "async_io_settings",
    deps = [
        "//posix_io/io_engine",
        "//streamer/impl/config/config",
        "//utils/env",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "async_io_settings_test",
    srcs = ["async_io_settings_test.cc"],
    deps = [
        ":async_io_settings",
        "//utils/temp/env",
    ],
)
