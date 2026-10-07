# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "device_writer",
    deps = [
        "//common/device",
        "//device",
        "//device/owned",
        "//streamer/impl/device_io/staging_pool",
        "//streamer/impl/device_io/stream_waiter",
        "//utils/logging",
    ],
    visibility = ["//streamer:__subpackages__"],
)

runai_cc_test(
    name = "device_writer_test",
    srcs = ["device_writer_test.cc"],
    deps = [
        ":device_writer",
        "//device/mock",
    ],
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
