# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "device_writer_client",
    deps = [
        "//common/device",
        "//streamer/impl/device_io/event_pool",
        "//streamer/impl/device_io/device_writer",
        "//streamer/impl/device_io/staging_pool",
        "//utils/logging",
    ],
    visibility = ["//streamer:__subpackages__"],
)

runai_cc_test(
    name = "device_writer_client_test",
    srcs = ["device_writer_client_test.cc"],
    deps = [
        ":device_writer_client",
        "//device/mock",
    ],
    env = {"RUNAI_STREAMER_LOG_TO_STDERR": "1"},
)
