# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "fs_async_router",
    deps = [
        "//streamer/impl/device_io/device_writer",
        "//common/exception",
        "//posix_io/alignment",
        "//posix_io/engine_factory",
        "//posix_io/mount_capabilities",
        "//posix_io/strategy",
        "//streamer/impl/async_io/async_io_worker",
        "//streamer/impl/config/fs_queue_depth",
        "//streamer/impl/request",
        "//streamer/impl/strategy_resolver",
        "//streamer/impl/workload",
        "//utils/logging",
        "//utils/threadpool",
    ],
)

runai_cc_test(
    name = "fs_async_router_test",
    srcs = ["fs_async_router_test.cc"],
    deps = [
        ":fs_async_router",
        "//common/exception",
    ],
)
