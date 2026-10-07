# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "async_io_worker",
    deps = [
        "//utils/env",
        "//streamer/impl/device_io/device_writer_client",
        "//utils/deque",
        "//posix_io/engine_factory",
        "//posix_io/alignment",
        "//posix_io/completion_mapper",
        "//posix_io/io_engine",
        "//posix_io/scratch_pool",
        "//posix_io/strategy",
        "//common/exception",
        "//streamer/impl/async_io/async_io_settings",
        "//streamer/impl/async_io/async_io_stats",
        "//streamer/impl/async_io/inflight_chunks",
        "//streamer/impl/batch",
        "//streamer/impl/workload",
        "//utils/capacity_worker",
        "//utils/fd",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "async_io_worker_test",
    srcs = ["async_io_worker_test.cc"],
    deps = [
        "//device/mock",
        ":async_io_worker",
        "//posix_io/alignment",
        "//posix_io/mock",
        "//streamer/impl/assigner",
        "//streamer/impl/batches",
        "//utils/random",
        "//utils/temp/env",
        "//utils/temp/file",
    ],
)
