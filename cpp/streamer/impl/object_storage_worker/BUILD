# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "object_storage_retry",
    hdrs = ["object_storage_retry.h"],
    srcs = ["object_storage_retry.cc"],
    deps = [
        "//common/backend_api/response",
    ],
)

runai_cc_auto_library(
    name = "object_storage_worker",
    hdrs = ["object_storage_worker.h"],
    srcs = ["object_storage_worker.cc"],
    deps = [
        ":object_storage_retry",
        "//utils/capacity_worker",
        "//utils/env",
        "//utils/logging",
        "//common/s3_wrapper",
        "//common/s3_credentials",
        "//common/range",
        "//common/exception",
        "//common/response_code",
        "//common/backend_api/response",
        "//streamer/impl/workload",
        "//streamer/impl/batch",
        "//streamer/impl/task",
        "//streamer/impl/config/config",
        "//streamer/impl/reader",
        "//streamer/impl/s3",
    ],
)

runai_cc_test(
    name = "object_storage_retry_test",
    srcs = ["object_storage_retry_test.cc"],
    deps = [":object_storage_retry"],
)

runai_cc_test(
    name = "object_storage_worker_test",
    srcs = ["object_storage_worker_test.cc"],
    deps = [
        ":object_storage_worker",
        "//streamer/impl/assigner",
        "//streamer/impl/batches",
        "//utils/threadpool",
        "//utils/random",
        "//utils/thread",
        "//utils/semaphore",
        "//utils/scope_guard",
        "//utils/dylib",
    ],
    data = ["//s3/s3_mock:libstreamers3.so"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../../../s3/s3_mock",
    ],
)
