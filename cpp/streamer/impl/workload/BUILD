# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "workload",
    deps = [
        "//utils/logging",
        "//common/response_code",
        "//streamer/impl/batch",
    ],
)

runai_cc_test(
    name = "workload_test",
    srcs = ["workload_test.cc"],
    deps = [":workload",
            "//streamer/impl/s3",
            "//streamer/impl/assigner",
            "//streamer/impl/batches",
            "//utils/random",
            "//utils/thread",
            "//utils/semaphore",
            "//utils/temp/file",],
    data = ["//s3/s3_mock:libstreamers3.so"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../../../s3/s3_mock",
    ],
)