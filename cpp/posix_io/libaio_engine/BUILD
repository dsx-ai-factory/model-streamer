# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "libaio_engine",
    deps = [
        "//common/exception",
        "//posix_io/alignment",
        "//posix_io/io_engine",
        "//common/response_code",
        "//utils/logging",
        "@aio//:aio",
    ],
)

runai_cc_test(
    name = "libaio_engine_test",
    srcs = ["libaio_engine_test.cc"],
    deps = [
        ":libaio_engine",
        "//posix_io/alignment",
        "//utils/random",
        "//utils/temp/file",
    ],
)
