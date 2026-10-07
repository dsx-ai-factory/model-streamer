# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "engine_factory",
    deps = [
        "//common/exception",
        "//posix_io/io_engine",
        "//posix_io/io_uring_engine",
        "//posix_io/io_uring_probe",
        "//posix_io/libaio_engine",
        "//posix_io/libaio_probe",
        "//posix_io/strategy",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "engine_factory_test",
    srcs = ["engine_factory_test.cc"],
    deps = [
        ":engine_factory",
        "//posix_io/libaio_probe",
    ],
)
