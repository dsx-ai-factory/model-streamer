# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library",  "runai_cc_test")

runai_cc_auto_library(
    name = "s3_wrapper",
    deps = [
        "//common/backend_api/object_storage",
        "//common/storage_uri",
        "//common/s3_credentials",
        "//common/backend_api/response",
        "//common/range",
        "//common/exception",
        "//utils/dylib",
        "//utils/env",
        "//utils/semver",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "s3_wrapper_test",
    srcs = ["s3_wrapper_test.cc"],
    deps = [
        ":s3_wrapper",
        "//s3/s3_mock:libstreamers3.so",
        "//utils/random",
        "//utils/temp/env",
    ],
    data = ["//s3/s3_mock:libstreamers3.so"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../../s3/s3_mock",
    ],
)
