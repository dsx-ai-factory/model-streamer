# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "ca_http_client_factory",
    deps = [
        "@aws",
        "//utils/env",
        "//utils/fd",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "ca_http_client_factory_test",
    srcs = ["ca_http_client_factory_test.cc"],
    deps = [
        ":ca_http_client_factory",
        "//s3/s3_init",
        "//utils/random",
        "//utils/temp/env",
        "//utils/temp/file",
    ],
)
