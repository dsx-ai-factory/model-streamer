# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "client",
    deps = [
        "//s3/client_configuration",
        "//common/backend_api/response",
        "//common/response_code",
        "//common/shared_queue",
        "//common/range",
        "//common/s3_wrapper",
        "//common/exception",
        "//common/storage_uri",
        "//common/client_mgr",
        "//common/backend_api/object_storage",
        "//utils/env",
        "//utils/fd",
    ],
)

runai_cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":client",
        "//s3/s3_init",
        "//common/exception",
        "//common/response_code",
        "//common/s3_credentials",
        "//common/backend_api/object_storage",
        "//utils/temp/file",
        "//utils/temp/env",
        "//utils/random",
    ],
)
