# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "object_storage",
    deps = [
        "//common/response",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "list_files_impl_test",
    srcs = ["list_files_impl_test.cc"],
    deps = [
        ":object_storage",
        "//common/response_code",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "object_storage_test",
    srcs = ["object_storage_test.cc"],
    deps = [
        ":object_storage",
        "//utils/temp/env",
    ],
)