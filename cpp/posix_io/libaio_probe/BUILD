# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "libaio_probe",
    deps = [
        "//common/response_code",
        "//utils/logging",
        "@aio//:aio",
    ],
)

runai_cc_test(
    name = "libaio_probe_test",
    srcs = ["libaio_probe_test.cc"],
    deps = [
        ":libaio_probe",
    ],
)
