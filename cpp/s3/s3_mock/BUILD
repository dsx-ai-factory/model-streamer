# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_portable_so")

runai_portable_so(
    name = "libstreamers3.so",
    srcs = ["s3_mock.h",
            "s3_mock.cc",
    ],
    deps = [
        "//common/backend_api/response",
        "//common/backend_api/object_storage",
        "//common/response_code",
        "//common/s3_credentials",
        "//common/range",
        "//utils/env",
        "//utils/random",
        "//utils/logging",
    ],
    ldscript = ":s3_mock.ldscript",
    visibility= ["//visibility:public"],
)
