# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_portable_so")

runai_portable_so(
    name = "libstreamergcs.so",
    srcs = [
        "gcs.h",
        "gcs.cc",
    ],
    deps = [
        "//common/client_mgr",
        "//common/range",
        "//gcs/client",
    ],
    ldscript = ":gcs.ldscript",
)
