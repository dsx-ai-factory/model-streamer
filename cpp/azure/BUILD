# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_portable_so")

runai_portable_so(
    name = "libstreamerazure.so",
    srcs = [
        "azure.h",
        "azure.cc",
    ],
    deps = [
        "//common/client_mgr",
        "//common/range",
        "//common/exception",
        "//utils/semver",
        "//azure/client",
    ],
    ldscript = ":azure.ldscript",
)
