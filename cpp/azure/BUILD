# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Google LLC
# SPDX-FileCopyrightText: Copyright (c) 2026 Hari
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
