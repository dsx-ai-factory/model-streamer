# SPDX-FileCopyrightText: Copyright (c) 2025 Google LLC
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        "//gcs/client_configuration",
    ],
    ldscript = ":gcs.ldscript",
)
