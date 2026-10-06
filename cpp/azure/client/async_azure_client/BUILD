# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cc_library(
    name = "async_azure_client",
    srcs = ["async_azure_client.cc"],
    hdrs = ["async_azure_client.h"],
    visibility = ["//azure:__subpackages__"],
    deps = [
        "//azure/azcache_provider",
        "//common/response_code",
        "//utils/threadpool",
        "//utils/logging",
        "@azure//:azure",
    ],
)
