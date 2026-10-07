# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cc_library(
    name = "client",
    hdrs = [
        "client.h",
    ],
    srcs = [
        "client.cc",
    ],
    defines = select({
        "//:azurite_testing": ["AZURITE_TESTING"],
        "//conditions:default": [],
    }),
    deps = [
        "@azure//:azure",
        "//common/backend_api/response",
        "//common/backend_api/object_storage",
        "//common/client_mgr",
        "//common/storage_uri",
        "//common/shared_queue",
        "//common/range",
        "//common/exception",
        "//utils/logging",
        "//utils/env",
        "//azure/client_configuration",
        "//azure/client/async_azure_client",
    ],
    visibility = ["//visibility:public"],
)
