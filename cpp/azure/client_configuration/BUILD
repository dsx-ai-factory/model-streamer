# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "client_configuration",
    deps = [
        "//utils/env",
        "//utils/logging",
    ],
    defines = select({
        "//:azurite_testing": ["AZURITE_TESTING"],
        "//conditions:default": [],
    }),
)

runai_cc_test(
    name = "client_configuration_test",
    srcs = ["client_configuration_test.cc"],
    deps = [
        ":client_configuration",
        "//utils/temp/env",
    ],
)
