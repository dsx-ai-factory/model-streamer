# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "x86_64",
    constraint_values = ["@platforms//cpu:x86_64"],
)

config_setting(
    name = "aarch64",
    constraint_values = ["@platforms//cpu:aarch64"],
)

alias(
    name = "storage",
    actual = select({
        ":x86_64": "//x86_64:sdk",
        ":aarch64": "//aarch64:sdk",
    }),
)

alias(
    name = "storage_grpc",
    actual = ":storage",
)
