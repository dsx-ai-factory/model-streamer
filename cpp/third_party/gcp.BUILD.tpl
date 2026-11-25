# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
%{SDK_TARGETS}
    }, no_match_error = "GCS SDK is not installed for this architecture. Use the devcontainer matching the target architecture."),
)

alias(
    name = "storage_grpc",
    actual = ":storage",
)
