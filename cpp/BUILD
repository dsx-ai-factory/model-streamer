# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Config setting for Azurite testing support
# Use with: bazel build --config=azurite //azure:libstreamerazure.so
config_setting(
    name = "azurite_testing",
    define_values = {
        "azurite_testing": "true",
    },
    visibility = ["//visibility:public"],
)
