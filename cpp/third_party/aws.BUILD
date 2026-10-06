# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@//third_party:aws.bzl", "aws_library")

package(default_visibility = ["//visibility:public"])

# s3 crt dependency diagram: https://docs.aws.amazon.com/sdkref/latest/guide/common-runtime.html

config_setting(
    name = "dynamic_link",
    define_values = {
        "USE_SYSTEM_LIBS": "1",
    },
)

config_setting(
    name = "target_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "target_aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
)

aws_library(name = "aws_aarch", arch = "aarch64")
aws_library(name = "aws_x86_64", arch = "x86_64")

alias(
    name = "aws",
    actual = select({
        ":target_x86_64": ":aws_x86_64",
        ":target_aarch64": ":aws_aarch",
    }),
)
