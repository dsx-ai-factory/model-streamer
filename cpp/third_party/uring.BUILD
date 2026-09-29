# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@//third_party:uring.bzl", "uring_library")

package(default_visibility = ["//visibility:public"])

# liburing, vendored into the image by .devcontainer/Dockerfile and linked statically by default.
# Design section 5.13.

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

uring_library(name = "uring_aarch", arch = "aarch64")
uring_library(name = "uring_x86_64", arch = "x86_64")

alias(
    name = "uring",
    actual = select({
        ":target_x86_64": ":uring_x86_64",
        ":target_aarch64": ":uring_aarch",
    }),
)
