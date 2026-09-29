# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@//third_party:aio.bzl", "aio_library")

package(default_visibility = ["//visibility:public"])

# libaio, vendored into the image by .devcontainer/Dockerfile and linked statically by default.
# The fallback engine, for hosts where io_uring is unavailable. Design section 5.13.

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

aio_library(name = "aio_aarch", arch = "aarch64")
aio_library(name = "aio_x86_64", arch = "x86_64")

alias(
    name = "aio",
    actual = select({
        ":target_x86_64": ":aio_x86_64",
        ":target_aarch64": ":aio_aarch",
    }),
)
