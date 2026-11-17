# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load(":gcp_repo_test.bzl", "gcp_repo_test_suite")
load(":headers_only.bzl", "headers_only")

gcp_repo_test_suite(name = "gcp_repo_tests")

# liburing (the io_uring engine) and libaio (the fallback engine where io_uring is unavailable).
# Design section 5.13.
#
# By default both come from their Bazel Central Registry modules (MODULE.bazel), compiled with
# this build's toolchain and flags (-O2, --force_pic) and linked statically into libstreamer.so.
# streamer.ldscript ends "local: *", so none of their symbols is exported.
#
# --define USE_SYSTEM_LIBS=1 is the packager's path: compile against the module's headers, as the
# default does, but link the system's shared liburing.so / libaio.so.1 instead of the archive.

config_setting(
    name = "dynamic_link",
    define_values = {
        "USE_SYSTEM_LIBS": "1",
    },
)

alias(
    name = "uring",
    actual = select({
        ":dynamic_link": ":uring_system",
        "//conditions:default": "@liburing//:uring",
    }),
    visibility = ["//visibility:public"],
)

headers_only(
    name = "uring_headers",
    dep = "@liburing//:uring",
)

cc_library(
    name = "uring_system",
    linkopts = ["-luring"],
    deps = [":uring_headers"],
)

alias(
    name = "aio",
    actual = select({
        ":dynamic_link": ":aio_system",
        "//conditions:default": "@libaio//:aio",
    }),
    visibility = ["//visibility:public"],
)

headers_only(
    name = "aio_headers",
    dep = "@libaio//:aio",
)

cc_library(
    name = "aio_system",
    linkopts = ["-laio"],
    deps = [":aio_headers"],
)
