# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")
load("//:rules.bzl", "runai_cc_test")

cc_library(
    name = "azcache_provider",
    srcs = ["azcache_provider_loader.cc"],
    hdrs = [
        "runai_azcache_provider.h",
        "azcache_provider_loader.h",
    ],
    visibility = ["//azure:__subpackages__"],
    deps = [
        "//common/exception",
        "//utils/dylib",
        "//utils/logging",
        "//utils/env",
    ],
    linkopts = ["-ldl"],
)

# Build the test cache provider as a shared library
cc_binary(
    name = "libsimple_file_cache_test.so",
    srcs = ["simple_file_cache_test.cc"],
    linkshared = True,
    linkstatic = False,
)

runai_cc_test(
    name = "azcache_provider_test",
    srcs = ["azcache_provider_test.cc"],
    data = [":libsimple_file_cache_test.so"],
    deps = [
        ":azcache_provider",
    ],
    linkopts = ["-ldl"],
)
