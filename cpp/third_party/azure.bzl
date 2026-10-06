# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bazel Skylib module with definitions for building Azure SDK."""

def azure_library(name, arch):
    native.cc_library(
        name = name,
        hdrs = native.glob(["%s-azure/include/**/*.*" % arch]),
        includes = [
            "%s-azure/include" % arch,
        ],
        strip_include_prefix = "%s-azure/include" % arch,
        linkopts = [
            "-L/opt/%s-azure/lib" % arch,
            "-Wl,--whole-archive",
            "-l:libazure-storage-blobs.a",
            "-l:libazure-storage-common.a",
            "-l:libazure-identity.a",
            "-l:libazure-core.a",
            "-Wl,--no-whole-archive",
            "-lpthread",
            "-ldl",
            "-lm",
            "-lrt",
            "-L/opt/%s-curl/lib" % arch,
            "-L/opt/%s-ssl/lib" % arch,
            "-L/opt/%s-zlib/lib" % arch,
            "-L/opt/%s-xml2/lib" % arch,
        ] + select({
            "//:dynamic_link": ["-lz", "-lssl", "-lcrypto", "-lcurl", "-lxml2"],
            "//conditions:default": [
                "-l:libcurl.a",
                "-l:libz.a",
                "-l:libssl.a",
                "-l:libcrypto.a",
                "-lxml2",  # Use shared libxml2 to avoid conflicts
            ],
        }),
    )
