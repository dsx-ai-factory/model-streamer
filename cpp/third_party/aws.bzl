# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bazel Skylib module with definitions for building AWS SDK."""

def aws_library(name, arch):
    native.cc_library(
        name = name,
        hdrs = native.glob(["%s-aws/include/**/*.*" % arch]),
        includes = [
            "%s-aws/include" % arch,
        ],
        strip_include_prefix = "%s-aws/include" % arch,
        linkopts = [
            "-lpthread",
            "-l:libaws-cpp-sdk-s3-crt.a",
            "-l:libaws-cpp-sdk-core.a",
            "-l:libaws-crt-cpp.a",
            "-l:libaws-c-mqtt.a",
            "-l:libaws-c-event-stream.a",
            "-l:libaws-c-s3.a",
            "-l:libaws-c-auth.a",
            "-l:libaws-c-http.a",
            "-l:libaws-c-io.a",
            "-l:libs2n.a",
            "-l:libaws-c-compression.a",
            "-l:libaws-c-cal.a",
            "-l:libaws-c-sdkutils.a",
            "-l:libaws-checksums.a",
            "-l:libaws-c-common.a",
            "-ldl -lm -lrt",
            "-L/opt/%s-curl/lib" % arch,
            "-L/opt/%s-ssl/lib" % arch,
            "-L/opt/%s-zlib/lib" % arch,
            "-L/opt/%s-aws/lib" % arch,
        ] + select({
            "//:dynamic_link": ["-lz", "-lssl", "-lcrypto", "-lcurl"],
            "//conditions:default": [
                "-l:libz.a",
                "-l:libcurl.a",
                "-l:libssl.a",
                "-l:libcrypto.a",
            ],
        }),
    )
