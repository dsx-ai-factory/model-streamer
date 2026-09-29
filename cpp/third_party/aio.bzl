# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bazel Skylib module with definitions for linking libaio."""

def aio_library(name, arch):
    native.cc_library(
        name = name,
        hdrs = native.glob(["%s-aio/include/**/*.h" % arch]),
        includes = [
            "%s-aio/include" % arch,
        ],
        strip_include_prefix = "%s-aio/include" % arch,
        linkopts = select({
            # The packager's path: their libaio.so.1, found on the system link path. Deliberately
            # WITHOUT -L/opt, unlike the default branch - that directory holds libaio.a and no
            # shared object, so ld would find the archive and link it statically, silently ignoring
            # the flag.
            "//:dynamic_link": ["-laio"],
            "//conditions:default": [
                "-L/opt/%s-aio/lib" % arch,
                "-l:libaio.a",
            ],
        }),
    )
