# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bazel Skylib module with definitions for linking liburing."""

def uring_library(name, arch):
    native.cc_library(
        name = name,
        hdrs = native.glob(["%s-uring/include/**/*.h" % arch]),
        includes = [
            "%s-uring/include" % arch,
        ],
        strip_include_prefix = "%s-uring/include" % arch,
        linkopts = select({
            # The packager's path: their liburing.so, found on the system link path. Deliberately
            # WITHOUT -L/opt, unlike the default branch - that directory holds liburing.a and no
            # shared object, so ld would find the archive and link it statically, silently ignoring
            # the flag.
            "//:dynamic_link": ["-luring"],
            "//conditions:default": [
                "-L/opt/%s-uring/lib" % arch,
                "-l:liburing.a",
            ],
        }),
    )
