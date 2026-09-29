# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def runai_cc_test_dependencies():
    http_archive(
        name = "com_google_googletest",
        urls = [
            "https://mirror.bazel.build/github.com/google/googletest/archive/9816b96a6ddc0430671693df90192bbee57108b6.zip",
            "https://github.com/google/googletest/archive/9816b96a6ddc0430671693df90192bbee57108b6.zip",
        ],
        sha256 = "9cbca84c4256bed17df2c8f4d00c912c19d247c11c9ba6647cd6dd5b5c996b8d",
        strip_prefix = "googletest-9816b96a6ddc0430671693df90192bbee57108b6",
    )

def _runai_cc_binary(rule, linkopts=[], rpath_origin=False, **kwargs):
    linkopts = linkopts + ["-Wl,--gc-sections", "-Wl,--fatal-warnings"]

    if rpath_origin:
        linkopts = linkopts + ["-Wl,-rpath=$$ORIGIN,--disable-new-dtags"] # tell the linker (gold) to emit `DT_RPATH` instead of `DT_RUNPATH` (https://stackoverflow.com/a/47243544/9540328)

    rule(
        linkopts=linkopts,
        **kwargs
    )

def runai_cc_binary(**kwargs):
    _runai_cc_binary(native.cc_binary, **kwargs)

def runai_cc_test(deps=[], linkopts=[], **kwargs):
    _runai_cc_binary(
        native.cc_test,
        deps=deps + ["//cc/testing"],
        linkopts = linkopts + ["-lm"],
        **kwargs)

def runai_cc_library(copts=[], linkstatic=True, visibility=["//visibility:public"], **kwargs):
    native.cc_library(
        copts=copts + [
            "-ffunction-sections",
            "-fdata-sections",
        ],
        linkstatic=linkstatic,
        visibility=visibility,
        **kwargs
    )

def runai_cc_auto_library(hdrs=None, srcs=None, **kwargs):
    runai_cc_library(
        hdrs = hdrs if hdrs else native.glob(["**/*.h"]),
        srcs = srcs if srcs else native.glob(["**/*.cc","**/*.inc"], exclude=["**/*_test*.cc"]),
        **kwargs
    )

def _runai_portable_binary(rule, deps=[], linkopts=[], **kwargs):
    rule(
        deps = deps + [
            "//cc/portability",
        ],
        linkopts = linkopts + [
            "-Wl,--wrap=memcpy,--wrap=__fdelt_chk",

            # we link against librt.so in order to cause `clock_gettime()`
            # to be taken from glibc version 2.2.5.
            # without doing this, it will be taken from glibc 2.17.

            "-lrt",
        ],
        **kwargs
    )

def runai_cc_so(linkopts=[], deps=None, srcs=None, use_exception_override=None, **kwargs):
    srcs = srcs if srcs else []

    # A C++ target the uses exceptions should include the freeres target
    add_freeres = any([src.endswith(".cc") for src in srcs])

    # If the caller expliciltly requested to add or remove exception cleanup, respect it
    if use_exception_override != None:
       add_freeres = use_exception_override

    deps = deps if deps else []
    if add_freeres:
        # If a shared object uses exceptions, we need to register an emergency pool
        # cleanup procedure. Refer to "freeres/freeres.cc" for more details.
        deps += ["//cc/freeres"]

    runai_cc_binary(
        linkopts=linkopts + ["-shared"],
        deps=deps,
        srcs=srcs,
        **kwargs
    )

def runai_portable_so(ldscript=None, deps=[], linkopts=[], **kwargs):
    if ldscript == None:
        # a version script for the linker should be provided in order
        # for the portable shared object to export very specific symbols
        # and have its API explicitly dictated. here's a good reference:
        # https://docs.oracle.com/cd/E19120-01/open.solaris/819-0690/chapter2-14824/index.html
        fail("'ldscript' must be provided to 'runai_portable_so()'")

    _runai_portable_binary(
        rule = runai_cc_so,
        deps = deps + [ldscript],
        linkopts = linkopts + [
            "-Wl,--version-script=$(location %s)" % ldscript,
        ],
        **kwargs)
