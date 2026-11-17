# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A C/C++ library's headers without its code, for linking a system copy of the library instead."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _headers_only_impl(ctx):
    # The compilation context alone: headers, include paths and defines. No linking context, so
    # nothing of the dependency's own archive reaches the link.
    return [CcInfo(compilation_context = ctx.attr.dep[CcInfo].compilation_context)]

headers_only = rule(
    implementation = _headers_only_impl,
    attrs = {
        "dep": attr.label(
            mandatory = True,
            providers = [CcInfo],
        ),
    },
    doc = "Forwards the headers of `dep`, but not its code.",
)
