# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions used to setup the C++ toolchain."""

def __tool_path(toolchain_name, tool_name):
    return "/usr/bin/%s-%s" % (toolchain_name, tool_name)

def get_target_triplet(os, arch):
    return arch + "-" + os

def runai_crosstool_tools(toolchain_name):
    return {
        "ld":           __tool_path(toolchain_name, "ld"),
        "gcc":          __tool_path(toolchain_name, "gcc"),
        "g++":          __tool_path(toolchain_name, "g++"),
        "ar":           __tool_path(toolchain_name, "ar"),
        "cpp":          __tool_path(toolchain_name, "cpp"),
        "gcov":         __tool_path(toolchain_name, "gcov"),
        "nm":           __tool_path(toolchain_name, "nm"),
        "objdump":      __tool_path(toolchain_name, "objdump"),
        "strip":        __tool_path(toolchain_name, "strip"),
        "cc1plus":      __tool_path(toolchain_name, "cc1plus")
    }
