# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Utility functions used to setup the C++ toolchain."""

def get_target_triplet(os, arch):
    return arch + "-" + os

def native_tools():
    return {
        tool: "/usr/bin/" + tool
        for tool in ["ld", "gcc", "g++", "ar", "cpp", "gcov", "nm", "objdump", "strip"]
    }
