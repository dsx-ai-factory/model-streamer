// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

// Shared contract: compile this header in both the SDK builder and its consumers.
// Changing the SDK compiler/ABI requires updating this contract and rebuilding the image.
#if !defined(__GNUC__) || defined(__clang__) || __GNUC__ != 9
#error "Prebuilt GCS SDK requires GCC 9. Use the matching devcontainer or rebuild the SDK with a new compiler contract."
#endif
#if __cplusplus < 201703L
#error "Prebuilt GCS SDK requires C++17 or newer."
#endif
#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 1
#error "Prebuilt GCS SDK requires libstdc++ with _GLIBCXX_USE_CXX11_ABI=1."
#endif
#ifdef _GLIBCXX_DEBUG
#error "Prebuilt GCS SDK does not support _GLIBCXX_DEBUG. Rebuild the SDK for that ABI."
#endif
