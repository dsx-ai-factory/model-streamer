/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// GCC >= 5.* introduced a new mechanisem called emergency pool, for allocating exception on a memory constrained system.
// This pool is being used if malloc has failed, and has a size of ~72K (depending on the implementation).
// This introduces a "memory-leak" by design, as this pool is never freed. [8]
// Normally this isn't an issue, unless you are loading a shared libary which compiled statically with libstdc++.
// In such case, the pool is re-allocated every time the library has been loaded, which is really problematic.
// GCC contributers add a hook function for sanitizers, to forcefully delete this pool, called __gnu_cxx::__freeres.
// Since its an hidden function, we have to forward declare it ourselves before using it. [9]
// Note: In this implementation we register the hook using a shared object destructor.
// The memory pool will be deleted once the shared object is unloaded in an unspecified order, which means attempting to
// use it during library tear-down may cause a segmentation fault.
// The emergency pool is only used if an exception was thrown *and* `malloc` failed in an OOM system, so
// this is a calculated risk we are willing to live with.
namespace __gnu_cxx
{
    __attribute__((visibility("hidden"))) void __freeres();
} // namespace __gnu_cxx

__attribute__((destructor)) void __free_pool()
{
    __gnu_cxx::__freeres();
}
