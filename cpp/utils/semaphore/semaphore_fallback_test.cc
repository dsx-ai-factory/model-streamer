/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>

extern "C" void* __real_dlsym(void* handle, const char* name);

extern "C" void* __wrap_dlsym(void* handle, const char* name)
{
    return std::strcmp(name, "sem_clockwait") == 0 ? nullptr : __real_dlsym(handle, name);
}
