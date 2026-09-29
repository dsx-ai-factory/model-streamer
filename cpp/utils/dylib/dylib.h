/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlfcn.h>
#include <link.h>

#include <string>
#include <vector>

namespace runai::llm::streamer::utils
{

struct Dylib
{
    Dylib() = default; // empty creation
    Dylib(const std::string & name, int flag = RTLD_NOW);
    ~Dylib();

    Dylib(Dylib && other) = delete;
    Dylib & operator=(Dylib && other) = delete;

    Dylib(const Dylib &)                = delete;
    Dylib & operator=(const Dylib &)    = delete;

    inline operator bool() const   { return _h != nullptr; }
    inline operator bool()         { return _h != nullptr; }

    template <typename T = void*>
    T dlsym(const std::string & name) const
    {
        return reinterpret_cast<T>(Dylib::dlsym(_h, name));
    }

    template <typename T, typename U = typename std::conditional<std::is_function<T>::value, T *, T>::type>
    static U dlsym(void * const h, const std::string & name)
    {
        return reinterpret_cast<U>(Dylib::dlsym(h, name));
    }

    // // basic wrappers with error checking
    static void * dlopen(const std::string & name, int flag = RTLD_NOW);
    static void * dlsym(void * const h, const std::string & name);
    static void dlclose(void * const h);

 protected:
    void * _h = nullptr;
};

} // namespace runai::llm::streamer::utils
