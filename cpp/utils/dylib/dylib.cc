/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/dylib/dylib.h"

#include <unistd.h>

#include <regex>
#include <utility>

#include "utils/logging/logging.h"
#include "utils/scope_guard/scope_guard.h"

namespace runai::llm::streamer::utils
{

Dylib::Dylib(const std::string & name, int flag) :
    _h(Dylib::dlopen(name, flag))
{}

Dylib::~Dylib()
{
    try
    {
        if (_h != nullptr)
        {
            Dylib::dlclose(_h);
        }
    }
    catch (...)
    {}
}

void * Dylib::dlopen(const std::string & name, int flag)
{
    void * const handle = ::dlopen(name.c_str(), flag);

    if (handle == nullptr)
    {
        LOG(ERROR) << "Failed loading dylib '" << name << "' (" << ::dlerror() << ")";
        throw std::exception();
    }

    return handle;
}

void * Dylib::dlsym(void * const h, const std::string & name)
{
    void * result = ::dlsym(h, name.c_str());

    if (result == nullptr)
    {
        LOG(ERROR) << "Failed getting symbol '" << name << "' (" << ::dlerror() << ")";
        throw std::exception();
    }

    return result;
}

void Dylib::dlclose(void * const h)
{
    if (::dlclose(h) != 0)
    {
        LOG(ERROR) << "Failed closing dylib handle (" << ::dlerror() << ")";
        throw std::exception();
    }
}

} // namespace runai::llm::streamer::utils
