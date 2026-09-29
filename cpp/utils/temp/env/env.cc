/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/temp/env/env.h"

#include <stdlib.h>

#include <chrono>
#include <thread>
#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::utils::temp
{

static constexpr auto PRE_SETENV_DELAY = std::chrono::milliseconds{10};

Env::Env(const std::string & value) : Env(random::string(), value)
{}

Env::Env(const std::string & name, char const * const value) :
    name(name),
    value(value)
{
    // What was there before, so the destructor puts it back instead of removing it.
    if (const char * existing = ::getenv(name.c_str()))
    {
        previous_value = existing;
        had_value = true;
    }

    // We are sleeping for 10ms here in order to avoid a getenv/setenv race that occurs
    // sometimes upon launching a thread (calling getenv()) and using this method to
    // set an environment variable in our UT.
    std::this_thread::sleep_for(PRE_SETENV_DELAY);

    // OVERWRITE. Without it a variable already in the environment wins, and the test then runs under
    // a value it did not choose while its assertions describe the value it asked for. setenv returns
    // success when it declines to overwrite, so that failure is completely silent.
    PASSERT(::setenv(name.c_str(), value, 1 /* overwrite */) != -1) << "Failed setting environment variable '" << name << "' to \"" << value << "\"";
}

Env::Env(const std::string & name, const std::string & value) :
    Env(name, value.c_str())
{}

Env::Env(const std::string & name, int value) :
    Env(name, std::to_string(value))
{}

Env::Env(const std::string & name, unsigned long value) :
    Env(name, std::to_string(value))
{}

Env::Env(const std::string & name, bool value) :
    Env(name, static_cast<int>(value))
{}

Env::Env(const std::string & name, float value) :
    Env(name, std::to_string(value))
{}

Env::~Env()
{
    if (name.empty())
    {
        return;
    }

    // Put back what was there, rather than removing it. Removing would delete a value this object
    // never set - which matters when tests run in one process and a later test reads the variable.
    if (had_value)
    {
        if (::setenv(name.c_str(), previous_value.c_str(), 1 /* overwrite */) == -1)
        {
            LOG(ERROR) << "Failed restoring environment variable '" << name << "'";
        }
    }
    else if (::unsetenv(name.c_str()) == -1)
    {
        LOG(ERROR) << "Failed unsetting environment variable '" << name << "'";
    }
}

UnsetEnv::UnsetEnv(const std::string & name) :
    name(name)
{
    const char * previous = ::getenv(name.c_str());
    if (previous != nullptr)
    {
        had_value = true;
        previous_value = previous;
    }
    if (::unsetenv(name.c_str()) == -1)
    {
        LOG(ERROR) << "Failed unsetting environment variable '" << name << "'";
    }
}

UnsetEnv::~UnsetEnv()
{
    if (had_value)
    {
        if (::setenv(name.c_str(), previous_value.c_str(), 1 /* overwrite */) == -1)
        {
            LOG(ERROR) << "Failed restoring environment variable '" << name << "'";
        }
    }
    else if (::unsetenv(name.c_str()) == -1)
    {
        LOG(ERROR) << "Failed unsetting environment variable '" << name << "'";
    }
}

} // namespace runai::llm::streamer::utils::temp
