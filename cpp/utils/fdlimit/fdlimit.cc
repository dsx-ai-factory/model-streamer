/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/fdlimit/fdlimit.h"

#include <iostream>
#include <sstream>
#include <vector>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::utils
{

rlim_t get_cur_file_descriptors()
{
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0)
    {
        return limit.rlim_cur;  // Maximum limit on file descriptors
    }

    LOG(ERROR) << "Failed to get current fd limit";
    return 0;
}

rlim_t get_max_file_descriptors()
{
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0)
    {
        return limit.rlim_max;  // Maximum limit on file descriptors
    }

    LOG(ERROR) << "Failed to get maximal fd_limit";
    return 0;
}

FdLimitSetter::FdLimitSetter(rlim_t new_limit)
{
    ASSERT(getrlimit(RLIMIT_NOFILE, &_previous) == 0) << "Failed to get fd limit";

    if (new_limit > _previous.rlim_max)
    {
        LOG(WARNING) << "Cannot set fd limit above hard limit " <<  _previous.rlim_max;
        new_limit =  _previous.rlim_max;
    }

    // Set new limit (only soft limit, keep hard limit unchanged)
    struct rlimit temp_limit = { new_limit, _previous.rlim_max };
    ASSERT(setrlimit(RLIMIT_NOFILE, &temp_limit) == 0) << "Failed to set fd limit to " << new_limit;

    LOG(DEBUG) << "Temporary Fd limit set to: " << new_limit;
    _valid = true;
}

FdLimitSetter::~FdLimitSetter()
{
    try
    {
        if (_valid)
        {
            // Restore the original limit
            if (setrlimit(RLIMIT_NOFILE, &_previous) != 0)
            {
                LOG(ERROR) << "Failed to restore fd limit to " << _previous.rlim_cur;
            }
            else
            {
                LOG(DEBUG) << "Fd limit restored to: " << _previous.rlim_cur;
            }
        }
    }
    catch(const std::exception& e)
    {
    }
}

FdLimitSetter::FdLimitSetter(FdLimitSetter&& other) noexcept:
    _previous(other._previous),
    _valid(other._valid)
{
    other._valid = false;
}

FdLimitSetter& FdLimitSetter::operator=(FdLimitSetter&& other) noexcept
{
    if (this != &other) {
        _previous = other._previous;
        _valid = other._valid;
        other._valid = false;
    }
    return *this;
}

} // namespace runai::llm::streamer::utils
