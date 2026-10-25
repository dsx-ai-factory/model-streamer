/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unistd.h>
#include <sys/resource.h>
#include <string>
#include <ostream>

namespace runai::llm::streamer::utils
{

rlim_t get_max_file_descriptors();
rlim_t get_cur_file_descriptors();

// The descriptors this process has open, from /proc/self/fd (Linux). 0 when that cannot be read.
rlim_t count_open_fds();

struct FdLimitSetter
{
    explicit FdLimitSetter(rlim_t new_limit);
    ~FdLimitSetter();

    FdLimitSetter(const FdLimitSetter&) = delete;
    FdLimitSetter& operator=(const FdLimitSetter&) = delete;

    FdLimitSetter(FdLimitSetter&& other) noexcept;
    FdLimitSetter& operator=(FdLimitSetter&& other) noexcept;

    // Raises the soft limit to at least `limit`, capped at the hard limit; never lowers it. The limit
    // restored on destruction is unchanged. Best effort: warns on failure. Returns whether it reaches `limit`.
    bool raise_to(rlim_t limit);

 private:
    struct rlimit _previous;
    bool _valid = false;
};

} // namespace runai::llm::streamer::utils
