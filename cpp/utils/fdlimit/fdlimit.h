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

struct FdLimitSetter
{
    explicit FdLimitSetter(rlim_t new_limit);
    ~FdLimitSetter();

    FdLimitSetter(const FdLimitSetter&) = delete;
    FdLimitSetter& operator=(const FdLimitSetter&) = delete;

    FdLimitSetter(FdLimitSetter&& other) noexcept;
    FdLimitSetter& operator=(FdLimitSetter&& other) noexcept;

 private:
    struct rlimit _previous;
    bool _valid = false;
};

} // namespace runai::llm::streamer::utils
