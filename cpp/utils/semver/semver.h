/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <ostream>

namespace runai::llm::streamer::utils
{

struct Semver
{
    Semver() = default; // empty creation

    // explicit creation
    Semver(unsigned short major, unsigned short minor);
    Semver(unsigned short major, unsigned short minor, unsigned short patch);

    // string creation
    Semver(const char *);
    Semver(const std::string &);

    unsigned short major;
    unsigned short minor;
    unsigned short patch;

 private:
    unsigned long stoul(const std::string & str);
};

std::ostream & operator<<(std::ostream &, const Semver &);

bool operator==(const Semver &, const Semver &);
bool operator!=(const Semver &, const Semver &);
bool operator>(const Semver &, const Semver &);
bool operator>=(const Semver &, const Semver &);
bool operator<=(const Semver &, const Semver &);

Semver get_glibc_version();

} // namespace runai::llm::streamer::utils
