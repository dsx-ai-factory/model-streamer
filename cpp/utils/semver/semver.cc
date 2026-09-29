/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/semver/semver.h"

#include <unistd.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::utils
{

Semver::Semver(unsigned short major, unsigned short minor):
    Semver(major, minor, 0)
{}

Semver::Semver(unsigned short major, unsigned short minor, unsigned short patch) :
    major(major),
    minor(minor),
    patch(patch)
{}

Semver::Semver(const char * str) : Semver(std::string(str))
{}

Semver::Semver(const std::string & str)
{
    // Extract numeric version part (e.g., "2.31" from "glibc 2.31")
    size_t pos = 0;
    while (pos < str.size() && !std::isdigit(str[pos]))
    {
        pos++;
    }

    ASSERT(pos < str.size()) << "No GLIBC version number found in string: " << str;

    auto start_pos = pos;
    while (pos < str.size() && (std::isdigit(str[pos]) || str[pos] == '.'))
    {
        pos++;
    }

    std::string nstr = str.substr(start_pos, pos - start_pos);

    std::vector<std::string> split;
    std::stringstream stream(nstr);

    std::string element;

    while (std::getline(stream, element, '.'))
    {
        split.push_back(element);
    }

    ASSERT(split.size() > 0 && split.size() <= 3) << "Malformed version string (" << nstr << ")";

    major = this->stoul(split.at(0));
    minor = split.size() > 1 ? this->stoul(split.at(1)) : 0;
    patch = split.size() > 2 ? this->stoul(split.at(2)) : 0;
}

unsigned long Semver::stoul(const std::string & str)
{
    std::string::size_type idx;
    const auto value = std::stoul(str, /* out */ &idx);

    ASSERT(idx == str.size()) << "Failed parsing string '" << str << "' as `unsigned long`";

    return value;
}

std::ostream & operator<<(std::ostream & os, const Semver & version)
{
    return os << version.major << '.' << version.minor << '.' << version.patch;
}

bool operator==(const Semver & lhs, const Semver & rhs)
{
    return \
        lhs.major == rhs.major &&
        lhs.minor == rhs.minor &&
        lhs.patch == rhs.patch;
}

bool operator!=(const Semver & lhs, const Semver & rhs)
{
    return !(lhs == rhs);
}

bool operator>(const Semver & lhs, const Semver & rhs)
{
    if (lhs.major != rhs.major)
    {
        return lhs.major > rhs.major;
    }

    if (lhs.minor != rhs.minor)
    {
        return lhs.minor > rhs.minor;
    }

    return lhs.patch > rhs.patch;
}

bool operator>=(const Semver & lhs, const Semver & rhs)
{
    return (lhs == rhs) || (lhs > rhs);
}

bool operator<=(const Semver & lhs, const Semver & rhs)
{
    return !(lhs > rhs);
}

Semver get_glibc_version()
{
    char version[256];
    confstr(_CS_GNU_LIBC_VERSION, version, sizeof(version));
    LOG(DEBUG) << "GLIBC version is " << version;
    std::string str(version);
    LOG(DEBUG) << "Semver = " << Semver(str);
    return Semver(str);
}

} // namespace runai::llm::streamer::utils
