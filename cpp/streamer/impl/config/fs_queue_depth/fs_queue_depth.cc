/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/config/fs_queue_depth/fs_queue_depth.h"

#include <algorithm>
#include <limits>
#include <cctype>
#include <ostream>
#include <set>
#include <stdexcept>

#include "common/exception/exception.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

namespace
{

std::string trimmed(const std::string & text)
{
    const auto first = text.find_first_not_of(" \t");
    if (first == std::string::npos)
    {
        return std::string();
    }

    const auto last = text.find_last_not_of(" \t");
    return text.substr(first, last - first + 1);
}

std::string lowered(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// A positive count, or throw. Zero divides by zero downstream, and clamping it the way
// getenv_positive does would leave a typo looking like a working setting.
unsigned positive_count(const std::string & text, const std::string & whole)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    {
        LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH=" << whole << " : '" << text
                   << "' is not a positive number";
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }

    try
    {
        const auto value = std::stoul(text);
        if (value == 0 || value > std::numeric_limits<unsigned>::max())
        {
            LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH=" << whole << " : " << text
                       << " is out of range - it must be at least 1";
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }
        return static_cast<unsigned>(value);
    }
    catch (const std::out_of_range &)
    {
        LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH=" << whole << " : " << text << " is too large";
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }
}

} // namespace

FsQueueDepth::FsQueueDepth(unsigned value) :
    _default(value)
{}

FsQueueDepth FsQueueDepth::parse(const std::string & value)
{
    FsQueueDepth out;

    const auto whole = trimmed(value);
    if (whole.empty())
    {
        LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH is empty; it must start with a number of reads";
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }

    std::set<std::string> seen;
    size_t begin = 0;
    bool first = true;

    while (begin <= whole.size())
    {
        const auto comma = whole.find(',', begin);
        const auto element = trimmed(whole.substr(begin, comma == std::string::npos
                                                        ? std::string::npos : comma - begin));

        if (first)
        {
            if (element.find('=') != std::string::npos)
            {
                LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH=" << whole
                           << " must begin with a number, before any <type>=<value> entries";
                throw common::Exception(common::ResponseCode::InvalidParameterError);
            }

            out._default = positive_count(element, whole);
            first = false;
        }
        else
        {
            const auto equals = element.find('=');
            if (equals == std::string::npos)
            {
                LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH=" << whole << " : '" << element
                           << "' is not <type>=<value>";
                throw common::Exception(common::ResponseCode::InvalidParameterError);
            }

            const auto type = lowered(trimmed(element.substr(0, equals)));
            if (type.empty())
            {
                LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH=" << whole << " : '" << element
                           << "' has no filesystem type before the '='";
                throw common::Exception(common::ResponseCode::InvalidParameterError);
            }

            // Rejected, not resolved by order: both numbers were meant, so picking one silently
            // would leave the other looking as if it applied.
            if (!seen.insert(type).second)
            {
                LOG(ERROR) << "RUNAI_STREAMER_FS_QUEUE_DEPTH=" << whole << " : '" << type
                           << "' appears more than once";
                throw common::Exception(common::ResponseCode::InvalidParameterError);
            }

            out._entries.push_back(Entry{ type, positive_count(trimmed(element.substr(equals + 1)), whole) });
        }

        if (comma == std::string::npos)
        {
            break;
        }
        begin = comma + 1;
    }

    return out;
}

unsigned FsQueueDepth::for_type(const std::string & fs_type) const
{
    const auto type = lowered(fs_type);

    for (const auto & entry : _entries)
    {
        // Prefix, so `nfs` covers `nfs4` without the user knowing which the kernel reports.
        if (type.compare(0, entry.type.size(), entry.type) == 0)
        {
            return entry.value;
        }
    }

    return _default;
}

unsigned FsQueueDepth::default_value() const
{
    return _default;
}

const std::vector<FsQueueDepth::Entry> & FsQueueDepth::entries() const
{
    return _entries;
}

std::ostream & operator<<(std::ostream & os, const FsQueueDepth & depth)
{
    os << depth.default_value();

    for (const auto & entry : depth.entries())
    {
        os << ", " << entry.type << "=" << entry.value;
    }

    return os;
}

}; // namespace runai::llm::streamer::impl
