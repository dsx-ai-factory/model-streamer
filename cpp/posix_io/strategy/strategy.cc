/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/strategy/strategy.h"

#include <algorithm>
#include <map>

#include "common/exception/exception.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// One table, so a name and its enumerator cannot drift apart.
const std::map<Strategy, const char *> _names = {
    { Strategy::IoUringDirect,   "io_uring_direct"   },
    { Strategy::IoUringBuffered, "io_uring_buffered" },
    { Strategy::LibaioDirect,    "libaio_direct"     },
    { Strategy::SyncBuffered,    "sync_buffered"     },
};

std::string trim(const std::string & s)
{
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos)
    {
        return std::string();
    }
    return s.substr(first, s.find_last_not_of(" \t") - first + 1);
}

} // namespace

bool is_async(Strategy strategy)
{
    return strategy != Strategy::SyncBuffered;
}

bool is_direct(Strategy strategy)
{
    return strategy == Strategy::IoUringDirect || strategy == Strategy::LibaioDirect;
}

const char * name(Strategy strategy)
{
    const auto it = _names.find(strategy);
    return it == _names.end() ? "unknown" : it->second;
}

bool strategy_from_name(const std::string & name, Strategy & out)
{
    const auto it = std::find_if(_names.begin(), _names.end(), [&name](const auto & pair)
    {
        return name == pair.second;
    });

    if (it == _names.end())
    {
        return false;
    }

    out = it->first;
    return true;
}

std::vector<Strategy> parse_candidates(const std::string & value)
{
    std::vector<Strategy> candidates;

    size_t pos = 0;
    while (pos <= value.size())
    {
        const auto comma = value.find(',', pos);
        const auto token = trim(value.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));

        // Stray or trailing comma. Rejected, not skipped - a typo must not become a fallback.
        if (token.empty())
        {
            LOG(ERROR) << "Empty strategy name in '" << value << "'";
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }

        Strategy strategy;
        if (!strategy_from_name(token, strategy))
        {
            LOG(ERROR) << "Unknown read strategy '" << token << "'; expected one of "
                       << "io_uring_direct, io_uring_buffered, libaio_direct, sync_buffered";
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }

        // A repeat is unreachable: the first occurrence decides. So it is a typo, not a preference.
        if (std::find(candidates.begin(), candidates.end(), strategy) != candidates.end())
        {
            LOG(ERROR) << "Duplicate read strategy '" << token << "' in '" << value << "'";
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }

        candidates.push_back(strategy);

        if (comma == std::string::npos)
        {
            break;
        }
        pos = comma + 1;
    }

    if (candidates.empty())
    {
        LOG(ERROR) << "Empty read strategy list";
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }

    return candidates;
}

std::ostream & operator<<(std::ostream & os, Strategy strategy)
{
    return os << name(strategy);
}

}; // namespace runai::llm::streamer::posix_io
