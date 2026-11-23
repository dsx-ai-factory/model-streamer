/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/s3_credentials/s3_credentials.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::common::s3
{

const char * const Credentials::ACCESS_KEY_ID_KEY = "access_key_id";
const char * const Credentials::SECRET_ACCESS_KEY_KEY = "secret_access_key";
const char * const Credentials::SESSION_TOKEN_KEY = "session_token";
const char * const Credentials::REGION_KEY = "region";
const char * const Credentials::ENDPOINT_KEY = "endpoint";

Credentials::Credentials(const char ** keys, const char ** values, unsigned num_params)
{
    // num_params > 0 but no arrays: the caller declared parameters and did not pass them - a bug, not an
    // empty credential set (which is num_params == 0).
    if (num_params > 0 && (keys == nullptr || values == nullptr))
    {
        LOG(ERROR) << "Credentials: " << num_params << " parameters declared but the "
                   << (keys == nullptr ? "keys" : "values") << " array is null; ignoring all";
        return;
    }

    for (unsigned i = 0; i < num_params; ++i)
    {
        if (keys[i] == nullptr || values[i] == nullptr)
        {
            LOG(ERROR) << "Credentials: null " << (keys[i] == nullptr ? "key" : "value")
                       << " at parameter index " << i << "; skipping it";
            continue;
        }
        _params[keys[i]] = values[i];
    }
}

Credentials::Credentials(const char * access_key_id, const char * secret_access_key, const char * session_token, const char * region, const char * endpoint)
{
    if (access_key_id != nullptr)     _params[ACCESS_KEY_ID_KEY] = access_key_id;
    if (secret_access_key != nullptr) _params[SECRET_ACCESS_KEY_KEY] = secret_access_key;
    if (session_token != nullptr)     _params[SESSION_TOKEN_KEY] = session_token;
    if (region != nullptr)            _params[REGION_KEY] = region;
    if (endpoint != nullptr)          _params[ENDPOINT_KEY] = endpoint;
}

void Credentials::set(const std::string & key, const std::string & value)
{
    _params[key] = value;
}

bool Credentials::has(const std::string & key) const
{
    return _params.find(key) != _params.end();
}

std::optional<std::string> Credentials::get(const std::string & key) const
{
    auto it = _params.find(key);
    if (it == _params.end())
    {
        return std::nullopt;
    }
    return it->second;
}

const std::map<std::string, std::string> & Credentials::params() const
{
    return _params;
}

std::optional<std::string> Credentials::endpoint() const
{
    return get(ENDPOINT_KEY);
}

void Credentials::to_object_client_config(std::vector<common::backend_api::ObjectConfigParam_t> & config) const
{
    for (const auto & [key, value] : _params)
    {
        // endpoint is a dedicated backend client-config field (endpoint_url), not an initial param
        if (key == ENDPOINT_KEY)
        {
            continue;
        }
        config.push_back({ key.c_str(), value.c_str() });
    }
}

bool Credentials::operator==(const Credentials & other) const
{
    return _params == other._params;
}

bool Credentials::operator!=(const Credentials & other) const
{
    return !(*this == other);
}

}; //namespace runai::llm::streamer::common::s3
