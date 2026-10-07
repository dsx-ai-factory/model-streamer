/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <optional>

namespace runai::llm::streamer::impl::azure
{

struct ClientConfiguration
{
    // Azure client configuration options
    std::optional<std::string> account_name;
    std::optional<std::string> account_key;
    std::optional<std::string> sas_token;
    std::string endpoint_suffix = "blob.core.windows.net";
#ifdef AZURITE_TESTING
    // Connection string is only available for Azurite/local testing
    std::optional<std::string> connection_string;
#endif
    
    // Concurrency settings
    unsigned int max_concurrency = 8;

    // How many clients the caller will build. The threads are divided by it, so the process-wide
    // total stays the same whatever that count is. No default: the streamer resolves it in Config and
    // always states it.
    explicit ClientConfiguration(unsigned concurrent_readers);
};

} // namespace runai::llm::streamer::impl::azure
