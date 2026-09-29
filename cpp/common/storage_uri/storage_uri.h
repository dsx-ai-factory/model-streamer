/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace runai::llm::streamer::common::s3
{

struct StorageUri
{
    StorageUri(const std::string & uri);

    std::string scheme;
    std::string uri;
    std::string bucket;
    std::string path;

    bool is_gcs() const;
    bool is_azure() const;
};

struct StorageUri_C
{
    StorageUri_C(const StorageUri & uri);

    const char * bucket;
    const char * path;
};

std::ostream & operator<<(std::ostream &, const StorageUri &);

}; //namespace runai::llm::streamer::common::s3
