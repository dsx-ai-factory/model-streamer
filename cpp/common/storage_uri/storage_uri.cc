/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex>
#include <vector>

#include "common/storage_uri/storage_uri.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::common::s3
{

static const std::string gcsProtocol("gs");
static const std::string azureProtocol("az");

StorageUri::StorageUri(const std::string & uri) : uri(uri)
{
    static const std::regex awsRegex("^(s3|gs|az)://([^/]+)/(.+)$");

    std::smatch match;

    LOG(SPAM) << "Checking: " << uri;

    if (!std::regex_match(uri, match, awsRegex))
    {
        LOG(SPAM) << "'" << uri << "' is not in s3 format";
        throw std::exception();
    }

    scheme = match[1];
    bucket = match[2];
    path = match[3];

    LOG(SPAM) << "scheme: " << scheme << " bucket: " << bucket << " path: " << path;
}

std::ostream & operator<<(std::ostream & os, const StorageUri & uri)
{
    return os << "scheme: " << uri.scheme << " bucket: " << uri.bucket << " path: " << uri.path;
}

bool StorageUri::is_gcs() const
{
    return scheme == gcsProtocol;
}

bool StorageUri::is_azure() const
{
    return scheme == azureProtocol;
}

StorageUri_C::StorageUri_C(const StorageUri & uri) :
    bucket(uri.bucket.c_str()),
    path(uri.path.c_str())
{}

}; // namespace runai::llm::streamer::common::s3
