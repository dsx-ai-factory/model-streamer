/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "google/cloud/common_options.h"
#include "model_streamer_gcs_sdk/compatibility.h"

#include "common/storage_uri/storage_uri.h"

namespace runai::llm::streamer::impl::gcs
{

struct ClientConfiguration
{
    // How many clients the caller will build. The threads are divided by it, so the process-wide
    // total stays the same whatever that count is. No default: the streamer resolves it in Config and
    // always states it.
    explicit ClientConfiguration(unsigned concurrent_readers);
    google::cloud::Options options;
    unsigned max_concurrency;
    bool use_grpc;
    // Where the client's credentials are expected to come from, for log and error messages.
    std::string credentials;
};

// Describes where the SDK will look for credentials, following its lookup order: our key file
// variable, then GOOGLE_APPLICATION_CREDENTIALS, then the gcloud well-known file or the metadata
// server. Reads the environment only - the last two cannot be told apart without probing the metadata
// server, so the description names both.
std::string credentials_source();

}; //namespace runai::llm::streamer::impl::gcs
