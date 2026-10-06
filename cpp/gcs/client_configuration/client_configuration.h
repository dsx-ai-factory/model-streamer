/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "google/cloud/common_options.h"

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
};

}; //namespace runai::llm::streamer::impl::gcs
