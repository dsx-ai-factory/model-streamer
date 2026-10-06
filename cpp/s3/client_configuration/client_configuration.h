/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <aws/core/Aws.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/model/BucketLocationConstraint.h>


#include <cstddef>

#include "common/storage_uri/storage_uri.h"

namespace runai::llm::streamer::impl::s3
{

// In-flight window (bytes) for object-storage submission: a bandwidth-delay product from
// the target throughput (target_gbps, i.e. throughputTargetGbps), floored at a couple of
// chunk_bytesize (the client's default_storage_chunk_size). RUNAI_STREAMER_S3_MAX_INFLIGHT_MIB
// (in MiB) overrides it directly.
size_t inflight_window_bytes(size_t chunk_bytesize, double target_gbps);

struct ClientConfiguration
{
    ClientConfiguration();

    Aws::S3Crt::ClientConfiguration config;
};

}; //namespace runai::llm::streamer::impl::s3
