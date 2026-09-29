/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <aws/core/Aws.h>
#include <aws/s3-crt/S3CrtClient.h>
//#include <aws/s3-crt/model/BucketLocationConstraint.h>

namespace runai::llm::streamer::impl::s3
{

struct S3Init
{
    S3Init();
    ~S3Init();

    Aws::SDKOptions options;
};

}; //namespace runai::llm::streamer::impl::s3