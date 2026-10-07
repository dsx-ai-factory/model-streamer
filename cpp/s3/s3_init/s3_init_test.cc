/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/s3_init/s3_init.h"

#include <gtest/gtest.h>

#include <memory>

namespace runai::llm::streamer::impl::s3
{

TEST(Creation, Sanity)
{
    std::unique_ptr<S3Init> init;
    EXPECT_NO_THROW(init = std::make_unique<S3Init>());
}

}; // namespace runai::llm::streamer::impl::s3