/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/client_configuration/client_configuration.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "s3/s3_init/s3_init.h"
#include "utils/temp/env/env.h"

namespace runai::llm::streamer::impl::s3
{

// The SDK must be initialized before a client configuration can be built, and IMDS must be off or the
// constructor spends five seconds asking EC2 for a region (see the note in s3.cc).
//
// RUNAI_STREAMER_S3_TARGET_GBPS is unset here rather than per test: every case measures against the
// SDK's own default, so a value inherited from the environment would change what is being compared.
class ClientConfigurationTest : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        _imds = std::make_unique<utils::temp::Env>(std::string("AWS_EC2_METADATA_DISABLED"), std::string("true"));
        _target = std::make_unique<utils::temp::UnsetEnv>(std::string("RUNAI_STREAMER_S3_TARGET_GBPS"));
        _part = std::make_unique<utils::temp::UnsetEnv>(std::string("RUNAI_STREAMER_S3_CLIENT_PART_SIZE"));
        _init = std::make_unique<S3Init>();
    }

    void TearDown() override
    {
        _init.reset();
        _part.reset();
        _target.reset();
        _imds.reset();
    }

    std::unique_ptr<utils::temp::Env> _imds;
    std::unique_ptr<utils::temp::UnsetEnv> _target;
    std::unique_ptr<utils::temp::UnsetEnv> _part;
    std::unique_ptr<S3Init> _init;
};

// Undocumented and unset by default: raising it removes the CRT's own splitting of a large read into
// parallel parts, measured at 5.7x slower on a 1 GiB chunk. It exists for benchmarking that effect.
TEST_F(ClientConfigurationTest, The_Part_Size_Is_The_Sdk_Default_Unless_Overridden)
{
    utils::temp::UnsetEnv part(std::string("RUNAI_STREAMER_S3_CLIENT_PART_SIZE"));

    const auto sdk_default = ClientConfiguration().config.partSize;
    EXPECT_GT(sdk_default, 0u);

    utils::temp::Env override_(std::string("RUNAI_STREAMER_S3_CLIENT_PART_SIZE"), 32UL * 1024 * 1024);
    EXPECT_EQ(ClientConfiguration().config.partSize, 32u * 1024 * 1024);
}

// Per client, as it has always been - the streamer builds one client per unit of concurrency, so the
// process target is this times that count.
TEST_F(ClientConfigurationTest, The_Target_Is_Per_Client)
{
    const auto sdk_default = ClientConfiguration().config.throughputTargetGbps;
    EXPECT_GT(sdk_default, 0.0);

    utils::temp::Env target(std::string("RUNAI_STREAMER_S3_TARGET_GBPS"), 25UL);
    EXPECT_DOUBLE_EQ(ClientConfiguration().config.throughputTargetGbps, 25.0);
}

}; // namespace runai::llm::streamer::impl::s3
