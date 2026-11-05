/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gcs/client_configuration/client_configuration.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include "common/exception/exception.h"
#include "utils/temp/env/env.h"

namespace runai::llm::streamer::impl::gcs
{

namespace
{

// The formula the configuration implements, restated so the tests assert against the machine they run
// on rather than a literal thread count.
unsigned expected(unsigned readers)
{
    const unsigned nprocs = std::thread::hardware_concurrency();
    const unsigned floor = nprocs == 0 ? 8U : 1U;
    return std::max(floor, nprocs * 2 / readers);
}

} // namespace

// The variable the formula reads is cleared before each test, so a test states only what it SETS. A
// non-zero RUNAI_STREAMER_S3_MAX_CONNECTIONS replaces the formula outright, so a run that inherits it
// fails these against correct code - `bazel test` scrubs the environment, but `--test_env` and a
// direct binary run do not.
class GcsConcurrency : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        _connections = std::make_unique<utils::temp::UnsetEnv>(std::string("RUNAI_STREAMER_S3_MAX_CONNECTIONS"));
    }

    void TearDown() override
    {
        _connections.reset();
    }

    std::unique_ptr<utils::temp::UnsetEnv> _connections;
};

// The threads are sized by the number of CLIENTS the caller will build, which reaches the plugin as a
// client parameter - no environment variable is read for it here.
TEST_F(GcsConcurrency, Threads_Follow_The_Reader_Count)
{
    for (const unsigned readers : { 2U, 4U, 8U, 16U })
    {
        EXPECT_EQ(ClientConfiguration(readers).max_concurrency, expected(readers)) << readers;
    }
}

// The property the divisor exists for: N clients with nprocs*2/N threads each is the same total
// whatever N is.
//
// Only reader counts that DIVIDE the total are checked. The division truncates otherwise, and that
// loss belongs to the formula rather than to the count it was given.
TEST_F(GcsConcurrency, The_Total_Thread_Count_Does_Not_Move)
{
    const unsigned nprocs = std::thread::hardware_concurrency();
    ASSERT_GT(nprocs, 0u) << "the formula falls back to a fixed 8 with no cores to divide";

    const unsigned total = nprocs * 2;
    unsigned checked = 0;

    for (unsigned readers = 2; readers <= total; ++readers)
    {
        if (total % readers != 0)
        {
            continue;
        }

        EXPECT_EQ(readers * ClientConfiguration(readers).max_concurrency, total) << readers;
        ++checked;
    }

    EXPECT_GT(checked, 0u) << "no reader count divided " << total << ", so nothing was asserted";
}

// Only a caller outside the streamer can send zero, and dividing by it would be a crash rather than a
// misconfiguration. Floored to one - the streamer itself always states a count.
TEST_F(GcsConcurrency, Zero_Is_Floored_Rather_Than_Dividing_By_Zero)
{
    EXPECT_EQ(ClientConfiguration(0).max_concurrency, expected(1));
}

// An explicit connection count names the threads directly, so the reader count does not divide it.
TEST_F(GcsConcurrency, An_Explicit_Connection_Count_Overrides_The_Formula)
{
    utils::temp::Env connections(std::string("RUNAI_STREAMER_S3_MAX_CONNECTIONS"), 7UL);

    EXPECT_EQ(ClientConfiguration(16).max_concurrency, 7u);
}

// Both credential variables are cleared before each test, so a test states only what it SETS.
class GcsCredentials : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        _key_file = std::make_unique<utils::temp::UnsetEnv>(std::string("RUNAI_STREAMER_GCS_CREDENTIAL_FILE"));
        _adc_file = std::make_unique<utils::temp::UnsetEnv>(std::string("GOOGLE_APPLICATION_CREDENTIALS"));
    }

    void TearDown() override
    {
        _adc_file.reset();
        _key_file.reset();
    }

    std::unique_ptr<utils::temp::UnsetEnv> _key_file;
    std::unique_ptr<utils::temp::UnsetEnv> _adc_file;
};

TEST_F(GcsCredentials, Without_Variables_The_Sdk_Falls_Back_To_The_Metadata_Server)
{
    const auto source = credentials_source();

    EXPECT_NE(source.find("metadata server"), std::string::npos) << source;
    EXPECT_NE(ClientConfiguration(1).credentials.find("metadata server"), std::string::npos);
}

TEST_F(GcsCredentials, The_Adc_Variable_Names_Its_File)
{
    const std::string file = ::testing::TempDir() + "adc.json";
    std::ofstream(file) << "{}";
    utils::temp::Env adc(std::string("GOOGLE_APPLICATION_CREDENTIALS"), file);

    const auto source = credentials_source();

    EXPECT_NE(source.find("GOOGLE_APPLICATION_CREDENTIALS"), std::string::npos) << source;
    EXPECT_NE(source.find(file), std::string::npos) << source;
    EXPECT_EQ(source.find("not readable"), std::string::npos) << source;
}

// The SDK fails on a missing file only when it first authenticates, so the description says so up front.
TEST_F(GcsCredentials, A_Missing_Adc_File_Is_Reported)
{
    utils::temp::Env adc(std::string("GOOGLE_APPLICATION_CREDENTIALS"), std::string("/nonexistent/adc.json"));

    const auto source = credentials_source();

    EXPECT_NE(source.find("not readable"), std::string::npos) << source;
}

// The streamer's own key file variable is applied as an explicit credential, ahead of the SDK's lookup.
TEST_F(GcsCredentials, The_Key_File_Variable_Takes_Precedence)
{
    utils::temp::Env adc(std::string("GOOGLE_APPLICATION_CREDENTIALS"), std::string("/nonexistent/adc.json"));
    utils::temp::Env key(std::string("RUNAI_STREAMER_GCS_CREDENTIAL_FILE"), std::string("/path/key.json"));

    const auto source = credentials_source();

    EXPECT_NE(source.find("RUNAI_STREAMER_GCS_CREDENTIAL_FILE"), std::string::npos) << source;
    EXPECT_NE(source.find("/path/key.json"), std::string::npos) << source;
    EXPECT_EQ(source.find("GOOGLE_APPLICATION_CREDENTIALS"), std::string::npos) << source;
}

// An unreadable key file is an invalid parameter rather than a client that fails on its first read.
TEST_F(GcsCredentials, An_Unreadable_Key_File_Fails_Configuration)
{
    utils::temp::Env key(std::string("RUNAI_STREAMER_GCS_CREDENTIAL_FILE"), std::string("/nonexistent/key.json"));

    EXPECT_THROW(ClientConfiguration(1), common::Exception);
}

}; // namespace runai::llm::streamer::impl::gcs
