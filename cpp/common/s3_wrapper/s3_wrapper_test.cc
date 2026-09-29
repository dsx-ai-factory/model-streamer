/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/s3_wrapper/s3_wrapper.h"

#include <gtest/gtest.h>
#include <cstdint>

#include "utils/dylib/dylib.h"
#include "utils/random/random.h"
#include "utils/temp/env/env.h"
namespace runai::llm::streamer::common::s3
{

struct S3WrappertTest : ::testing::Test
{
    S3WrappertTest() :
        uri("s3://" + utils::random::string() + "/" + utils::random::string()),
        credentials(
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr)),
        params(std::make_shared<StorageUri>(uri), credentials, utils::random::number<size_t>(), 8)
    {
        auto ptr = utils::random::number<uintptr_t>();
        request_id = reinterpret_cast<common::backend_api::ObjectRequestId_t>(ptr);
    }

    void TearDown() override
    {
        S3ClientWrapper::shutdown();
    }

    StorageUri uri;
    Credentials credentials;
    S3ClientWrapper::Params params;
    common::backend_api::ObjectRequestId_t request_id;
};

TEST_F(S3WrappertTest, Creation_Sanity)
{
    EXPECT_NO_THROW(S3ClientWrapper wrapper(params));
}

TEST_F(S3WrappertTest, Creation)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    EXPECT_EQ(verify_mock(), 0);

    {
        S3ClientWrapper wrapper(params);
        EXPECT_EQ(verify_mock(), 1);
    }
    S3ClientWrapper::shutdown();
    EXPECT_EQ(verify_mock(), 0);
}

TEST_F(S3WrappertTest, Read)
{
    S3ClientWrapper wrapper(params);
    Range range;
    auto response_code = wrapper.async_read(params, request_id, range, nullptr);
    EXPECT_EQ(response_code, common::ResponseCode::Success);
}

TEST_F(S3WrappertTest, Response)
{
    S3ClientWrapper wrapper(params);
    Range range;
    auto response_code = wrapper.async_read(params, request_id, range, nullptr);
    EXPECT_EQ(response_code, common::ResponseCode::Success);

    std::vector<backend_api::ObjectCompletionEvent_t> event_buffer;
    response_code = wrapper.async_read_response(event_buffer, 1);
    EXPECT_EQ(response_code, common::ResponseCode::Success);
    EXPECT_EQ(event_buffer.size(), 1);
    EXPECT_EQ(event_buffer.at(0).request_id, request_id);
    EXPECT_EQ(event_buffer.at(0).response_code, common::ResponseCode::Success);
}

TEST_F(S3WrappertTest, Cleanup)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto is_shutdown = dylib.dlsym<bool(*)(void)>("runai_mock_s3_is_shutdown");
    EXPECT_EQ(verify_mock(), 0);
    {
        S3ClientWrapper wrapper(params);
        Range range;
        wrapper.async_read(params, request_id, range, nullptr);

        EXPECT_EQ(verify_mock(), 1);

        S3ClientWrapper::shutdown();
        EXPECT_EQ(verify_mock(), 1);
        EXPECT_EQ(is_shutdown(), false); // client still holds reference to backend handle
    }

    S3ClientWrapper::shutdown();
    EXPECT_EQ(verify_mock(), 0);
    EXPECT_EQ(is_shutdown(), false); // backend handle is closed when the process exits
}

TEST_F(S3WrappertTest, Endpoint_Exists)
{
    auto endpoint = utils::random::string();
    utils::temp::Env endpoint_env("AWS_ENDPOINT_URL", endpoint);

    Credentials credentials_;
    S3ClientWrapper::Params params_(std::make_shared<StorageUri>(uri), credentials_, utils::random::number<size_t>(), 8);
    S3ClientWrapper wrapper(params);

    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;

    EXPECT_EQ(params_.to_config(initial_params).endpoint_url, endpoint);
}

TEST_F(S3WrappertTest, Endpoint_In_Credentials)
{
    auto endpoint = utils::random::string();
    Credentials credentials_(nullptr, nullptr, nullptr, nullptr, endpoint.c_str());

    S3ClientWrapper::Params params_(std::make_shared<StorageUri>(uri), credentials_, utils::random::number<size_t>(), 8);
    S3ClientWrapper wrapper(params);

    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
    EXPECT_EQ(params_.to_config(initial_params).endpoint_url, endpoint);
}

TEST_F(S3WrappertTest, Shutdown_Policy)
{
    utils::Dylib dylib("libstreamers3.so");
    auto verify_mock = dylib.dlsym<int(*)(void)>("runai_mock_s3_clients");
    auto is_shutdown = dylib.dlsym<bool(*)(void)>("runai_mock_s3_is_shutdown");
    // set shutdown policy to on process exit
    auto set_shutdown_policy = dylib.dlsym<void(*)(common::backend_api::ObjectShutdownPolicy_t)>("runai_s3_mock_set_backend_shutdown_policy");
    set_shutdown_policy(common::backend_api::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN);

    EXPECT_EQ(verify_mock(), 0);
    {
        S3ClientWrapper wrapper(params);
        Range range;
        wrapper.async_read(params, request_id, range, nullptr);

        EXPECT_EQ(verify_mock(), 1);

        EXPECT_EQ(is_shutdown(), false);

        EXPECT_NO_THROW(S3ClientWrapper::shutdown());
        EXPECT_EQ(is_shutdown(), false); // client still holds reference to backend handle
    }

    EXPECT_NO_THROW(S3ClientWrapper::shutdown());
    EXPECT_EQ(is_shutdown(), true);
}

TEST(BackendHandle, GetPluginType_S3)
{
    auto storage_uri = std::make_shared<StorageUri>("s3://bucket/path");
    ObjectPluginType plugin_type = S3ClientWrapper::BackendHandle::get_libstreamers_plugin_type(storage_uri);
    EXPECT_EQ(plugin_type, ObjectPluginType::ObjStorageS3);
}

TEST(BackendHandle, GetPluginType_GCS)
{
    auto storage_uri = std::make_shared<StorageUri>("gs://bucket/path");
    ObjectPluginType plugin_type = S3ClientWrapper::BackendHandle::get_libstreamers_plugin_type(storage_uri);
    EXPECT_EQ(plugin_type, ObjectPluginType::ObjStorageGCS);
}

// A typed field rather than a dictionary entry, so it needs no key, no parsing and no validation -
// and GCS and Azure divide by the same number the streamer resolved.
TEST(Params, The_Reader_Count_Reaches_The_Client_Config)
{
    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;

    const S3ClientWrapper::Params params(std::make_shared<StorageUri>("s3://bucket/path"), Credentials(),
                                         8 * 1024 * 1024, 8);

    EXPECT_EQ(params.to_config(initial_params).concurrent_readers, 8u);
}

// The short constructor takes no credentials, and must still carry the count: it is not a default,
// so an overload that dropped it would divide by a number nobody chose.
TEST(Params, The_Reader_Count_Survives_The_Credentialless_Constructor)
{
    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;

    const S3ClientWrapper::Params params(std::make_shared<StorageUri>("s3://bucket/path"), 8 * 1024 * 1024, 4);

    EXPECT_EQ(params.to_config(initial_params).concurrent_readers, 4u);
}

}; // namespace runai::llm::streamer::common::s3
