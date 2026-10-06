/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "common/client_mgr/client_mgr.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

#include "common/backend_api/object_storage/object_storage.h"
#include "common/storage_uri/storage_uri.h"
#include "common/s3_credentials/s3_credentials.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::common
{

bool strequals(const char *a, const char *b) {
    if (a == nullptr && b == nullptr) {
        return true;
    }
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return strcmp(a, b) == 0;
}

struct Helper : IClient
{
    Helper(const common::backend_api::ObjectClientConfig_t & config) :
        config(config),
        counter(++global_counter),
        id(utils::random::number<size_t>())
    {
    }

    bool verify_credentials(const common::backend_api::ObjectClientConfig_t& other) const {
        if (!strequals(config.endpoint_url, other.endpoint_url)) {
            return false;
        }
        if (config.num_initial_params != other.num_initial_params) {
            return false;
        }
        for (size_t i = 0; i < config.num_initial_params; ++i) {
            if (!strequals(config.initial_params[i].key, other.initial_params[i].key)) {
                return false;
            }
            if (!strequals(config.initial_params[i].value, other.initial_params[i].value)) {
                return false;
            }
        }
        return true;
    }

    const common::backend_api::ObjectClientConfig_t & config;
    static size_t global_counter;
    const size_t counter;

    const size_t id;
};

inline constexpr char HelperName[] = "Helper";

using ClientMgrHelper = ClientMgr<Helper, HelperName>;

size_t Helper::global_counter = 0;

struct ClientMgrTest : ::testing::Test
{
    ClientMgrTest() :
        uri(create_uri()),
        credentials(
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr)),
        params(uri, credentials, utils::random::number<size_t>(), 8)
    {}

    void TearDown() override
    {
        ClientMgrHelper::clear();

        EXPECT_EQ(ClientMgrHelper::size(), 0);
        EXPECT_EQ(ClientMgrHelper::unused(), 0);
    }

    static std::shared_ptr<common::s3::StorageUri> create_uri()
    {
        return std::make_shared<common::s3::StorageUri>("s3://" + utils::random::string() + "/" + utils::random::string());
    }

    std::shared_ptr<common::s3::StorageUri> uri;
    common::s3::Credentials credentials;
    common::s3::S3ClientWrapper::Params params;
};

TEST_F(ClientMgrTest, Creation_Sanity)
{
    EXPECT_EQ(ClientMgrHelper::size(), 0);
    EXPECT_EQ(ClientMgrHelper::unused(), 0);
}

TEST_F(ClientMgrTest, Create_Client)
{
    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
    const auto config = params.to_config(initial_params);
    Helper * helper = ClientMgrHelper::pop(config);

    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 0);

    // used clients are not cleared
    ClientMgrHelper::clear();
    EXPECT_EQ(ClientMgrHelper::size(), 1);

    ClientMgrHelper::push(helper);
    ClientMgrHelper::clear();
    EXPECT_EQ(ClientMgrHelper::size(), 0);
}

TEST_F(ClientMgrTest, Reuse_Client)
{
    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
    const auto config = params.to_config(initial_params);
    Helper * helper = ClientMgrHelper::pop(config);

    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 0);

    ClientMgrHelper::push(helper);
    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 1);

    const auto expected = helper->counter;

    unsigned n = utils::random::number(2, 20);
    for (unsigned i = 0; i < n; ++i)
    {
        const std::string path = utils::random::string();
        params.uri->path = path;
        std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
        const auto config = params.to_config(initial_params);
        Helper * helper = ClientMgrHelper::pop(config);

        EXPECT_TRUE(helper->verify_credentials(config));
        EXPECT_EQ(helper->counter, expected);

        EXPECT_EQ(ClientMgrHelper::size(), 1);
        EXPECT_EQ(ClientMgrHelper::unused(), 0);

        ClientMgrHelper::push(helper);
        EXPECT_EQ(ClientMgrHelper::size(), 1);
        EXPECT_EQ(ClientMgrHelper::unused(), 1);
    }
}

TEST_F(ClientMgrTest, Credentials_Changed)
{
    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
    const auto config = params.to_config(initial_params);
    Helper * helper = ClientMgrHelper::pop(config);

    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 0);

    ClientMgrHelper::push(helper);
    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 1);

    auto expected = helper->counter;

    unsigned n = utils::random::number(2, 20);
    for (unsigned i = 0; i < n; ++i)
    {
        common::s3::Credentials new_credentials(
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr),
            (utils::random::boolean() ? utils::random::string().c_str() : nullptr));

        auto new_uri = std::make_shared<common::s3::StorageUri>(*uri);
        new_uri->path = utils::random::string();
        common::s3::S3ClientWrapper::Params new_params(new_uri, new_credentials, utils::random::number<size_t>(), 8);
        std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
        const auto new_config = new_params.to_config(initial_params);
        bool changed = !helper->verify_credentials(new_config);
        Helper * helper = ClientMgrHelper::pop(new_config);

        EXPECT_TRUE(helper->verify_credentials(new_config));

        EXPECT_EQ(helper->counter != expected, changed);

        expected = helper->counter;

        EXPECT_EQ(ClientMgrHelper::size(), 1);
        EXPECT_EQ(ClientMgrHelper::unused(), 0);

        ClientMgrHelper::push(helper);
        EXPECT_EQ(ClientMgrHelper::size(), 1);
        EXPECT_EQ(ClientMgrHelper::unused(), 1);
    }
}

TEST_F(ClientMgrTest, Change_Bucket)
{
    std::set<Helper *> used;
    unsigned m = utils::random::number(2, 20);
    for (unsigned i = 0; i < m; ++i)
    {
        std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
        const auto config = params.to_config(initial_params);
        Helper * helper = ClientMgrHelper::pop(config);

        EXPECT_EQ(ClientMgrHelper::size(), i + 1);
        EXPECT_EQ(ClientMgrHelper::unused(), 0);

        used.insert(helper);
    }

    for (auto helper : used)
    {
        ClientMgrHelper::push(helper);
    }
    EXPECT_EQ(ClientMgrHelper::unused(), m);

    unsigned n = utils::random::number(2, 20);
    auto uri_ = create_uri();
    while (uri_->bucket == uri->bucket)
    {
        uri_ = create_uri();
    }

    used.clear();
    for (unsigned i = 0; i < n; ++i)
    {
        const std::string path = utils::random::string();
        uri_->path = path;
        common::s3::S3ClientWrapper::Params new_params(uri_, credentials, utils::random::number<size_t>(), 8);
        std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
        const auto new_config = new_params.to_config(initial_params);
        Helper * helper = ClientMgrHelper::pop(new_config);
        EXPECT_EQ(ClientMgrHelper::unused(), (m > i + 1 ? m - i - 1 : 0));

        EXPECT_EQ(ClientMgrHelper::size(), (m > i + 1 ? m : i + 1));
        used.insert(helper);
    }

    for (auto helper : used)
    {
        ClientMgrHelper::push(helper);
    }
}

TEST_F(ClientMgrTest, Remove_Client)
{
    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
    const auto config = params.to_config(initial_params);
    Helper * helper = ClientMgrHelper::pop(config);

    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 0);

    ClientMgrHelper::push(helper);
}

TEST_F(ClientMgrTest, Remove_Client_Is_Reentrant)
{
    std::vector<common::backend_api::ObjectConfigParam_t> initial_params;
    const auto config = params.to_config(initial_params);
    Helper * helper = ClientMgrHelper::pop(config);

    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 0);

    ClientMgrHelper::push(helper);

    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 1);

    // push is reentrant
    EXPECT_NO_THROW(ClientMgrHelper::push(helper));
    EXPECT_EQ(ClientMgrHelper::size(), 1);
    EXPECT_EQ(ClientMgrHelper::unused(), 1);
}

}; //namespace runai::llm::streamer::common
