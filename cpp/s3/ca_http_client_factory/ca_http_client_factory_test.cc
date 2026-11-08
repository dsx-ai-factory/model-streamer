/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/ca_http_client_factory/ca_http_client_factory.h"

#include <gtest/gtest.h>

#include <aws/core/config/ConfigAndCredentialsCacheManager.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "s3/s3_init/s3_init.h"
#include "utils/random/random.h"
#include "utils/temp/env/env.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::impl::s3
{

namespace
{

// No AWS_CA_BUNDLE and a config file without a profile ca_bundle, so only the candidates apply
class CaHttpClientFactoryTest : public ::testing::Test
{
 protected:
    utils::temp::Dir dir;
    const std::string config_path = dir.path + "/aws_config";
    utils::temp::UnsetEnv ca_bundle_env{std::string("AWS_CA_BUNDLE")};
    utils::temp::Env config_file_env{std::string("AWS_CONFIG_FILE"), config_path};
    S3Init init;

    void SetUp() override
    {
        write_config("");
    }

    void write_config(const std::string & ca_bundle)
    {
        std::ofstream out(config_path);
        out << "[default]\n";
        if (!ca_bundle.empty())
        {
            out << "ca_bundle = " << ca_bundle << "\n";
        }
        out.close();
        Aws::Config::ReloadCachedConfigFile();
    }
};

} // namespace

TEST_F(CaHttpClientFactoryTest, System_Bundles_Cover_The_Common_Distributions)
{
    const auto & bundles = system_ca_bundles();
    ASSERT_FALSE(bundles.empty());
    EXPECT_EQ(bundles.front(), "/etc/ssl/certs/ca-certificates.crt");                     // Debian/Ubuntu
    EXPECT_NE(std::find(bundles.begin(), bundles.end(), "/etc/pki/tls/certs/ca-bundle.crt"), bundles.end()); // RHEL
}

TEST_F(CaHttpClientFactoryTest, First_Existing_Candidate_Is_Used)
{
    utils::temp::File second(dir.path, "second.pem", utils::random::buffer(16));
    utils::temp::File third(dir.path, "third.pem", utils::random::buffer(16));
    const std::vector<std::string> candidates = {dir.path + "/missing.pem", second.location(), third.location()};

    EXPECT_EQ(default_ca_file(candidates), second.location());
}

TEST_F(CaHttpClientFactoryTest, No_Candidate_Exists)
{
    EXPECT_EQ(default_ca_file({dir.path + "/missing.pem"}), "");
}

TEST_F(CaHttpClientFactoryTest, Aws_Ca_Bundle_Takes_Precedence)
{
    utils::temp::File system(dir.path, "system.pem", utils::random::buffer(16));
    utils::temp::File profile(dir.path, "profile.pem", utils::random::buffer(16));
    write_config(profile.location());

    utils::temp::Env env(std::string("AWS_CA_BUNDLE"), std::string("/custom/ca.pem"));
    EXPECT_EQ(default_ca_file({system.location()}), "/custom/ca.pem");
}

TEST_F(CaHttpClientFactoryTest, Profile_Ca_Bundle_Takes_Precedence_Over_System_Bundles)
{
    utils::temp::File system(dir.path, "system.pem", utils::random::buffer(16));
    utils::temp::File profile(dir.path, "profile.pem", utils::random::buffer(16));
    write_config(profile.location());

    EXPECT_EQ(default_ca_file({system.location()}), profile.location());
}

TEST_F(CaHttpClientFactoryTest, Configured_Ca_Is_Kept)
{
    utils::temp::Env env(std::string("AWS_CA_BUNDLE"), std::string("/custom/ca.pem"));

    Aws::Client::ClientConfiguration with_file;
    with_file.caFile = "/explicit/ca.pem";
    EXPECT_EQ(with_default_ca(with_file).caFile, "/explicit/ca.pem");

    Aws::Client::ClientConfiguration with_path;
    with_path.caPath = "/explicit/certs";
    EXPECT_EQ(with_default_ca(with_path).caFile, "");
    EXPECT_EQ(with_default_ca(with_path).caPath, "/explicit/certs");
}

TEST_F(CaHttpClientFactoryTest, Missing_Ca_Is_Filled_In)
{
    utils::temp::Env env(std::string("AWS_CA_BUNDLE"), std::string("/custom/ca.pem"));

    Aws::Client::ClientConfiguration config;
    config.region = "eu-west-1";
    const auto result = with_default_ca(config);
    EXPECT_EQ(result.caFile, "/custom/ca.pem");
    EXPECT_EQ(result.region, "eu-west-1");
}

TEST_F(CaHttpClientFactoryTest, Sdk_Creates_Clients_Through_The_Factory)
{
    // S3Init installs the factory: the SDK's own client creation goes through it and yields a client
    Aws::Client::ClientConfiguration config;
    EXPECT_NE(Aws::Http::CreateHttpClient(config), nullptr);
}

}; // namespace runai::llm::streamer::impl::s3
