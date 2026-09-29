/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/client/client.h"

#include <aws/core/client/CoreErrors.h>
#include <aws/core/config/ConfigAndCredentialsCacheManager.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "s3/s3_init/s3_init.h"
#include "common/exception/exception.h"
#include "common/response_code/response_code.h"
#include "common/s3_credentials/s3_credentials.h"
#include "common/backend_api/object_storage/object_storage.h"
#include "utils/temp/file/file.h"
#include "utils/temp/env/env.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::impl::s3
{

TEST(ParseEndpointScheme, HttpPrefix)
{
    auto result = parse_endpoint_scheme("http://10.1.254.10");
    EXPECT_EQ(result.host, "10.1.254.10");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTP);
}

TEST(ParseEndpointScheme, HttpPrefixWithPort)
{
    auto result = parse_endpoint_scheme("http://10.1.254.10:9000");
    EXPECT_EQ(result.host, "10.1.254.10:9000");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTP);
}

TEST(ParseEndpointScheme, HttpsPrefix)
{
    auto result = parse_endpoint_scheme("https://s3.amazonaws.com");
    EXPECT_EQ(result.host, "s3.amazonaws.com");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTPS);
}

TEST(ParseEndpointScheme, HttpsPrefixWithPort)
{
    auto result = parse_endpoint_scheme("https://cwobject.com:443");
    EXPECT_EQ(result.host, "cwobject.com:443");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTPS);
}

TEST(ParseEndpointScheme, NoScheme)
{
    auto result = parse_endpoint_scheme("10.1.254.10");
    EXPECT_EQ(result.host, "10.1.254.10");
    EXPECT_FALSE(result.scheme.has_value());
}

TEST(ParseEndpointScheme, NoSchemeWithPort)
{
    auto result = parse_endpoint_scheme("localhost:9000");
    EXPECT_EQ(result.host, "localhost:9000");
    EXPECT_FALSE(result.scheme.has_value());
}

TEST(ParseEndpointScheme, EmptyString)
{
    auto result = parse_endpoint_scheme("");
    EXPECT_EQ(result.host, "");
    EXPECT_FALSE(result.scheme.has_value());
}

TEST(ParseEndpointScheme, HttpInHostname)
{
    // Ensure we only strip the scheme prefix, not "http" appearing elsewhere
    auto result = parse_endpoint_scheme("httpbin.org");
    EXPECT_EQ(result.host, "httpbin.org");
    EXPECT_FALSE(result.scheme.has_value());
}

TEST(ParseEndpointScheme, HttpUpperCase)
{
    auto result = parse_endpoint_scheme("HTTP://10.1.254.10");
    EXPECT_EQ(result.host, "10.1.254.10");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTP);
}

TEST(ParseEndpointScheme, HttpsMixedCase)
{
    auto result = parse_endpoint_scheme("Https://s3.example.com");
    EXPECT_EQ(result.host, "s3.example.com");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTPS);
}

TEST(ParseEndpointScheme, HttpSchemeOnly)
{
    auto result = parse_endpoint_scheme("http://");
    EXPECT_EQ(result.host, "");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTP);
}

TEST(ParseEndpointScheme, HttpsSchemeOnly)
{
    auto result = parse_endpoint_scheme("https://");
    EXPECT_EQ(result.host, "");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTPS);
}

TEST(ParseEndpointScheme, PortWithoutHost)
{
    auto result = parse_endpoint_scheme("http://:9000");
    EXPECT_EQ(result.host, ":9000");
    ASSERT_TRUE(result.scheme.has_value());
    EXPECT_EQ(result.scheme.value(), Aws::Http::Scheme::HTTP);
}

TEST(ApplicationRetryableError, HonorsAwsClassification)
{
    const int internal_failure = static_cast<int>(Aws::Client::CoreErrors::INTERNAL_FAILURE);
    EXPECT_TRUE(application_retryable_error(true, -1, internal_failure));   // transport error without an HTTP response
    EXPECT_TRUE(application_retryable_error(false, -1, internal_failure));  // CRT budget exhaustion clears ShouldRetry
    EXPECT_TRUE(application_retryable_error(true, 500, internal_failure));
    EXPECT_TRUE(application_retryable_error(true, 503, internal_failure));
    EXPECT_TRUE(application_retryable_error(false, 503, internal_failure));
    EXPECT_FALSE(application_retryable_error(false, 301, internal_failure));
}

TEST(ApplicationRetryableError, RejectsPermanentClientErrors)
{
    for (const int status : {400, 401, 403, 404})
    {
        EXPECT_FALSE(application_retryable_error(true, status, 0));
    }

    // These client-side status codes are explicitly transient.
    EXPECT_TRUE(application_retryable_error(true, 408, 0));
    EXPECT_TRUE(application_retryable_error(true, 429, 0));
}

TEST(ApplicationRetryableError, RejectsPermanentErrorsWithoutHttpResponse)
{
    using Aws::Client::CoreErrors;
    for (const auto error_type : {CoreErrors::INVALID_PARAMETER_VALUE,
                                  CoreErrors::MISSING_PARAMETER,
                                  CoreErrors::REQUEST_TIME_TOO_SKEWED,
                                  CoreErrors::CLIENT_SIGNING_FAILURE,
                                  CoreErrors::USER_CANCELLED})
    {
        EXPECT_FALSE(application_retryable_error(false, -1, static_cast<int>(error_type)));
    }
}

namespace
{

void write_config_with_ca_bundle(const std::string & path, const std::string & ca_bundle)
{
    std::ofstream out(path, std::ios::trunc);
    out << "[default]\n";
    out << "ca_bundle = " << ca_bundle << "\n";
}

common::backend_api::ObjectClientConfig_t make_config(std::vector<common::backend_api::ObjectConfigParam_t> & params)
{
    // provide a region so the underlying S3 client can be constructed
    params.push_back(common::backend_api::ObjectConfigParam_t{common::s3::Credentials::REGION_KEY, "us-east-1"});

    common::backend_api::ObjectClientConfig_t config{};
    config.endpoint_url = nullptr;
    config.default_storage_chunk_size = 8 * 1024 * 1024;
    config.initial_params = params.data();
    config.num_initial_params = static_cast<unsigned>(params.size());
    return config;
}

} // namespace

// Verifies that when AWS_CA_BUNDLE is not set, the S3 client resolves the CA
// bundle from the "ca_bundle" setting in the shared AWS config profile.
TEST(CaBundle, ResolvedFromConfigProfile)
{
    // ensure the environment variable does not shadow the profile setting
    // (restored on scope exit to avoid cross-test pollution)
    utils::temp::UnsetEnv ca_bundle_env("AWS_CA_BUNDLE");

    utils::temp::Dir dir;
    const std::string config_path = dir.path + "/aws_config";
    const std::string missing_ca = dir.path + "/missing_ca.pem";
    utils::temp::File valid_ca(dir.path, "ca.pem", utils::random::buffer(32));

    // point the SDK at our temporary config file
    utils::temp::Env config_file_env("AWS_CONFIG_FILE", config_path);

    S3Init init;

    std::vector<common::backend_api::ObjectConfigParam_t> params;
    const auto config = make_config(params);

    // 1) ca_bundle points to a non-existent file -> the client rejects it.
    //    This only fires if the profile ca_bundle was actually read.
    write_config_with_ca_bundle(config_path, missing_ca);
    Aws::Config::ReloadCachedConfigFile();
    try
    {
        S3Client client(config);
        FAIL() << "expected CaFileNotFound for a missing ca_bundle from the config profile";
    }
    catch (const common::Exception & e)
    {
        EXPECT_EQ(e.error(), common::ResponseCode::CaFileNotFound);
    }

    // 2) ca_bundle points to an existing file -> accepted, client is created
    write_config_with_ca_bundle(config_path, valid_ca.path);
    Aws::Config::ReloadCachedConfigFile();
    EXPECT_NO_THROW(S3Client client(config));
}

}; // namespace runai::llm::streamer::impl::s3
