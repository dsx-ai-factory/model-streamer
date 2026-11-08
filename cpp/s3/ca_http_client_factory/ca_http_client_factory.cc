/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/ca_http_client_factory/ca_http_client_factory.h"

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/config/ConfigAndCredentialsCacheManager.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>

#include <csignal>

#include "utils/env/env.h"
#include "utils/fd/fd.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl::s3
{

namespace
{

constexpr char AllocationTag[] = "RunaiCaHttpClientFactory";

// The SDK's default factory installs a handler that swallows SIGPIPE (rather than ignoring it, which
// child processes would inherit across exec); do the same.
void swallow_signal(int /* signal */)
{
}

} // namespace

const std::vector<std::string> & system_ca_bundles()
{
    static const std::vector<std::string> bundles = {
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/ca-bundle.pem",
        "/etc/pki/tls/cacert.pem",
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
        "/etc/ssl/cert.pem",
    };
    return bundles;
}

std::string default_ca_file(const std::vector<std::string> & candidates)
{
    std::string ca_file;
    if (utils::try_getenv("AWS_CA_BUNDLE", ca_file) && !ca_file.empty())
    {
        return ca_file;
    }

    const auto profile_ca = Aws::Config::GetCachedConfigValue(Aws::Auth::GetConfigProfileName(), "ca_bundle");
    if (!profile_ca.empty())
    {
        return profile_ca;
    }

    for (const auto & candidate : candidates)
    {
        if (utils::Fd::exists(candidate))
        {
            return candidate;
        }
    }
    return {};
}

Aws::Client::ClientConfiguration with_default_ca(const Aws::Client::ClientConfiguration & config)
{
    if (!config.caFile.empty() || !config.caPath.empty())
    {
        return config;
    }

    Aws::Client::ClientConfiguration result(config);
    const auto ca_file = default_ca_file();
    if (ca_file.empty())
    {
        LOG(WARNING) << "No CA bundle found for AWS SDK requests (set AWS_CA_BUNDLE); using the bundled curl default";
        return result;
    }
    if (!utils::Fd::exists(ca_file))
    {
        LOG(ERROR) << "CA file for AWS SDK requests not found: " << ca_file;
    }
    LOG(DEBUG) << "AWS SDK requests verify TLS with CA file " << ca_file;
    result.caFile = ca_file;
    return result;
}

CaHttpClientFactory::CaHttpClientFactory(bool init_curl, bool install_sigpipe_handler) :
    _init_curl(init_curl),
    _install_sigpipe_handler(install_sigpipe_handler)
{
}

std::shared_ptr<Aws::Http::HttpClient> CaHttpClientFactory::CreateHttpClient(
    const Aws::Client::ClientConfiguration & config) const
{
    return Aws::MakeShared<Aws::Http::CurlHttpClient>(AllocationTag, with_default_ca(config));
}

std::shared_ptr<Aws::Http::HttpRequest> CaHttpClientFactory::CreateHttpRequest(
    const Aws::String & uri,
    Aws::Http::HttpMethod method,
    const Aws::IOStreamFactory & stream_factory) const
{
    return CreateHttpRequest(Aws::Http::URI(uri), method, stream_factory);
}

std::shared_ptr<Aws::Http::HttpRequest> CaHttpClientFactory::CreateHttpRequest(
    const Aws::Http::URI & uri,
    Aws::Http::HttpMethod method,
    const Aws::IOStreamFactory & stream_factory) const
{
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(AllocationTag, uri, method);
    request->SetResponseStreamFactory(stream_factory);
    return request;
}

void CaHttpClientFactory::InitStaticState()
{
    if (_init_curl)
    {
        Aws::Http::CurlHttpClient::InitGlobalState();
    }
    if (_install_sigpipe_handler)
    {
        ::signal(SIGPIPE, swallow_signal);
    }
}

void CaHttpClientFactory::CleanupStaticState()
{
    if (_init_curl)
    {
        Aws::Http::CurlHttpClient::CleanupGlobalState();
    }
}

}; // namespace runai::llm::streamer::impl::s3
