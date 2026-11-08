/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpClientFactory.h>

#include <memory>
#include <string>
#include <vector>

namespace runai::llm::streamer::impl::s3
{

// System CA bundle locations: Debian/Ubuntu, RHEL/Fedora, openSUSE, OpenELEC, RHEL/CentOS 7+, Alpine.
// The same list, in the same order, that Go's crypto/x509 and the AWS CRT check.
const std::vector<std::string> & system_ca_bundles();

// The CA file for an AWS SDK client that does not name one: AWS_CA_BUNDLE, then the profile's
// ca_bundle (the order the S3 client uses), then the first of `candidates` that exists. Empty if none.
std::string default_ca_file(const std::vector<std::string> & candidates = system_ca_bundles());

// The configuration an SDK client is created with: unchanged if it names a CA file or path,
// otherwise with caFile = default_ca_file().
Aws::Client::ClientConfiguration with_default_ca(const Aws::Client::ClientConfiguration & config);

// The SDK's default HTTP client factory (curl), except that clients created without a CA get one from
// with_default_ca. The SDK builds the clients for STS (web identity / IRSA), SSO and IMDS itself, with no
// CA, so they used only the CA path compiled into the bundled curl: none at all in the cross-compiled
// aarch64 builds, the Debian path elsewhere, so they failed on other distributions and ignored
// AWS_CA_BUNDLE, which the S3 client honors.
class CaHttpClientFactory : public Aws::Http::HttpClientFactory
{
 public:
    CaHttpClientFactory(bool init_curl, bool install_sigpipe_handler);

    std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
        const Aws::Client::ClientConfiguration & config) const override;

    std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
        const Aws::String & uri,
        Aws::Http::HttpMethod method,
        const Aws::IOStreamFactory & stream_factory) const override;

    std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
        const Aws::Http::URI & uri,
        Aws::Http::HttpMethod method,
        const Aws::IOStreamFactory & stream_factory) const override;

    void InitStaticState() override;
    void CleanupStaticState() override;

 private:
    const bool _init_curl;
    const bool _install_sigpipe_handler;
};

}; // namespace runai::llm::streamer::impl::s3
