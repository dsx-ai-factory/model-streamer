/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <optional>
#include <utility>
#include <vector>

#include <aws/core/http/Scheme.h>

#include "s3/client_configuration/client_configuration.h"
#include "common/backend_api/response/response.h"

#include "common/backend_api/object_storage/object_storage.h"
#include "common/client_mgr/client_mgr.h"
#include "common/storage_uri/storage_uri.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "common/shared_queue/shared_queue.h"
#include "common/range/range.h"

namespace runai::llm::streamer::impl::s3
{

struct EndpointParseResult
{
    Aws::String host;
    std::optional<Aws::Http::Scheme> scheme;
};

EndpointParseResult parse_endpoint_scheme(const Aws::String & endpoint);

// Preserve AWS's retryability classification after the CRT retry policy is exhausted while never retrying
// permanent client-side failures. Feature enablement is applied separately at the response call site.
bool application_retryable_error(bool aws_should_retry, int http_status, int error_type);

struct S3ClientBase : common::IClient
{
    S3ClientBase(const common::backend_api::ObjectClientConfig_t & config);

     // verify that clien's credentials have not change
    bool verify_credentials(const common::backend_api::ObjectClientConfig_t & config) const;

 protected:
    std::optional<Aws::String> _key;
    std::optional<Aws::String> _secret;
    std::optional<Aws::String> _token;
    std::optional<Aws::String> _region;
    const std::optional<Aws::String> _endpoint;
    std::unique_ptr<Aws::Auth::AWSCredentials> _client_credentials;
    const size_t _chunk_bytesize;

 private:
    bool verify_credentials_member(const std::optional<Aws::String>& client_member, const std::optional<Aws::String>& input_member, const char * name) const;
};

struct S3Client : S3ClientBase
{
    S3Client(const common::backend_api::ObjectClientConfig_t & config);

    common::backend_api::ResponseCode_t async_read(const char* path,
                                                   common::backend_api::ObjectRange_t range,
                                                   char* destination_buffer,
                                                   common::backend_api::ObjectRequestId_t request_id);

    common::backend_api::Response async_read_response();

    // Synchronously lists all objects under prefix (full URI, e.g. "s3://bucket/models/").
    // Appends (full_uri, size) pairs to results; handles S3 pagination internally.
    common::ResponseCode list_files(const char* prefix, int is_recursive,
                                    std::vector<std::pair<std::string, size_t>>& results);

    // Stop sending requests to the object store
    // Requests that were already sent cannot be cancelled, since the Aws S3CrtClient does not support aborting requests
    // The S3CrtClient d'tor will wait for response of all teh sent requests, which can take a while
    void stop();

    // In-flight submission window (bytes) for this client: a bandwidth-delay product from
    // the configured chunk size (ObjectClientConfig_t.default_storage_chunk_size) and the
    // target throughput (throughputTargetGbps). Surfaced to the generic layer via the
    // backend config query.
    size_t max_inflight_bytes() const { return _max_inflight_bytes; }

    using S3ClientBase::verify_credentials;

 private:
    std::atomic<bool> _stop;
    const bool _application_retries_enabled;   // RUNAI_STREAMER_S3_TIMEOUT > 0, cached at client creation
    ClientConfiguration _client_config;
    std::unique_ptr<Aws::S3Crt::S3CrtClient> _client;
    size_t _max_inflight_bytes = 0;

    // queue of asynchronous responses
    using Responder = common::SharedQueue<common::backend_api::Response>;
    std::shared_ptr<Responder> _responder;
};

}; //namespace runai::llm::streamer::impl::s3
