/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <optional>
#include <utility>
#include <vector>

#include "azure/client_configuration/client_configuration.h"
#include "azure/client/async_azure_client/async_azure_client.h"
#include "common/backend_api/response/response.h"

#include "common/backend_api/object_storage/object_storage.h"
#include "common/client_mgr/client_mgr.h"
#include "common/storage_uri/storage_uri.h"
#include "common/shared_queue/shared_queue.h"
#include "common/range/range.h"

#include <azure/storage/blobs.hpp>

namespace runai::llm::streamer::impl::azure
{

struct AzureClient : common::IClient
{
    AzureClient(const common::backend_api::ObjectClientConfig_t& config);
    ~AzureClient();

    // verify that client's credentials have not changed
    bool verify_credentials(const common::backend_api::ObjectClientConfig_t & config) const;

    common::ResponseCode async_read(const char* path,
                                    common::backend_api::ObjectRange_t range,
                                    char* destination_buffer,
                                    common::backend_api::ObjectRequestId_t request_id);

    common::backend_api::Response async_read_response();

    common::ResponseCode list_files(const char* prefix, int is_recursive,
                                    std::vector<std::pair<std::string, size_t>>& results);

    // Stop sending requests to the object store
    // If stopped before all requests for an async_read() call are sent, subsequent request chunks will not be sent.
    void stop();

    // In-flight window (bytes) advertised to the generic layer: async threadpool size x chunk x margin
    size_t max_inflight_bytes() const { return _max_inflight_bytes; }

 private:
    std::atomic<bool> _stop;
    ClientConfiguration _client_config;
    const size_t _chunk_bytesize;
    size_t _max_inflight_bytes = 0;

    // Azure credentials
    std::optional<std::string> _account_name;
    std::optional<std::string> _account_key;
    std::optional<std::string> _sas_token;
    std::string _endpoint_suffix;
#ifdef AZURITE_TESTING
    std::optional<std::string> _connection_string;
#endif
    
    // Azure Blob Storage client and async wrapper
    std::shared_ptr<Azure::Storage::Blobs::BlobServiceClient> _blob_service_client;
    std::unique_ptr<AsyncAzureClient> _async_client;

    // queue of asynchronous responses
    using Responder = common::SharedQueue<common::backend_api::Response>;
    std::shared_ptr<Responder> _responder;
    std::mutex _responder_mutex;  // Protects _responder access
};

} // namespace runai::llm::streamer::impl::azure
