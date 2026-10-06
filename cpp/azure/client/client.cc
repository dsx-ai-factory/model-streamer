/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <string>
#include <utility>
#include <optional>
#include <vector>
#include <future>
#include <mutex>
#include <memory>
#include <functional>
#include <cstring>

#include <azure/storage/blobs.hpp>
#include <azure/identity/default_azure_credential.hpp>
#include <azure/storage/common/storage_credential.hpp>
#include <azure/core/exception.hpp>

#include "azure/client/client.h"
#include "azure/azcache_provider/azcache_provider_loader.h"
#include "common/exception/exception.h"
#include "utils/logging/logging.h"
#include "utils/env/env.h"

using namespace Azure::Storage::Blobs;

namespace runai::llm::streamer::impl::azure
{

constexpr char kAzureApplicationId[] = "azpartner-runai";

AzureClient::AzureClient(const common::backend_api::ObjectClientConfig_t& config) :
    _stop(false),
    _client_config(config.concurrent_readers),
    _responder(nullptr),
    _chunk_bytesize(config.default_storage_chunk_size)
{
    // advertise an in-flight window sized to the async threadpool: margin x pool x chunk,
    // floored at two chunks so the window can always keep the pool busy
    _max_inflight_bytes = std::max(
        static_cast<size_t>(common::backend_api::inflight_window_margin() * _client_config.max_concurrency * _chunk_bytesize),
        static_cast<size_t>(2) * _chunk_bytesize);

    // ClientConfiguration reads environment variables
    _account_name = _client_config.account_name;
    _account_key = _client_config.account_key;
    _sas_token = _client_config.sas_token;
    _endpoint_suffix = _client_config.endpoint_suffix;
#ifdef AZURITE_TESTING
    _connection_string = _client_config.connection_string;
#endif

    // Parse configuration parameters from API (overrides environment)
    // Empty values are treated as unset to avoid overriding env vars with no-ops
    auto ptr = config.initial_params;
    if (ptr)
    {
        for (size_t i = 0; i < config.num_initial_params; ++i, ++ptr)
        {
            const char* key = ptr->key;
            const char* value = ptr->value;

            if (!value || strlen(value) == 0)
            {
                continue;
            }

            if (strcmp(key, "account_name") == 0)
            {
                _account_name = std::string(value);
            }
            else if (strcmp(key, "account_key") == 0)
            {
                _account_key = std::string(value);
            }
            else if (strcmp(key, "sas_token") == 0)
            {
                _sas_token = std::string(value);
            }
            else if (strcmp(key, "endpoint_suffix") == 0)
            {
                _endpoint_suffix = std::string(value);
            }
#ifdef AZURITE_TESTING
            else if (strcmp(key, "connection_string") == 0)
            {
                _connection_string = std::string(value);
            }
#endif
            else
            {
                LOG(WARNING) << "Unknown Azure parameter: " << key;
            }
        }
    }

    // Initialize Azure Blob Storage client
    try {
        BlobClientOptions options;

        // Set application ID for telemetry (prepended to User-Agent)
        // Reference: https://azure.github.io/azure-sdk/general_azurecore.html#user-agent-format
        options.Telemetry.ApplicationId = kAzureApplicationId;

        // Using Azure SDK defaults for retry policy
        // Reference: https://learn.microsoft.com/en-us/azure/storage/common/storage-retry-policy

        LOG(DEBUG) << "Azure client concurrency: " << _client_config.max_concurrency;

#ifdef AZURITE_TESTING
        if (_connection_string.has_value()) {
            // Connection string authentication (for Azurite testing only)
            // For Azurite, use --skipApiVersionCheck flag instead of setting API version
            // Reference: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite
            _blob_service_client = std::make_shared<BlobServiceClient>(
                BlobServiceClient::CreateFromConnectionString(_connection_string.value(), options)
            );
            LOG(DEBUG) << "Azure client initialized with connection string (Azurite testing)";
        } else
#endif
        if (_account_name.has_value()) {
            if (_sas_token.has_value()) {
                // Use SAS token authentication (no credential object needed)
                // The token is appended as query string to the service URL
                std::string token = _sas_token.value();
                if (!token.empty() && token[0] == '?') {
                    token = token.substr(1);
                }
                std::string url = "https://" + _account_name.value() + "." + _endpoint_suffix + "?" + token;
                _blob_service_client = std::make_shared<BlobServiceClient>(url, options);
                LOG(DEBUG) << "Azure client initialized with SAS token for account: " << _account_name.value();
            } else if (_account_key.has_value()) {
                // Use StorageSharedKeyCredential (account name + account key)
                std::string url = "https://" + _account_name.value() + "." + _endpoint_suffix;
                auto credential = std::make_shared<Azure::Storage::StorageSharedKeyCredential>(
                    _account_name.value(), _account_key.value());
                _blob_service_client = std::make_shared<BlobServiceClient>(url, credential, options);
                LOG(DEBUG) << "Azure client initialized with StorageSharedKeyCredential for account: " << _account_name.value();
            } else {
                // Use DefaultAzureCredential (managed identity, Azure CLI, environment variables, etc.)
                // Reference: https://learn.microsoft.com/en-us/azure/developer/cpp/sdk/authentication
                std::string url = "https://" + _account_name.value() + "." + _endpoint_suffix;
                // Share a single DefaultAzureCredential across all clients in the process to better
                // utilize token caching and reduce chances of overwhelming authentication sources
                // (e.g., IMDS) which can result in fatal throttling errors.
                static auto credential = std::make_shared<Azure::Identity::DefaultAzureCredential>();
                _blob_service_client = std::make_shared<BlobServiceClient>(url, credential, options);
                LOG(DEBUG) << "Azure client initialized with DefaultAzureCredential for account: " << _account_name.value();
            }
        } else {
#ifdef AZURITE_TESTING
            LOG(ERROR) << "Azure credentials required. Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME.";
#else
            LOG(ERROR) << "Azure account name is required. Set AZURE_STORAGE_ACCOUNT_NAME environment variable.";
#endif
            throw common::Exception(common::ResponseCode::InvalidParameterError);
        }

        // Create cache provider loader from environment
        auto cache_loader = AzCacheProviderLoader::from_env();

        // Create async client with ThreadPool and cache loader
        _async_client = std::make_unique<AsyncAzureClient>(_blob_service_client, _client_config.max_concurrency, cache_loader);

    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to initialize Azure client: " << e.what();
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }
}

AzureClient::~AzureClient()
{
    stop();
    // AsyncAzureClient destructor will handle cleanup of ThreadPool
}

bool AzureClient::verify_credentials(const common::backend_api::ObjectClientConfig_t & config) const
{
    // Parse config credentials without creating full client. The reader count is irrelevant here -
    // only the credential fields are compared - but it is stated rather than assumed.
    ClientConfiguration temp_config(config.concurrent_readers);
    std::optional<std::string> temp_account_name = temp_config.account_name;
    std::optional<std::string> temp_account_key = temp_config.account_key;
    std::optional<std::string> temp_sas_token = temp_config.sas_token;
    std::string temp_endpoint_suffix = temp_config.endpoint_suffix;
#ifdef AZURITE_TESTING
    std::optional<std::string> temp_connection_string = temp_config.connection_string;
#endif

    // Override with config parameters
    auto ptr = config.initial_params;
    if (ptr) {
        for (size_t i = 0; i < config.num_initial_params; ++i, ++ptr) {
            if (strcmp(ptr->key, "account_name") == 0) temp_account_name = std::string(ptr->value);
            else if (strcmp(ptr->key, "account_key") == 0) temp_account_key = std::string(ptr->value);
            else if (strcmp(ptr->key, "sas_token") == 0) temp_sas_token = std::string(ptr->value);
            else if (strcmp(ptr->key, "endpoint_suffix") == 0) temp_endpoint_suffix = std::string(ptr->value);
#ifdef AZURITE_TESTING
            else if (strcmp(ptr->key, "connection_string") == 0) temp_connection_string = std::string(ptr->value);
#endif
        }
    }

#ifdef AZURITE_TESTING
    if (_connection_string.has_value()) {
        return (_connection_string == temp_connection_string);
    }
#endif
    return (_account_name == temp_account_name) && (_account_key == temp_account_key) && (_sas_token == temp_sas_token) && (_endpoint_suffix == temp_endpoint_suffix);
}

common::backend_api::Response AzureClient::async_read_response()
{
    std::shared_ptr<Responder> responder;
    {
        std::lock_guard<std::mutex> lock(_responder_mutex);
        if (_responder == nullptr)
        {
            LOG(WARNING) << "Requesting response with uninitialized responder";
            return common::ResponseCode::FinishedError;
        }
        responder = _responder;
    }

    return responder->pop();
}

void AzureClient::stop()
{
    _stop = true;
    std::lock_guard<std::mutex> lock(_responder_mutex);
    if (_responder != nullptr)
    {
        _responder->stop();
    }
}

common::ResponseCode AzureClient::async_read(const char* path, 
                                             common::backend_api::ObjectRange_t range, 
                                             char* destination_buffer, 
                                             common::backend_api::ObjectRequestId_t request_id)
{
    std::shared_ptr<Responder> responder;
    {
        std::lock_guard<std::mutex> lock(_responder_mutex);
        if (_responder == nullptr)
        {
            _responder = std::make_shared<Responder>(1);
        }
        else
        {
            _responder->increment(1);
        }
        responder = _responder;
    }

    char * buffer_ = destination_buffer;

    // Split range into chunks.
    // When a cache provider is loaded, issue a single read per range
    // so the provider can handle alignment and prefetching internally.
    size_t num_chunks;
    if (_async_client->is_cache_enabled())
    {
        num_chunks = 1;
    }
    else
    {
        num_chunks = std::max(1UL, range.length / _chunk_bytesize);
    }
    LOG(SPAM) << "Number of chunks is: " << num_chunks;

    // Counter for tracking chunk completions
    auto counter = std::make_shared<std::atomic<unsigned>>(num_chunks);
    auto is_success = std::make_shared<std::atomic<bool>>(true);

    // Parse Azure URI az://container/blob
    const auto uri = common::s3::StorageUri(path);
    std::string container_name(uri.bucket);
    std::string blob_name(uri.path);

    size_t total_ = range.length;
    size_t offset_ = range.offset;

    for (unsigned i = 0; i < num_chunks && !_stop; ++i)
    {
        size_t bytesize_ = (i == num_chunks - 1 ? total_ : _chunk_bytesize);

        // Capture current buffer, offset for this specific chunk
        char* chunk_buffer = buffer_;
        size_t chunk_offset = offset_;
        
        // Launch async download with callback - AsyncAzureClient ThreadPool handles both download and callback
        _async_client->DownloadBlobRangeAsync(
            _account_name.value_or(""),
            container_name,
            blob_name,
            chunk_buffer,
            chunk_offset,
            bytesize_,
            [request_id, counter, is_success, responder](common::ResponseCode response_code, const std::string& error_msg) {
                if (response_code == common::ResponseCode::Success) {
                    const auto running = counter->fetch_sub(1);
                    LOG(SPAM) << "Async read request " << request_id << " succeeded - " << running << " running";
                    
                    if (running == 1) {
                        common::backend_api::Response r(request_id, common::ResponseCode::Success);
                        responder->push(std::move(r));
                    }
                } else {
                    LOG(ERROR) << "Failed to download Azure blob of request " << request_id << ": " << error_msg;
                    bool previous = is_success->exchange(false);
                    if (previous) {
                        // Propagate the specific error code from Azure SDK
                        common::backend_api::Response r(request_id, response_code);
                        responder->push(std::move(r));
                    }
                }
            }
        );
        
        total_ -= bytesize_;
        offset_ += bytesize_;
        buffer_ += bytesize_;
    }

    return _stop ? common::ResponseCode::FinishedError : common::ResponseCode::Success;
}

common::ResponseCode AzureClient::list_files(
    const char* prefix,
    int is_recursive,
    std::vector<std::pair<std::string, size_t>>& results)
{
    const auto uri = common::s3::StorageUri(prefix);
    const std::string uri_prefix = uri.scheme + "://" + uri.bucket + "/";

    auto container_client = _blob_service_client->GetBlobContainerClient(uri.bucket);

    ListBlobsOptions options;
    if (!uri.path.empty())
    {
        options.Prefix = uri.path;
    }

    auto process_page = [&](const auto& page_result)
    {
        for (const auto& blob : page_result.Blobs)
        {
            if (!blob.Name.empty() && blob.Name.back() == '/')
            {
                continue;
            }
            results.emplace_back(uri_prefix + blob.Name, static_cast<size_t>(blob.BlobSize));
        }
    };

    if (is_recursive)
    {
        for (auto page = container_client.ListBlobs(options); page.HasPage(); page.MoveToNextPage())
        {
            process_page(page);
        }
    }
    else
    {
        for (auto page = container_client.ListBlobsByHierarchy("/", options); page.HasPage(); page.MoveToNextPage())
        {
            process_page(page);
        }
    }

    return common::ResponseCode::Success;
}

} // namespace runai::llm::streamer::impl::azure
