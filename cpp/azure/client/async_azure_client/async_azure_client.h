/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <future>
#include <memory>
#include <string>
#include <functional>

#include <azure/storage/blobs.hpp>

#include "common/response_code/response_code.h"
#include "utils/logging/logging.h"
#include "utils/threadpool/threadpool.h"
#include "azure/azcache_provider/azcache_provider_loader.h"

namespace runai::llm::streamer::impl::azure
{

typedef std::function<void()> DownloadBlobFn;
typedef std::function<void(common::ResponseCode response_code, const std::string& error_msg)> CompletionCallback;

struct DownloadBlobTask {
    DownloadBlobFn taskFn;
    CompletionCallback callback;

public:
    void execute();
};

inline DownloadBlobFn createDownloadBlobFn(
    std::shared_ptr<Azure::Storage::Blobs::BlobServiceClient> client,
    const std::string& container_name,
    const std::string& blob_name,
    char* buffer,
    size_t offset,
    size_t length)
{
    // Explicitly copy strings to ensure they're captured properly
    std::string container = container_name;
    std::string blob = blob_name;
    
    return [client, container, blob, buffer, offset, length]() {
        using namespace Azure::Storage::Blobs;
        auto container_client = client->GetBlobContainerClient(container);
        auto blob_client = container_client.GetBlockBlobClient(blob);
        
        DownloadBlobToOptions download_options;
        download_options.Range = Azure::Core::Http::HttpRange();
        download_options.Range.Value().Offset = offset;
        download_options.Range.Value().Length = length;
        
        blob_client.DownloadTo(
            reinterpret_cast<uint8_t*>(buffer), length, download_options
        );
    };
}

/**
 * An async wrapper for Azure Blob Storage client operations.
 * Uses ThreadPool to manage concurrent blob download operations.
 * 
 * When a cache provider is configured, the client will read from
 * the cache instead of Azure Blob Storage.
 */
struct AsyncAzureClient
{
public:
    AsyncAzureClient(std::shared_ptr<Azure::Storage::Blobs::BlobServiceClient> client,
                     unsigned max_pool_size,
                     std::shared_ptr<AzCacheProviderLoader> cache_loader = nullptr);

    /** Returns true when a cache provider .so has been loaded. */
    bool is_cache_enabled() const { return _cache_enabled; }

    /**
     * Download a range of a blob asynchronously.
     * 
     * When a cache provider is loaded, all reads are served from the cache.
     * The cache provider is responsible for handling cache misses internally.
     */
    void DownloadBlobRangeAsync(
        const std::string& account_name,
        const std::string& container_name,
        const std::string& blob_name,
        char* buffer,
        size_t offset,
        size_t length,
        CompletionCallback callback)
    {
        if (_cache_enabled)
        {
            // Capture shared_ptr to loader to ensure lifetime
            auto loader = _cache_loader;
            DownloadBlobTask task{
                [loader, account_name, container_name, blob_name, buffer, offset, length]() {
                    bool ok = loader->read(
                        account_name, container_name, blob_name, buffer, offset, length);
                    if (!ok)
                    {
                        throw std::runtime_error("Cache provider read failed");
                    }
                },
                std::move(callback)
            };
            push_task(std::move(task));
        }
        else
        {
            DownloadBlobFn downloadFn = createDownloadBlobFn(_client, container_name, blob_name, buffer, offset, length);

            DownloadBlobTask task{
                std::move(downloadFn),
                std::move(callback)
            };

            push_task(std::move(task));
        }
    }

private:
    void push_task(DownloadBlobTask&& task);

    std::shared_ptr<Azure::Storage::Blobs::BlobServiceClient> _client;
    utils::ThreadPool<DownloadBlobTask> _pool;
    std::shared_ptr<AzCacheProviderLoader> _cache_loader;
    const bool _cache_enabled;
};

}; // namespace runai::llm::streamer::impl::azure
