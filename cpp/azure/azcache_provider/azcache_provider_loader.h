/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "azure/azcache_provider/runai_azcache_provider.h"

namespace runai::llm::streamer::impl::azure
{

/**
 * Cache mode matching RUNAI_STREAMER_DIST pattern.
 */
enum class CacheMode
{
    Disabled,  // "0" — never use cache
    Required,  // "1" — must load cache, fail if not found
    Auto       // "auto" or unset — use if available, silently disable otherwise
};

/**
 * Configuration for constructing a cache provider loader.
 */
struct CacheProviderConfig
{
    std::string lib_path;  // Path to the cache provider .so (empty = not configured)
    CacheMode mode = CacheMode::Auto;
};

/**
 * Shared handle to the dlopen'd cache provider library.
 * Loaded once, shared across all AzureClient instances via weak_ptr.
 * When the last strong reference is released, calls shutdown() (if exported)
 * to gracefully shut down the provider's resources.
 */
struct CacheLibHandle
{
    CacheLibHandle(const std::string& path, CacheMode mode);
    ~CacheLibHandle();

    CacheLibHandle(const CacheLibHandle&) = delete;
    CacheLibHandle& operator=(const CacheLibHandle&) = delete;

    void* lib_handle;
    blob_read_fn read_fn;
    shutdown_fn close_fn;  // may be nullptr if not exported
    std::string lib_path;
};

/**
 * AzCacheProviderLoader dynamically loads a cache provider .so at runtime
 * via dlopen/dlsym.
 *
 * The library handle is shared across all loaders via a process-wide weak_ptr.
 * The first loader to create the handle loads the library; when the last loader
 * is destroyed, the handle destructor calls shutdown() for graceful shutdown.
 *
 * Use from_env() to create from environment variables.
 *
 * Thread-safe once constructed (immutable state after construction).
 */
class AzCacheProviderLoader
{
public:
    /**
     * Create a loader from environment variables:
     *   RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED: "0", "1", or "auto" (default)
     *   RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB: explicit .so path override
     *
     * @throws common::Exception if mode=Required and library cannot be loaded.
     */
    static std::shared_ptr<AzCacheProviderLoader> from_env();

    /**
     * Create a loader with explicit configuration (for testing).
     */
    explicit AzCacheProviderLoader(const CacheProviderConfig& config);
    ~AzCacheProviderLoader();

    bool is_enabled() const { return _enabled; }

    /**
     * Read blob data through the cache provider.
     *
     * @param account    Azure Storage account name
     * @param container  Azure container name
     * @param blob       Blob path within the container
     * @param buffer     Destination buffer (caller-allocated)
     * @param offset     Byte offset within the blob
     * @param length     Number of bytes to read
     * @return true on success (length bytes read), false on error
     */
    bool read(const std::string& account,
              const std::string& container,
              const std::string& blob,
              char* buffer,
              size_t offset,
              size_t length);

    AzCacheProviderLoader(const AzCacheProviderLoader&) = delete;
    AzCacheProviderLoader& operator=(const AzCacheProviderLoader&) = delete;

private:
    std::shared_ptr<CacheLibHandle> _handle;
    bool _enabled;

    static std::weak_ptr<CacheLibHandle> s_shared_handle;
    static std::mutex s_handle_mutex;
};

} // namespace runai::llm::streamer::impl::azure
