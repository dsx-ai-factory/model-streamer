/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * RunAI Azure Cache Provider Interface (Experimental)
 *
 * This header defines the contract for Azure Blob Storage cache providers.
 * A cache provider is a shared library (.so) that exports the blob_read symbol.
 * When installed as a Python package alongside runai-model-streamer, the cache
 * provider is auto-discovered and loaded at runtime — no configuration needed.
 *
 * Control via environment variables:
 *   RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED=0    — disable cache entirely
 *   RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED=1    — require cache (fail if not found)
 *   RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED=auto — use if available (default)
 *
 * The cache provider is responsible for:
 *   - Serving cached blob data when available
 *   - Populating the cache (e.g., fetching from Azure Blob Storage on miss)
 *   - Managing its own lifecycle and resources
 *
 * RunAI is NOT responsible for populating the cache.
 *
 * Example usage in a custom cache provider:
 *
 *   #include "runai_azcache_provider.h"
 *
 *   extern "C" uint32_t cache_abi_version(void) {
 *       return RUNAI_CACHE_ABI_VERSION;
 *   }
 *
 *   extern "C" ssize_t blob_read(
 *       const char* account, const char* container,
 *       const char* blob,
 *       void* buf, size_t offset, size_t length,
 *       char* error_buf, size_t error_buf_size)
 *   {
 *       // Your cache logic here
 *       // On error: snprintf(error_buf, error_buf_size, "details...");
 *       return length;  // bytes read on success, -1 on error
 *   }
 *
 *   extern "C" void shutdown(void) {
 *       // Optional: graceful shutdown — stop threads, close connections.
 *       // Called once when the streamer no longer needs the cache provider.
 *   }
 */

#ifndef RUNAI_STREAMER_AZCACHE_PROVIDER_H
#define RUNAI_STREAMER_AZCACHE_PROVIDER_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Recommended error buffer size for callers */
#define RUNAI_CACHE_ERROR_BUF_SIZE 512

/*
 * Read a range of bytes from a cached Azure blob.
 *
 * Parameters:
 *   account        - Azure Storage account name
 *   container      - Azure Blob Storage container name
 *   blob           - Blob path within the container
 *   buf            - Output buffer (caller-allocated, at least 'length' bytes)
 *   offset         - Byte offset within the blob to start reading from
 *   length         - Number of bytes to read
 *   error_buf      - Caller-owned buffer for error message (NUL-terminated on error).
 *                    May be NULL if error_buf_size is 0 (errors are silently discarded).
 *                    Message may be truncated if buffer is too small.
 *   error_buf_size - Size of the error_buf in bytes
 *
 * Returns:
 *   Number of bytes read on success (must equal 'length' for success).
 *   -1 on error (error_buf will contain a NUL-terminated message if provided).
 *
 * Contract:
 *   - A return value != length is treated as an error by the caller.
 *   - Providers must not return partial reads as success; return -1 instead.
 */
typedef ssize_t (*blob_read_fn)(
    const char* account,
    const char* container,
    const char* blob,
    void* buf,
    size_t offset,
    size_t length,
    char* error_buf,
    size_t error_buf_size);

/* Symbol name that the cache provider .so must export */
#define BLOB_READ_SYMBOL "blob_read"

/*
 * Optional: graceful shutdown of the cache provider.
 *
 * Called once when the streamer no longer needs the cache provider.
 * The provider should stop background threads, close connections,
 * and release resources. After shutdown() returns, no further
 * blob_read() calls will be made.
 *
 * If the provider does not export this symbol, shutdown is skipped
 * and resources are released at process exit by the OS.
 *
 * Contract:
 *   - Called at most once per loaded library handle.
 *   - Must not throw.
 *   - Must be safe to call after all blob_read() calls have completed.
 *   - Must tolerate partial initialization (e.g., if constructor failed midway).
 */
typedef void (*shutdown_fn)(void);

#define SHUTDOWN_SYMBOL "shutdown"

/*
 * ABI versioning — guards against incompatible .so after interface changes.
 *
 * The cache provider .so must export cache_abi_version() returning
 * RUNAI_CACHE_ABI_VERSION. The loader rejects libraries that do not export
 * the symbol or return an unsupported version.
 */
#define RUNAI_CACHE_ABI_VERSION 1
#define RUNAI_CACHE_ABI_VERSION_SYMBOL "cache_abi_version"

typedef uint32_t (*runai_cache_abi_version_fn)(void);

/* Environment variable to control cache mode: "0", "1", or "auto" */
#define RUNAI_AZURE_CACHE_ENABLED_ENV "RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_ENABLED"

/* Environment variable pointing to the cache provider .so path (explicit override) */
#define RUNAI_AZURE_CACHE_LIB_ENV "RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB"

#ifdef __cplusplus
}
#endif

#endif /* RUNAI_STREAMER_AZCACHE_PROVIDER_H */
