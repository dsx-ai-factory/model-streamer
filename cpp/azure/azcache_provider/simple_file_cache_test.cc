/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Simple File-Based Cache Provider — TEST ONLY
 *
 * This is a minimal test implementation of the blob_read interface.
 * It serves cached blobs from local files on disk:
 *
 *   <RUNAI_CACHE_DIR>/<container>/<blob>
 *
 * For example, if RUNAI_CACHE_DIR=/mnt/cache, a blob at
 * container "models" path "llama/weights.safetensors" would be cached at:
 *
 *   /mnt/cache/models/llama/weights.safetensors
 *
 * This test implementation does NOT fall back to Azure Blob Storage on a
 * cache miss — it simply returns an error. A production cache provider
 * should fetch from Azure Blob Storage when data is not in the cache,
 * store it locally, and serve the read from the newly cached copy.
 *
 * The cache must be pre-populated before use (e.g., by the test harness).
 *
 * To build:
 *   g++ -shared -fPIC -o libsimple_file_cache_test.so simple_file_cache_test.cc
 *
 * To use:
 *   export RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB=/path/to/libsimple_file_cache_test.so
 *   export RUNAI_CACHE_DIR=/mnt/cache
 */

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char* get_cache_dir(void)
{
    const char* dir = getenv("RUNAI_CACHE_DIR");
    return dir ? dir : "/mnt/cache";
}

static void set_error(char* error_buf, size_t error_buf_size, const char* fmt, ...)
    __attribute__((format(printf, 3, 4)));

static void set_error(char* error_buf, size_t error_buf_size, const char* fmt, ...)
{
    if (error_buf && error_buf_size > 0)
    {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error_buf, error_buf_size, fmt, args);
        va_end(args);
    }
}

/* Reject absolute paths and ".." components to prevent path traversal attacks. */
static int has_path_traversal(const char* s)
{
    if (s[0] == '/')
        return 1;
    const char* p = s;
    while ((p = strstr(p, "..")) != NULL)
    {
        int at_start = (p == s || *(p - 1) == '/');
        int at_end   = (*(p + 2) == '\0' || *(p + 2) == '/');
        if (at_start && at_end) return 1;
        p += 2;
    }
    return 0;
}

static int g_closed = 0;

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
uint32_t cache_abi_version(void)
{
    return 1;
}

__attribute__((visibility("default")))
void shutdown(void)
{
    g_closed = 1;
}

__attribute__((visibility("default")))
int shutdown_called(void)
{
    return g_closed;
}

__attribute__((visibility("default")))
void shutdown_reset(void)
{
    g_closed = 0;
}

__attribute__((visibility("default")))
ssize_t blob_read(
    const char* account,
    const char* container,
    const char* blob,
    void* buf,
    size_t offset,
    size_t length,
    char* error_buf,
    size_t error_buf_size)
{
    if (!account || !container || !blob || !buf)
    {
        set_error(error_buf, error_buf_size, "null argument");
        return -1;
    }

    if (has_path_traversal(container) || has_path_traversal(blob))
    {
        set_error(error_buf, error_buf_size, "path traversal rejected");
        return -1;
    }

    /* Build path: <cache_dir>/<container>/<blob> */
    const char* cache_dir = get_cache_dir();
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s/%s", cache_dir, container, blob);
    if (n < 0 || (size_t)n >= sizeof(path))
    {
        set_error(error_buf, error_buf_size, "cache path too long");
        return -1;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        set_error(error_buf, error_buf_size, "open failed for '%s': %s", path, strerror(errno));
        return -1;
    }

    ssize_t bytes_read = pread(fd, buf, length, (off_t)offset);
    if (bytes_read < 0)
    {
        set_error(error_buf, error_buf_size, "pread failed for '%s': %s", path, strerror(errno));
        close(fd);
        return -1;
    }

    close(fd);

    if ((size_t)bytes_read != length)
    {
        set_error(error_buf, error_buf_size,
                 "short read for '%s': expected %zu bytes, got %zd",
                 path, length, bytes_read);
        return -1;
    }

    return bytes_read;
}

#ifdef __cplusplus
}
#endif
