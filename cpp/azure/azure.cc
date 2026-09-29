/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "azure/azure.h"
#include "azure/client/client.h"

#include <atomic>
#include <cstdio>
#include <cstring>

#include "common/backend_api/object_storage/list_files_impl.h"
#include "common/client_mgr/client_mgr.h"
#include "common/exception/exception.h"
#include "utils/env/env.h"
#include "utils/logging/logging.h"
#include "utils/semver/semver.h"

// For connecting to Azure Blob Storage:
// 1. uri should be in the format az://container/path
// 2. Credentials can be provided via environment variables or config parameters
// 3. See azure.h for detailed credential configuration options

namespace runai::llm::streamer::impl::azure
{

inline constexpr char AzureClientName[] = "Azure";
using AzureClientMgr = common::ClientMgr<AzureClient, AzureClientName>;

// In-flight window advertised via obj_get_backend_config("max_inflight_bytes"). Populated
// when a client is created (AzureClient computes it from its async threadpool size and
// default_storage_chunk_size). Uniform across clients in a process; SIZE_MAX = unset/unbounded.
static std::atomic<size_t> g_max_inflight_bytes{static_cast<size_t>(-1)};

// --- Backend API ---

const utils::Semver min_glibc_semver = utils::Semver(common::description(static_cast<int>(common::ResponseCode::GlibcPrerequisite)));
// Minimum chunk size aligned with S3 backend for consistency
// Azure Blob Storage supports ranged GETs down to single bytes, but larger chunks improve throughput
const size_t min_chunk_bytesize = 5 * 1024 * 1024; // 5 MiB (same as S3)

common::backend_api::ResponseCode_t obj_open_backend(common::backend_api::ObjectBackendHandle_t* out_backend_handle)
{
    common::ResponseCode ret = common::ResponseCode::Success;

    try
    {
        // verify prerequisites
        auto glibc_version = utils::get_glibc_version();
        if (min_glibc_semver > glibc_version)
        {
            LOG(ERROR) << "GLIBC version must be at least " << min_glibc_semver << ", instead of " << glibc_version;
            return common::ResponseCode::GlibcPrerequisite;
        }

        size_t chunk_size;
        if (utils::try_getenv("RUNAI_STREAMER_CHUNK_BYTESIZE", chunk_size))
        {
            LOG_IF(INFO, (chunk_size < min_chunk_bytesize)) << "Minimal chunk size to read from Azure is 5 MiB";
        }

        // Azure SDK doesn't require global initialization like AWS SDK
        LOG(INFO) << "Azure backend initialized successfully";
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to init Azure backend: " << e.what();
        ret = common::ResponseCode::FileAccessError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_close_backend(common::backend_api::ObjectBackendHandle_t backend_handle)
{
    // Azure SDK cleanup is automatic
    common::ResponseCode ret = common::ResponseCode::Success;
    return ret;
}

common::backend_api::ObjectShutdownPolicy_t obj_get_backend_shutdown_policy()
{
    return common::backend_api::OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT;
}

common::backend_api::ResponseCode_t obj_get_backend_config(common::backend_api::ObjectBackendHandle_t /* backend_handle */,
                                                           const char* key,
                                                           char* out_value_buffer,
                                                           unsigned int* in_out_buffer_len)
{
    try
    {
        if (!key || !out_value_buffer || !in_out_buffer_len)
        {
            LOG(ERROR) << "obj_get_backend_config called with null argument(s)";
            return common::ResponseCode::UnknownError;
        }

        if (std::strcmp(key, "max_inflight_bytes") == 0)
        {
            char tmp[32];
            const int n = std::snprintf(tmp, sizeof(tmp), "%zu", g_max_inflight_bytes.load());
            if (n < 0 || n >= static_cast<int>(sizeof(tmp)))
            {
                LOG(ERROR) << "Failed to format max_inflight_bytes value";
                return common::ResponseCode::UnknownError;
            }
            const unsigned int needed = static_cast<unsigned int>(n) + 1;   // including null terminator
            if (*in_out_buffer_len < needed)
            {
                LOG(ERROR) << "obj_get_backend_config buffer too small for key '" << key
                           << "': need " << needed << " bytes, have " << *in_out_buffer_len;
                *in_out_buffer_len = needed;   // report required size to the caller
                return common::ResponseCode::UnknownError;
            }
            std::memcpy(out_value_buffer, tmp, needed);
            *in_out_buffer_len = static_cast<unsigned int>(n);   // written length, excluding null terminator
            return common::ResponseCode::Success;
        }

        LOG(WARNING) << "Unknown backend config key: " << key;
        return common::ResponseCode::UnknownError;
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << "Caught exception in obj_get_backend_config: " << e.what();
    }
    return common::ResponseCode::UnknownError;
}

// --- Client API ---

common::backend_api::ResponseCode_t obj_create_client(common::backend_api::ObjectBackendHandle_t backend_handle,
                                                       const common::backend_api::ObjectClientConfig_t* client_initial_config,
                                                       common::backend_api::ObjectClientHandle_t* out_client_handle)
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        *out_client_handle = AzureClientMgr::pop(*client_initial_config);
        g_max_inflight_bytes.store(static_cast<AzureClient *>(*out_client_handle)->max_inflight_bytes());
    }
    catch(const common::Exception & e)
    {
        ret = e.error();
        *out_client_handle = nullptr;
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to create Azure client: " << e.what();
        ret = common::ResponseCode::FileAccessError;
        *out_client_handle = nullptr;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_remove_client(common::backend_api::ObjectClientHandle_t client_handle)
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        if (client_handle)
        {
            AzureClientMgr::push(static_cast<AzureClient *>(client_handle));
        }
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to remove Azure client: " << e.what();
        ret = common::ResponseCode::UnknownError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_remove_all_clients()
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        AzureClientMgr::clear();
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to remove all Azure clients: " << e.what();
        ret = common::ResponseCode::UnknownError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_cancel_all_reads()
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        AzureClientMgr::stop();
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to stop all Azure clients: " << e.what();
        ret = common::ResponseCode::UnknownError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_request_read(common::backend_api::ObjectClientHandle_t client_handle,
                                                      const char* path,
                                                      common::backend_api::ObjectRange_t range,
                                                      char* destination_buffer,
                                                      common::backend_api::ObjectRequestId_t request_id)
{
    auto client = static_cast<AzureClient *>(client_handle);
    if (client == nullptr)
    {
        LOG(ERROR) << "Azure client handle is null";
        return common::ResponseCode::InvalidParameterError;
    }

    return client->async_read(path, range, destination_buffer, request_id);
}

common::backend_api::ResponseCode_t obj_wait_for_completions(common::backend_api::ObjectClientHandle_t client_handle,
                                                             common::backend_api::ObjectCompletionEvent_t* event_buffer,
                                                             unsigned int max_events_to_retrieve,
                                                             unsigned int* out_num_events_retrieved,
                                                             common::backend_api::ObjectWaitMode_t wait_mode)
{
    auto client = static_cast<AzureClient *>(client_handle);
    if (client == nullptr)
    {
        LOG(ERROR) << "Azure client handle is null";
        return common::ResponseCode::InvalidParameterError;
    }

    *out_num_events_retrieved = 0;

    for (unsigned int i = 0; i < max_events_to_retrieve; ++i)
    {
        auto response = client->async_read_response();
        
        // Always store the response in the event buffer (matching S3/GCS behavior)
        event_buffer[i].request_id = response.handle;
        event_buffer[i].response_code = response.ret;
        event_buffer[i].bytes_transferred = 0;
        (*out_num_events_retrieved)++;
        
        // Break after storing FinishedError so caller can see it
        if (response.ret == common::ResponseCode::FinishedError)
        {
            break;
        }

        // For non-blocking mode, exit after first attempt
        if (wait_mode == common::backend_api::OBJECT_WAIT_MODE_NON_BLOCKING && i == 0)
        {
            break;
        }
    }

    return common::ResponseCode::Success;
}

common::backend_api::ResponseCode_t obj_list_files(
    common::backend_api::ObjectClientHandle_t client_handle,
    const char* prefix,
    int is_recursive,
    common::backend_api::ObjectFileEntry_t** out_entries,
    unsigned* out_num_entries)
{
    return common::backend_api::impl_obj_list_files<AzureClient>(
        client_handle, prefix, is_recursive, out_entries, out_num_entries);
}

void obj_free_file_list(
    common::backend_api::ObjectFileEntry_t* entries,
    unsigned num_entries)
{
    common::backend_api::impl_obj_free_file_list(entries, num_entries);
}

}; // namespace runai::llm::streamer::impl::azure
