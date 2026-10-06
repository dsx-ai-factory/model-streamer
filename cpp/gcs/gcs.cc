/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gcs/gcs.h"
#include "gcs/client/client.h"
#include "common/backend_api/object_storage/list_files_impl.h"
#include "common/client_mgr/client_mgr.h"

#include <atomic>
#include <cstdio>
#include <cstring>

#include "common/exception/exception.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl::gcs
{

inline constexpr char GCSClientName[] = "GCS";
using GCSClientMgr = common::ClientMgr<GCSClient, GCSClientName>;

// In-flight window advertised via obj_get_backend_config("max_inflight_bytes"). Populated
// when a client is created (GCSClient computes it from its async threadpool size and
// default_storage_chunk_size). Uniform across clients in a process; SIZE_MAX = unset/unbounded.
static std::atomic<size_t> g_max_inflight_bytes{static_cast<size_t>(-1)};

common::backend_api::ResponseCode_t obj_open_backend(common::backend_api::ObjectBackendHandle_t* out_backend_handle)
{
    // google-cloud-cpp SDK does not require any global initiaization.
    return common::ResponseCode::Success;
}

common::backend_api::ResponseCode_t obj_close_backend(common::backend_api::ObjectBackendHandle_t backend_handle)
{
    return common::ResponseCode::Success;
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

common::backend_api::ResponseCode_t obj_create_client(common::backend_api::ObjectBackendHandle_t backend_handle,
                                                       const common::backend_api::ObjectClientConfig_t* client_initial_config,
                                                       common::backend_api::ObjectClientHandle_t* out_client_handle)
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        *out_client_handle = GCSClientMgr::pop(*client_initial_config);
        g_max_inflight_bytes.store(static_cast<GCSClient *>(*out_client_handle)->max_inflight_bytes());
    }
    catch(const common::Exception & e)
    {
        ret = e.error();
        *out_client_handle = nullptr;
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to create GCS client";
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
           GCSClientMgr::push(static_cast<GCSClient *>(client_handle));
        }
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to remove GCS client";
        ret = common::ResponseCode::UnknownError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_remove_all_clients()
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        GCSClientMgr::clear();
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to remove all GCS clients";
        ret = common::ResponseCode::UnknownError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_cancel_all_reads()
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        GCSClientMgr::stop();
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to stop all GCS clients";
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
    try
    {
        if (!client_handle)
        {
            LOG(ERROR) << "Attempt to read with null gcs client";
            return common::ResponseCode::UnknownError;
        }
        auto ptr = static_cast<GCSClient *>(client_handle);
        return ptr->async_read(path, range, destination_buffer, request_id);
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while sending async request";
    }
    return common::ResponseCode::UnknownError;
}

common::backend_api::ResponseCode_t obj_wait_for_completions(common::backend_api::ObjectClientHandle_t client_handle,
                                                              common::backend_api::ObjectCompletionEvent_t* event_buffer,
                                                              unsigned int max_events_to_retrieve,
                                                              unsigned int* out_num_events_retrieved,
                                                              common::backend_api::ObjectWaitMode_t wait_mode)
{
    try
    {
        if (!client_handle)
        {
            LOG(ERROR) << "Attempt to get read response with null GCS client";
            return common::ResponseCode::UnknownError;
        }
        if (max_events_to_retrieve == 0)
        {
            LOG(ERROR) << "Attempt to get read response with max_events_to_retrieve = 0";
            return common::ResponseCode::UnknownError;
        }
        if (!event_buffer || !out_num_events_retrieved)
        {
            LOG(ERROR) << "Attempt to get read response with null event_buffer or out_num_events_retrieved";
            return common::ResponseCode::UnknownError;
        }

        // for now reads a single event

        auto ptr = static_cast<GCSClient *>(client_handle);
        auto response = ptr->async_read_response();
        *out_num_events_retrieved = 1;
        event_buffer[0].request_id = response.handle;
        event_buffer[0].response_code = response.ret;
        return common::ResponseCode::Success;
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while sending async request";
    }
    return common::ResponseCode::UnknownError;
}

common::backend_api::ResponseCode_t obj_list_files(
    common::backend_api::ObjectClientHandle_t client_handle,
    const char* prefix,
    int is_recursive,
    common::backend_api::ObjectFileEntry_t** out_entries,
    unsigned* out_num_entries)
{
    return common::backend_api::impl_obj_list_files<GCSClient>(
        client_handle, prefix, is_recursive, out_entries, out_num_entries);
}

void obj_free_file_list(
    common::backend_api::ObjectFileEntry_t* entries,
    unsigned num_entries)
{
    common::backend_api::impl_obj_free_file_list(entries, num_entries);
}

}; // namespace runai::llm::streamer::impl::gcs