/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/s3.h"
#include "s3/s3_init/s3_init.h"
#include "s3/client/client.h"

#include <atomic>
#include <cstdio>
#include <cstring>

#include "common/backend_api/object_storage/list_files_impl.h"
#include "common/client_mgr/client_mgr.h"
#include "common/exception/exception.h"
#include "utils/env/env.h"
#include "utils/semver/semver.h"

// For connecting to s3 providers other then aws:
// 1. uri should be in the format s3://bucket/path
// 2. endpoint url must be provided with environment variable AWS_ENDPOINT_URL
// 3. set environment variable AWS_EC2_METADATA_DISABLED = true
// 4. set RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING = 0
// 5. Credentials can be provided in ~/.aws/credentials file or by passing environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

// e.g. gcs: AWS_ENDPOINT_URL="https://storage.googleapis.com" RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_EC2_METADATA_DISABLED=true <streamer app> s3://bucket/path
// e.g. minio: AWS_ENDPOINT_URL="http://localhost:9000" RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_EC2_METADATA_DISABLED=true AWS_ACCESS_KEY_ID="minio" AWS_SECRET_ACCESS_KEY="miniostorage"

// Important: for s3 compatibale (e.g. minio) run the streamer with AWS_EC2_METADATA_DISABLED=true - see https://github.com/aws/aws-sdk-cpp/issues/1410
// This is because the Aws::ClientConfiguration c'tor tries to retrieve the region from the aws server, which causes a 5 seconds delay
// The ClientConfigurationProvider needs to be created before creating ClientConfiguration and S3Client in order to set the environment variable
// Then the ClientConfigurationProvider object should be destructed for restoring the environment variable
// Note: this workaround should be removed if the Aws::S3Crt::ClientConfiguration will accept shouldDisableIMDS

namespace runai::llm::streamer::impl::s3
{

inline constexpr char S3ClientName[] = "S3";
using S3ClientMgr = common::ClientMgr<S3Client, S3ClientName>;

// In-flight window advertised via obj_get_backend_config("max_inflight_bytes"). Populated
// when a client is created (S3Client computes it from default_storage_chunk_size and
// throughputTargetGbps). Uniform across clients in a process; SIZE_MAX = unset/unbounded.
static std::atomic<size_t> g_max_inflight_bytes{static_cast<size_t>(-1)};

// --- Backend API ---

const utils::Semver min_glibc_semver = utils::Semver(common::description(static_cast<int>(common::ResponseCode::GlibcPrerequisite)));
const size_t min_chunk_bytesize = 5 * 1024 * 1024;

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
            LOG_IF(INFO, (chunk_size < min_chunk_bytesize)) << "Minimal chunk size to read from S3 is 5 MiB";
        }

        // shutdown of S3 will occur when the main process exits, and the s3 library is destructed
        // attemting to shutdown outside the s3 library cause race and segmantation fault e.g. when the s3 trace logs are enabled
        static S3Init s3_init;
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to init S3 backend";
        ret = common::ResponseCode::S3NotSupported;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_close_backend(common::backend_api::ObjectBackendHandle_t backend_handle)
{
    // shutdown of S3 will occur when the main process exits
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
        *out_client_handle = S3ClientMgr::pop(*client_initial_config);
        g_max_inflight_bytes.store(static_cast<S3Client *>(*out_client_handle)->max_inflight_bytes());
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to create S3 client";
        ret = common::ResponseCode::UnknownError;
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
           S3ClientMgr::push(static_cast<S3Client *>(client_handle));
        }
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to remove S3 client";
        ret = common::ResponseCode::UnknownError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_remove_all_clients()
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        S3ClientMgr::clear();
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to remove all S3 clients";
        ret = common::ResponseCode::UnknownError;
    }
    return ret;
}

common::backend_api::ResponseCode_t obj_cancel_all_reads()
{
    common::ResponseCode ret = common::ResponseCode::Success;
    try
    {
        S3ClientMgr::stop();
    }
    catch(const std::exception & e)
    {
        LOG(ERROR) << "Failed to stop all S3 clients";
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
            LOG(ERROR) << "Attempt to read with null s3 client";
            return common::ResponseCode::UnknownError;
        }
        auto ptr = static_cast<S3Client *>(client_handle);
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
            LOG(ERROR) << "Attempt to get read response with null s3 client";
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

        auto ptr = static_cast<S3Client *>(client_handle);
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
    return common::backend_api::impl_obj_list_files<S3Client>(
        client_handle, prefix, is_recursive, out_entries, out_num_entries);
}

void obj_free_file_list(
    common::backend_api::ObjectFileEntry_t* entries,
    unsigned num_entries)
{
    common::backend_api::impl_obj_free_file_list(entries, num_entries);
}

}; // namespace runai::llm::streamer::impl::s3
