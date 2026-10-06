/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/backend_api/object_storage/object_storage.h"
#include "common/response/response.h"
#include "common/range/range.h"

// For connecting to s3 providers other then aws:
//
// 1. Uri should be in the format s3://bucket/path
//
// 2. Provide endpoint url only with environment variable AWS_ENDPOINT_URL
//    Note: the aws cpp sdk ignores the endpoint url in ~/.aws/config file
//    No need to set the endpoint for aws, if using the default endpoint
//
// 3. Set RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING = false
//    This parameter controls the parsing of the URL
//    By setting this option, you control whether the SDK uses virtual-hosted–style URLs (i.e., bucketname.s3.amazonaws.com) or path-style URLs (i.e., s3.amazonaws.com/bucketname).
//
// 4. Set environment variable AWS_EC2_METADATA_DISABLED = true
//    See https://github.com/aws/aws-sdk-cpp/issues/1410
//    This is because the Aws::ClientConfiguration c'tor tries to retrieve the region from the aws server, which causes a 5 seconds delay
//    The ClientConfigurationProvider needs to be created before creating ClientConfiguration and S3Client in order to set the environment variable
//    Then the ClientConfigurationProvider object should be destructed for restoring the environment variable
//    Note: this workaround should be removed if the Aws::S3Crt::ClientConfiguration will accept shouldDisableIMDS
//
// 5. Credentials can be provided in ~/.aws/credentials file or by passing environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
//
// aws:   <streamer app> s3://bucket/path
// gcs:   AWS_ENDPOINT_URL="https://storage.googleapis.com" RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=false AWS_EC2_METADATA_DISABLED=true <streamer app> s3://bucket/path
// seaweedfs: AWS_ENDPOINT_URL="http://localhost:9000" RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_EC2_METADATA_DISABLED=true AWS_ACCESS_KEY_ID="testkey" AWS_SECRET_ACCESS_KEY="testsecret" <streamer app> s3://bucket/path

namespace runai::llm::streamer::impl::s3
{

// --- Backend API ---

extern "C" common::backend_api::ResponseCode_t obj_open_backend(common::backend_api::ObjectBackendHandle_t* out_backend_handle);
extern "C" common::backend_api::ResponseCode_t obj_close_backend(common::backend_api::ObjectBackendHandle_t backend_handle);
extern "C" common::backend_api::ObjectShutdownPolicy_t obj_get_backend_shutdown_policy();

// Query a backend-wide configuration value by key. Currently supports
// "max_inflight_bytes" (the S3 in-flight submission window, bytes).
extern "C" common::backend_api::ResponseCode_t obj_get_backend_config(
    common::backend_api::ObjectBackendHandle_t backend_handle,
    const char* key,
    char* out_value_buffer,
    unsigned int* in_out_buffer_len);

// --- Client API ---

extern "C" common::backend_api::ResponseCode_t obj_create_client(
    common::backend_api::ObjectBackendHandle_t backend_handle,
    const common::backend_api::ObjectClientConfig_t* client_initial_config,
    common::backend_api::ObjectClientHandle_t* out_client_handle
);

extern "C" common::backend_api::ResponseCode_t obj_remove_client(
    common::backend_api::ObjectClientHandle_t client_handle
);

extern "C" common::backend_api::ResponseCode_t obj_request_read(
    common::backend_api::ObjectClientHandle_t client_handle,
    const char* path,
    common::backend_api::ObjectRange_t range,
    char* destination_buffer,
    common::backend_api::ObjectRequestId_t request_id
);

extern "C" common::backend_api::ResponseCode_t obj_wait_for_completions(common::backend_api::ObjectClientHandle_t client_handle,
                                                                        common::backend_api::ObjectCompletionEvent_t* event_buffer,
                                                                        unsigned int max_events_to_retrieve,
                                                                        unsigned int* out_num_events_retrieved,
                                                                        common::backend_api::ObjectWaitMode_t wait_mode);


// stop clients
// Stops the responder of each client, in order to notify callers which sent a request and are waiting for a response
extern "C" common::backend_api::ResponseCode_t obj_cancel_all_reads();

// release clients
extern "C" common::backend_api::ResponseCode_t obj_remove_all_clients();

// list files
extern "C" common::backend_api::ResponseCode_t obj_list_files(
    common::backend_api::ObjectClientHandle_t client_handle,
    const char* prefix,
    int is_recursive,
    common::backend_api::ObjectFileEntry_t** out_entries,
    unsigned* out_num_entries
);

extern "C" void obj_free_file_list(
    common::backend_api::ObjectFileEntry_t* entries,
    unsigned num_entries
);

}; //namespace runai::llm::streamer::impl::s3
