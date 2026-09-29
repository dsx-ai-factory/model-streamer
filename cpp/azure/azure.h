/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/backend_api/object_storage/object_storage.h"
#include "common/response/response.h"
#include "common/range/range.h"

// For connecting to Azure Blob Storage:
//
// 1. Uri should be in the format az://container/path
//
// 2. Authentication methods (checked in this order):
//    a. Connection string (Azurite testing only, requires AZURITE_TESTING build):
//       - Set AZURE_STORAGE_CONNECTION_STRING environment variable
//    b. SAS token:
//       - Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_SAS_TOKEN environment variables
//       - Token is appended as query string to the service URL
//    c. Storage account key:
//       - Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY environment variables
//       - Uses StorageSharedKeyCredential
//    d. DefaultAzureCredential (Recommended):
//       - Set AZURE_STORAGE_ACCOUNT_NAME environment variable
//       - DefaultAzureCredential tries multiple authentication methods in order:
//         * Environment variables (AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET) for service principal
//         * Managed Identity (no env vars needed when running in Azure)
//         * Azure CLI (az login)
//         * Azure PowerShell (Connect-AzAccount)
//         * Azure Developer CLI (azd auth login)
//
// 3. For local testing with Azurite:
//    - Use AZURE_STORAGE_CONNECTION_STRING environment variable
//    - Run Azurite with --skipApiVersionCheck flag
//
// Example usage:
// key-based: AZURE_STORAGE_ACCOUNT_NAME="account" AZURE_STORAGE_ACCOUNT_KEY="key" <streamer app> az://container/path
// sas-token: AZURE_STORAGE_ACCOUNT_NAME="account" AZURE_STORAGE_SAS_TOKEN="sv=2021-08-06&ss=b&..." <streamer app> az://container/path
// managed:   AZURE_STORAGE_ACCOUNT_NAME="account" <streamer app> az://container/path
// azurite:   AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;..." <streamer app> az://container/path
// sovereign: AZURE_STORAGE_ACCOUNT_NAME="account" AZURE_STORAGE_ENDPOINT_SUFFIX="blob.core.chinacloudapi.cn" <streamer app> az://container/path
// programmatic: Pass credentials in ObjectClientConfig_t.initial_params

namespace runai::llm::streamer::impl::azure
{

// --- Backend API ---

extern "C" common::backend_api::ResponseCode_t obj_open_backend(common::backend_api::ObjectBackendHandle_t* out_backend_handle);
extern "C" common::backend_api::ResponseCode_t obj_close_backend(common::backend_api::ObjectBackendHandle_t backend_handle);
extern "C" common::backend_api::ObjectShutdownPolicy_t obj_get_backend_shutdown_policy();

extern "C" common::backend_api::ResponseCode_t obj_get_backend_config(
    common::backend_api::ObjectBackendHandle_t backend_handle,
    const char* key,
    char* out_value_buffer,
    unsigned int* in_out_buffer_len
);

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

extern "C" common::backend_api::ResponseCode_t obj_wait_for_completions(
    common::backend_api::ObjectClientHandle_t client_handle,
    common::backend_api::ObjectCompletionEvent_t* event_buffer,
    unsigned int max_events_to_retrieve,
    unsigned int* out_num_events_retrieved,
    common::backend_api::ObjectWaitMode_t wait_mode
);

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

}; //namespace runai::llm::streamer::impl::azure
