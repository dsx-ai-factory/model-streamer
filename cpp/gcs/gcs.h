/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/backend_api/object_storage/object_storage.h"
#include "common/response/response.h"
#include "common/range/range.h"

namespace runai::llm::streamer::impl::gcs
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

}; //namespace runai::llm::streamer::impl::gcs
