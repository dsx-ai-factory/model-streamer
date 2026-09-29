/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/range/range.h"
#include "common/s3_credentials/s3_credentials.h"
#include "common/response_code/response_code.h"
#include "common/backend_api/response/response.h"
#include "common/backend_api/object_storage/object_storage.h"

namespace runai::llm::streamer::common::s3
{

// --- Backend API ---

extern "C" common::backend_api::ResponseCode_t obj_open_backend(common::backend_api::ObjectBackendHandle_t* out_backend_handle);
extern "C" common::backend_api::ResponseCode_t obj_close_backend(common::backend_api::ObjectBackendHandle_t backend_handle);
extern "C" common::backend_api::ObjectShutdownPolicy_t obj_get_backend_shutdown_policy();
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

extern "C" common::backend_api::ResponseCode_t obj_wait_for_completions(
    common::backend_api::ObjectClientHandle_t client_handle,
    common::backend_api::ObjectCompletionEvent_t* event_buffer,
    unsigned int max_events_to_retrieve,
    unsigned int* out_num_events_retrieved,
    common::backend_api::ObjectWaitMode_t wait_mode
);

extern "C" common::backend_api::ResponseCode_t obj_cancel_all_reads();
extern "C" common::backend_api::ResponseCode_t obj_remove_all_clients();

extern "C" common::backend_api::ResponseCode_t obj_list_files(
    common::backend_api::ObjectClientHandle_t client_handle,
    const char* prefix,
    int is_recursive,
    common::backend_api::ObjectFileEntry_t** out_entries,
    unsigned* out_num_entries
);

extern "C" void obj_free_file_list(common::backend_api::ObjectFileEntry_t* entries, unsigned num_entries);

// Test hooks for obj_list_files:
//   set the files that the next obj_list_files call returns
extern "C" void runai_mock_s3_set_files(const char** paths, const size_t* sizes, unsigned count);
//   force obj_list_files to return the given response code (Success restores normal behavior)
extern "C" void runai_mock_s3_set_list_files_response(common::backend_api::ResponseCode_t response);
//   the is_recursive argument observed by the last obj_list_files call (-1 if never called)
extern "C" int runai_mock_s3_last_list_files_is_recursive();

extern "C" void runai_mock_s3_set_response_time_ms(unsigned milliseconds);
// Sets the in-flight window (bytes) reported by obj_get_backend_config("max_inflight_bytes").
extern "C" void runai_mock_s3_set_inflight_window(size_t bytes);
// Peak per-client in-flight (submitted-but-not-completed) request count observed since the last cleanup.
extern "C" size_t runai_mock_s3_max_concurrent();
// Total obj_request_read calls since the last cleanup - how many reads the backend was actually asked
// for, which is what shows whether small ranges were packed into one request or issued one by one.
extern "C" size_t runai_mock_s3_requests();
// Fail (with FileAccessError) the completion of any read whose path contains substr; "" / nullptr disables.
extern "C" void runai_mock_s3_set_failing_path(const char* substr);
// Fail the next `count` submitted reads at completion with `response_code`. A retry that reuses the same
// request id is a new submission and consumes another count. Used to test application-level retries.
extern "C" void runai_mock_s3_set_read_failures(unsigned count, common::backend_api::ResponseCode_t response_code);
extern "C" size_t runai_mock_s3_total_read_requests();
// Peak number of completion events returned by a single obj_wait_for_completions call since the last cleanup.
extern "C" size_t runai_mock_s3_max_events_per_wait();
// When enabled, obj_wait_for_completions appends a FinishedError sentinel event (handle 0) after the
// ready completions, mimicking the azure/gcs plugins' responder-drain behavior.
extern "C" void runai_mock_s3_set_append_finished_sentinel(bool enabled);
extern "C" void runai_s3_mock_set_backend_shutdown_policy(common::backend_api::ObjectShutdownPolicy_t policy);
extern "C" int runai_mock_s3_clients();
// value of a config parameter (credential / endpoint_url) received by the last obj_create_client, or nullptr
extern "C" const char* runai_mock_s3_last_client_config_value(const char* key);
extern "C" void runai_mock_s3_cleanup();
extern "C" bool runai_mock_s3_is_shutdown();

}; //namespace runai::llm::streamer::common::s3
