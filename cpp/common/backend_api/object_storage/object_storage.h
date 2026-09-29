/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "common/response/response.h"

namespace runai::llm::streamer::common::backend_api
{

// --- Opaque Handles ---
using ObjectRequestId_t = uint64_t;
using ObjectBackendHandle_t = void*;
using ObjectClientHandle_t = void*;

// --- Enums and Basic Structs ---

typedef enum {
    OBJECT_WAIT_MODE_NON_BLOCKING,      // Returns immediately, even if no events are ready.
    OBJECT_WAIT_MODE_BLOCK,             // Waits indefinitely until at least one event is ready or an error occurs.
    OBJECT_WAIT_MODE_TIMED_BLOCK        // Waits for a specified duration for events.
} ObjectWaitMode_t;

typedef enum {
    OBJECT_SHUTDOWN_POLICY_ON_PROCESS_EXIT,
    OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN,
} ObjectShutdownPolicy_t;

using ResponseCode_t = common::ResponseCode;

// Multiplier applied to a backend's raw in-flight capacity when advertising
// max_inflight_bytes (see obj_get_backend_config), giving completion-driven refill enough
// headroom to keep the backend busy during the completion -> resubmit round-trip.
// Defaults to 1.5; override with RUNAI_STREAMER_INFLIGHT_WINDOW_MARGIN (a positive float).
double inflight_window_margin();

// --- Config Params ---
struct ObjectConfigParam_t
{
    const char* key;
    const char* value;
};

// --- Client Config ---
struct ObjectClientConfig_t
{
    const char* endpoint_url;
    size_t default_storage_chunk_size;         // Explicit parameter for default chunk size for reads with this client.
    const ObjectConfigParam_t* initial_params; // Array of key-value pairs for other client-specific settings
                                               // (e.g., credentials, CA bundle, region) to be applied at creation. Can be NULL.
    unsigned num_initial_params;               // Number of parameters in initial_params. Must be 0 if initial_params is NULL.

    // How many clients the caller will run at once. GCS and Azure divide their per-client threads by
    // it, so the total stays the same whatever the count. Passed rather than read from the
    // environment, so the value has one source. S3 ignores it: its target is per client.
    unsigned concurrent_readers;
};

struct ObjectRange_t
{
    size_t offset; // Starting byte offset for the read
    size_t length; // Number of bytes to read
};

// --- Completion Event ---
struct ObjectCompletionEvent_t
{
    ObjectRequestId_t request_id;     // ID provided by caller in obj_request_read
    ResponseCode response_code;       // Response code from the backend
    size_t bytes_transferred;         // Actual bytes read into the caller's destination_buffer
};

// --- Backend API ---

/**
 * Opens and initializes the specific object storage backend implemented by this library.
 * The caller (streamer) is responsible for loading the correct backend library based on the URI scheme.
 *
 * - out_backend_handle - handle to the opened backend instance, or null if not needed
 * return success or an error code.
 */
ResponseCode_t obj_open_backend(ObjectBackendHandle_t* out_backend_handle);

/**
 * Closes an opened backend instance and releases all associated resources.
 */
ResponseCode_t obj_close_backend(ObjectBackendHandle_t backend_handle);

/**
 * Gets the shutdown policy for the backend.
 */
ObjectShutdownPolicy_t obj_get_backend_shutdown_policy();

/**
 * Sets backend-wide configuration parameters.
 * These parameters apply globally to the backend instance and may affect
 * all clients created from it, unless overridden by client-specific settings.
 *
 * - backend_handle - Handle to the backend instance, obtained from obj_open_backend.
 * - params - A pointer to an array of ObjectConfigParam_t structures.
 *            Each structure in the array represents a key-value pair for a configuration setting.
 *            The keys and acceptable values are specific to the backend implementation.
 *            This pointer can be NULL if num_params is 0.
 * - num_params - The number of configuration parameters in the 'params' array.
 *                If 'params' is NULL, this must be 0.
 * return success if all parameters were successfully processed and applied, or error code
 */
ResponseCode_t obj_set_backend_config(
    ObjectBackendHandle_t backend_handle,
    const ObjectConfigParam_t* params,
    unsigned int num_params
);

/**
 * Gets a backend-wide configuration parameter.
 * Retrieves the string value of a configuration parameter associated with the backend instance.
 *
 * - backend_handle - Handle to the backend instance, obtained from obj_open_backend.
 * - key - A null-terminated string representing the name of the configuration parameter to retrieve.
 * - out_value_buffer - A caller-allocated buffer where the null-terminated string value
 *                      of the configuration parameter will be copied.
 * - in_out_buffer_len - As input, a pointer to an unsigned int holding the capacity of out_value_buffer.
 *                       As output, if successful, it's updated to the length of the string written (excluding null terminator).
 *                       If the buffer is too small, it's updated to the required buffer size (including null terminator).
 * - in_out_buffer_len - will be updated with the required size.
 * return success or an error code.
 */
ResponseCode_t obj_get_backend_config(
    ObjectBackendHandle_t backend_handle,
    const char* key,
    char* out_value_buffer,
    unsigned int* in_out_buffer_len
);

/**
 * Creates a client instance for interacting with a specific endpoint and initial configuration.
 * Further configuration can be applied using obj_set_client_config.
 *
 * - backend_handle - Handle to the initialized backend.
 * - client_initial_config - Configuration for this client, including endpoint, default chunk size,
 *                           and an optional array of initial key-value parameters (e.g., credentials, CA bundle).
 * - out_client_handle - Pointer to store the handle to the created client instance.
 * return success, or an error code.
 */
ResponseCode_t obj_create_client(
    ObjectBackendHandle_t backend_handle,
    const ObjectClientConfig_t* client_initial_config,
    ObjectClientHandle_t* out_client_handle
);

/**
 * Removes/destroys a client instance.
 * - client_handle - Handle to the client instance to remove.
 * return success, or an error code.
 */
ResponseCode_t obj_remove_client(
    ObjectClientHandle_t client_handle
);

/**
 * Submits an asynchronous read request for a single chunk of an object. The backend may internally divide the request into smaller chunks, according to the configured backend chunk size
 * - client_handle - Handle to the client instance making the request.
 * - path - Identifier for the object (e.g., "bucket/key" for object storage, or "/path/file.dat" for HTTP).
 * - range - The byte range (offset and length) to read from the object.
 * - destination_buffer - Caller-allocated buffer where the read data will be stored.
 *                        This buffer must remain valid until the completion event for this request indicates the read is finished or has failed.
 * - buffer_capacity - The total capacity in bytes of the destination_buffer.
 * - request_id - A unique identifier for this request, provided by the caller. This ID will be returned in the completion event.
 * return success if the request was successfully submitted, or an error code.
 */
ResponseCode_t obj_request_read(
    ObjectClientHandle_t client_handle,
    const char* path,
    ObjectRange_t range,
    char* destination_buffer,
    ObjectRequestId_t request_id
);

/**
* Waits for and retrieves one or more completion events for a given client.
* - client_handle - Handle to the client instance whose completions are being awaited.
* - event_buffer - Caller-allocated array to store the retrieved completion events.
* - max_events_to_retrieve - The maximum number of events to write into event_buffer (its capacity).
* - out_num_events_retrieved - Pointer to store the actual number of events written to event_buffer.
* - wait_mode - Specifies the waiting behavior (non-blocking, blocking indefinitely, or timed blocking).
* - timeout_ms_if_timed - The timeout duration in milliseconds.
*                         This parameter is only used if wait_mode is OBJECT_WAIT_MODE_TIMED_BLOCK. For other modes, its value is ignored (can be 0 or any value).
* Return success if events were checked (even if zero events were retrieved when wait_mode is OBJECT_WAIT_MODE_NON_BLOCKING)
* Returns Finished response code if all read requests are completed
* Returns other error codes for more severe issues (e.g., invalid client_handle).
 * *out_num_events_retrieved will be 0 if no events are ready for non-blocking calls.
 */
ResponseCode_t obj_wait_for_completions(
    ObjectClientHandle_t client_handle,
    ObjectCompletionEvent_t* event_buffer,
    unsigned int max_events_to_retrieve,
    unsigned int* out_num_events_retrieved,
    ObjectWaitMode_t wait_mode
);

/**
 * Attempts to cancel all currently active/pending asynchronous read requests
 * associated with all the client handle.
 * Cancellation is best-effort. Completion events should still be expected for requests that were already in flight or too late to cancel.
 * e.g. S3 backend cannot cancel an already requested reads
 * Return success if the request to cancel all operations was successfully initiated.
 */
ResponseCode_t obj_cancel_all_reads();

/**
 * Attempts to remove all clients.
 * Return success if the request to remove all clients was successfully initiated.
 */
ResponseCode_t obj_remove_all_clients();

// --- File Listing ---

/**
 * A single entry returned by obj_list_files.
 * 'path' points into a buffer owned by the entry array; it is NOT an independent
 * allocation. Do not free/delete individual paths. The whole result (entries and
 * all their paths) is released as one unit via obj_free_file_list.
 */
struct ObjectFileEntry_t
{
    char*  path;  // full URI (e.g. "s3://bucket/key"), null-terminated
    size_t size;  // file size in bytes
};

/**
 * Synchronously lists all files under a given URI prefix.
 * Cloud pagination is handled internally; the full result is returned in one call.
 *
 * - client_handle   - handle obtained from obj_create_client
 * - prefix          - full URI prefix to list (e.g. "s3://bucket/models/")
 * - is_recursive    - 0 = immediate children only, non-zero = all descendants
 * - out_entries     - set to a single heap-allocated array of ObjectFileEntry_t (the
 *                     entries and their path strings share one allocation); release
 *                     the whole thing with obj_free_file_list and never free the
 *                     entries or individual paths directly. Set to nullptr when empty.
 * - out_num_entries - number of entries in *out_entries
 * Returns Success or an error code. On error nothing is allocated and the output
 * parameters are left untouched.
 */
ResponseCode_t obj_list_files(
    ObjectClientHandle_t  client_handle,
    const char*           prefix,
    int                   is_recursive,
    ObjectFileEntry_t**   out_entries,
    unsigned*             out_num_entries
);

/**
 * Frees the result returned by obj_list_files as a single allocation (entries and
 * all their path strings). Safe to call with a nullptr entries pointer.
 */
void obj_free_file_list(ObjectFileEntry_t* entries, unsigned num_entries);

} // namespace runai::llm::streamer::common::backend_api
