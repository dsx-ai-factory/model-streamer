/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <future>

#include "gcs/client_configuration/client_configuration.h"
#include "gcs/client/async_gcs_client/async_gcs_client.h"

#include "google/cloud/storage/client.h"
#include "google/cloud/options.h"

#include "common/backend_api/response/response.h"
#include "common/client_mgr/client_mgr.h"
#include "common/storage_uri/storage_uri.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "common/shared_queue/shared_queue.h"
#include "common/range/range.h"

namespace runai::llm::streamer::impl::gcs
{

struct GCSClient : common::IClient
{
    GCSClient(const common::backend_api::ObjectClientConfig_t& config);

     // verify that client's credentials have not changed
    bool verify_credentials(const common::backend_api::ObjectClientConfig_t & config) const;

    common::ResponseCode async_read(const char* path, common::backend_api::ObjectRange_t range, char* destination_buffer, common::backend_api::ObjectRequestId_t request_id);

    common::backend_api::Response async_read_response();

    common::ResponseCode list_files(const char* prefix, int is_recursive,
                                    std::vector<std::pair<std::string, size_t>>& results);

    // Stop sending requests to the object store
    // If stopped before all requests for an async_read() call are sent, subsequent request chunks will not be sent.
    void stop();

    // In-flight window (bytes) advertised to the generic layer: async threadpool size x chunk x margin
    size_t max_inflight_bytes() const { return _max_inflight_bytes; }

 private:
    std::atomic<bool> _stop;
    ClientConfiguration _client_config;
    const size_t _chunk_bytesize;
    size_t _max_inflight_bytes = 0;
    std::unique_ptr<AsyncGcsClient> _client;

    // queue of asynchronous responses
    using Responder = common::SharedQueue<common::backend_api::Response>;
    std::shared_ptr<Responder> _responder;
};

}; //namespace runai::llm::streamer::impl::gcs
