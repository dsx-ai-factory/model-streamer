/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>
#include <thread>
#include <utility>

#include "gcs/client/async_gcs_client/async_gcs_client.h"
#include "google/cloud/storage/grpc_plugin.h"

namespace runai::llm::streamer::impl::gcs
{

void ReadObjectTask::execute() {
    auto stream = taskFn();
    promise.set_value(std::move(stream));
}

AsyncGcsClient::AsyncGcsClient(google::cloud::Options opts, unsigned int max_pool_size, bool use_grpc) :
    _pool([&](ReadObjectTask&& task, std::atomic<bool> & stopped)
        {
            task.execute();
        }, max_pool_size)
{
    if (use_grpc) {
        LOG(INFO) << "GCS client initialized with gRPC transport. "
                << "Make sure you use this with direct connectivity for best performance; "
                << "otherwise, it is recommended to not enable the gRPC client.";
        _client = std::make_unique<google::cloud::storage::Client>(
            google::cloud::storage::MakeGrpcClient(std::move(opts)));
    } else {
        LOG(DEBUG) << "GCS client initialized with default HTTP/JSON transport.";
        _client = std::make_unique<google::cloud::storage::Client>(std::move(opts));
    }
}

// Implementation of the private helper function.
void AsyncGcsClient::push_task(ReadObjectTask&& task)
{
    _pool.push(std::move(task));
}

}; // namespace runai::llm::streamer::impl::gcs
