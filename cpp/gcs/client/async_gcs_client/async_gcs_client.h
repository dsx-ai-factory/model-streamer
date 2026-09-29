/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <future>
#include <memory>
#include <string>

#include "google/cloud/future.h"
#include "google/cloud/storage/client.h"

#include "utils/threadpool/threadpool.h"

namespace runai::llm::streamer::impl::gcs
{

typedef std::function<google::cloud::storage::ObjectReadStream()> ReadObjectFn;

struct ReadObjectTask {
    ReadObjectFn taskFn;
    google::cloud::promise<google::cloud::storage::ObjectReadStream> promise;

public:
    void execute();
};

template<typename... Options>
static ReadObjectFn createReadObjectFn(google::cloud::storage::Client * client, const std::string &bucket_name, const std::string &object_name, Options&&... opts) {
    return [=]() {
        return client->ReadObject(bucket_name, object_name, opts...);
    };
}

/**
 * An async wrapper for the synchronous google::cloud::storage::Client.
 */
struct AsyncGcsClient
{
public:
    AsyncGcsClient(google::cloud::Options options, unsigned int max_concurrency, bool use_grpc = false);

    template<typename... Options>
    google::cloud::future<google::cloud::storage::ObjectReadStream> ReadObjectAsync(
        std::string const& bucket_name,
        std::string const& object_name,
        Options&&... opts)
    {
        auto client = _client.get();
        ReadObjectFn readFn = createReadObjectFn(client, bucket_name, object_name, opts...);

        google::cloud::promise<google::cloud::storage::ObjectReadStream> promise;
        auto future = promise.get_future();

        ReadObjectTask task{
            std::move(readFn),
            std::move(promise)
        };

        // Push the request onto the thread pool's queue for processing.
        this->push_task(std::move(task));

        return future;
    }

private:
    // Private helper to push a request to the ThreadPool.
    void push_task(ReadObjectTask&& task);

    std::unique_ptr<google::cloud::storage::Client> _client;
    utils::ThreadPool<ReadObjectTask> _pool;
};

}; // namespace runai::llm::streamer::impl::gcs
