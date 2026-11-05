/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <string>
#include <utility>
#include <optional>
#include <vector>
#include <future>
#include <memory>
#include <functional>

#include "google/cloud/future.h"
#include "google/cloud/storage/client.h"
#include "google/cloud/storage/oauth2/credentials.h"
#include "google/cloud/common_options.h"
#include "google/cloud/status_or.h"

#include "gcs/client/client.h"

#include "common/exception/exception.h"

#include "utils/logging/logging.h"
#include "utils/env/env.h"
#include "utils/fd/fd.h"

namespace runai::llm::streamer::impl::gcs
{

GCSClient::GCSClient(const common::backend_api::ObjectClientConfig_t& config) :
    _stop(false),
    _client_config(config.concurrent_readers),
    _responder(nullptr),
    _chunk_bytesize(config.default_storage_chunk_size)
{
    // advertise an in-flight window sized to the async threadpool: margin x pool x chunk,
    // floored at two chunks so the window can always keep the pool busy
    _max_inflight_bytes = std::max(
        static_cast<size_t>(common::backend_api::inflight_window_margin() * _client_config.max_concurrency * _chunk_bytesize),
        static_cast<size_t>(2) * _chunk_bytesize);

    _client = std::make_unique<AsyncGcsClient>(
        _client_config.options, 
        _client_config.max_concurrency,
        _client_config.use_grpc
    );
}

bool GCSClient::verify_credentials(const common::backend_api::ObjectClientConfig_t & config) const
{
    // TODO: Verify credentials once they are passed in via config.
    return true;
}

common::backend_api::Response GCSClient::async_read_response()
{
    if (_responder == nullptr)
    {
        LOG(WARNING) << "Requesting response with uninitialized responder";
        return common::ResponseCode::FinishedError;
    }

    return _responder->pop();
}

// Reads the stream into dest_buffer. The SDK's status is checked before the byte count: a failed
// request (TLS, auth, missing object) delivers no bytes, and its status is the only record of why.
google::cloud::Status write_stream_to_buffer(
        google::cloud::storage::ObjectReadStream && stream,
        char * dest_buffer,
        size_t bytesize) {
    size_t bytes_received = 0;
    while (stream.read(dest_buffer, bytesize)) {
        bytes_received += stream.gcount();
    }
    // a read that stops short of bytesize returns false, but its bytes are still in gcount()
    bytes_received += stream.gcount();
    stream.Close();

    auto status = stream.status();
    if (!status.ok()) {
        return status;
    }
    if (bytes_received != bytesize) {
        return google::cloud::Status(google::cloud::StatusCode::kDataLoss,
            "received " + std::to_string(bytes_received) + " bytes, but " + std::to_string(bytesize) + " were requested");
    }
    return {};
}

common::ResponseCode GCSClient::async_read(const char* path, common::backend_api::ObjectRange_t range, char* destination_buffer, common::backend_api::ObjectRequestId_t request_id)
{
    if (_responder == nullptr)
    {
        _responder = std::make_shared<Responder>(1);
    }
    else
    {
        _responder->increment(1);
    }

    char * buffer_ = destination_buffer;
    // split range into chunks
    size_t size = std::max(1UL, range.length/_chunk_bytesize);
    LOG(SPAM) << "Number of chunks is: " << size;

    // each range is divided into chunks (size is the number of chunks)
    // when all the chunks have been read successfuly the response for that range is pushed to the responder

    auto counter = std::make_shared< std::atomic<unsigned> >(size);
    // success flag for the current range is passed to the client
    auto is_success = std::make_shared< std::atomic<bool> >(true);

    const auto uri = common::s3::StorageUri(path);

    std::string bucket_name(uri.bucket);
    std::string path_name(uri.path);

    size_t total_ = range.length;
    size_t offset_ = range.offset;

    for (unsigned i = 0; i < size && !_stop; ++i)
    {
        size_t bytesize_ = (i == size - 1 ? total_ : _chunk_bytesize);

        _client->ReadObjectAsync(bucket_name, path_name, google::cloud::storage::ReadRange(offset_, offset_ + bytesize_)).then(
            [dest_buffer = buffer_, responder = _responder, request_id, bytesize_, offset_, counter, is_success, path = std::string(path), credentials = _client_config.credentials](auto f) {
            auto stream = f.get();
            const auto status = write_stream_to_buffer(std::move(stream), dest_buffer, bytesize_);
            if (status.ok())
            {
                const auto running = counter->fetch_sub(1);
                LOG(SPAM) << "Async read request " << request_id << " succeeded - " << running << " running";
                // send success response only if all the requests have succeeded
                // note that unsuccessful attempts do not update the counter
                if (running == 1)
                {
                    common::backend_api::Response r(request_id, common::ResponseCode::Success);
                    responder->push(std::move(r));
                }
            }
            else
            {
                // Note: currently a failure to read any sub range fails the entire read request
                //       a retry mechanism should be added for failed reads
                bool previous = is_success->exchange(false);
                // send error response and log the cause only once - the other chunks of a failed
                // request usually fail for the same reason
                if (previous)
                {
                    LOG(ERROR) << "Failed to read " << path << " range [" << offset_ << ", " << offset_ + bytesize_
                               << ") of request " << request_id << ": " << status << " (credentials: " << credentials << ")";
                    common::backend_api::Response r(request_id, common::ResponseCode::FileAccessError);
                    responder->push(std::move(r));
                }
                else
                {
                    LOG(DEBUG) << "Another chunk of failed request " << request_id << " failed: " << status;
                }
            }
        });

        total_ -= bytesize_;
        offset_ += bytesize_;
        buffer_ += bytesize_;
    }

    return _stop ? common::ResponseCode::FinishedError : common::ResponseCode::Success;
}

void GCSClient::stop()
{
    _stop = true;
    if (_responder != nullptr)
    {
        _responder->stop();
    }
}

common::ResponseCode GCSClient::list_files(
    const char* prefix,
    int is_recursive,
    std::vector<std::pair<std::string, size_t>>& results)
{
    const auto uri = common::s3::StorageUri(prefix);
    const std::string uri_prefix = uri.scheme + "://" + uri.bucket + "/";

    google::cloud::storage::Client gcs_client(_client_config.options);

    namespace gcs = google::cloud::storage;

    auto list_objects = [&](auto&&... opts)
    {
        for (auto&& meta : gcs_client.ListObjects(uri.bucket, opts...))
        {
            if (!meta)
            {
                LOG(ERROR) << "GCS ListObjects failed for " << prefix << ": " << meta.status()
                           << " (credentials: " << _client_config.credentials << ")";
                return common::ResponseCode::FileAccessError;
            }
            const auto& name = meta->name();
            if (!name.empty() && name.back() == '/')
            {
                continue;
            }
            results.emplace_back(uri_prefix + name, static_cast<size_t>(meta->size()));
        }
        return common::ResponseCode::Success;
    };

    if (is_recursive)
    {
        return list_objects(gcs::Prefix(uri.path));
    }
    else
    {
        return list_objects(gcs::Prefix(uri.path), gcs::Delimiter("/"));
    }
}

}; // namespace runai::llm::streamer::impl::gcs
