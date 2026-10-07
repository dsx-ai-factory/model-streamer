/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/workload/workload.h"
#include "common/device/device.h"

#include <gtest/gtest.h>
#include <memory>
#include <atomic>
#include <numeric>
#include <set>
#include <string>
#include <utility>

#include "streamer/impl/assigner/assigner.h"
#include "streamer/impl/batches/batches.h"

#include "utils/random/random.h"
#include "utils/temp/file/file.h"
#include "utils/thread/thread.h"
#include "utils/scope_guard/scope_guard.h"
#include "utils/semaphore/semaphore.h"
#include "common/exception/exception.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

namespace
{

// Build a request whose ranges tile each file contiguously and are packed consecutively into a single
// buffer - the layout the previous single-destination API implied. Every file therefore coalesces to
// exactly one ContiguousTransfer, which keeps these tests equivalent to what they asserted before.
std::vector<FileRanges> build_contiguous_request(const std::vector<std::string> & paths,
                                                 const std::vector<std::vector<size_t>> & chunks,
                                                 char * buffer)
{
    std::vector<FileRanges> request(paths.size());

    char * dst = buffer;
    for (size_t i = 0; i < paths.size(); ++i)
    {
        request[i].path = paths[i];
        request[i].ranges.reserve(chunks[i].size());

        size_t offset = 0;
        for (const auto size : chunks[i])
        {
            request[i].ranges.push_back(ReadRange{ offset, size, dst });
            offset += size;
            dst += size;
        }
    }

    return request;
}

} // namespace

TEST(Workload, Sanity)
{
    auto num_files = utils::random::number(1, 10);
    LOG(DEBUG) << "number of files " << num_files;

    common::s3::S3ClientWrapper::Params s3_params;

    std::atomic<bool> stopped(false);
    std::vector<std::string> paths;
    std::vector<std::vector<size_t>> chunks(num_files);
    std::vector<std::vector<uint8_t>> data(num_files);
    std::vector<unsigned> num_chunks(num_files);
    std::vector<utils::temp::File> file;
    std::vector<std::set<int>> expected_responses(num_files);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    auto config = std::make_shared<Config>(utils::random::number(1, 20), utils::random::number(1, 20), utils::random::number<size_t>(1, 1024), chunk_size, false /* do not force minimum chunk size */);
    auto responder = std::make_shared<common::Responder>(0);

    size_t total_bytes = 0;
    for (unsigned i = 0; i < num_files; ++i)
    {
        auto size = utils::random::number(1000, 100000);
        num_chunks[i] = utils::random::number(1, 20);
        EXPECT_LT(num_chunks[i], size);
        responder->increment(num_chunks[i]);
        total_bytes += size;
        data[i] = utils::random::buffer(size);
        file.push_back(utils::temp::File(data[i]));
        paths.push_back(file[i].path);

        // The ranges must exist before the Assigner is built - it coalesces them - so the chunk sizes
        // are chosen here rather than after the assignment as they used to be.
        chunks[i] = utils::random::chunks(size, num_chunks[i]);
        EXPECT_EQ(chunks[i].size(), num_chunks[i]);
        EXPECT_EQ(std::accumulate(chunks[i].begin(), chunks[i].end(), static_cast<size_t>(0)), size);

        for (unsigned j = 0; j < num_chunks[i]; ++j)
        {
            expected_responses[i].insert(j);
        }
    }

    std::vector<char> buffer(total_bytes);
    const auto request = build_contiguous_request(paths, chunks, buffer.data());

    {
        Assigner assigner(request, config);

        // each file's ranges are adjacent in both the file and the buffer, so each file is one transfer
        EXPECT_EQ(assigner.transfers().size(), num_files);
        EXPECT_LE(assigner.num_workloads(), config->concurrency);

        std::vector<Workload> workloads(assigner.num_workloads());

        for (const auto & transfer : assigner.transfers())
        {
            EXPECT_GT(transfer.tasks.size(), 0);
            EXPECT_LE(transfer.tasks.size(), config->concurrency);
            EXPECT_EQ(transfer.first_range_index, 0);
            EXPECT_EQ(transfer.range_sizes.size(), num_chunks[transfer.file_index]);

            Batches batches(utils::random::number(), transfer.file_index, transfer.tasks, config, responder,
                            request[transfer.file_index].path, s3_params,
                            transfer.range_sizes, transfer.first_range_index, common::Device::host());

            for (size_t j = 0; j < batches.size(); ++j)
            {
                auto & batch = batches[j];
                workloads[batch.workload_index].add_batch(std::move(batch));
            }
        }
        // execute workloads
        for (auto & workload : workloads)
        {
            workload.execute(stopped);
        }
    }

    // wait for all the requests to finish

    for (unsigned file_idx = 0; file_idx < num_files; ++file_idx)
    {
        for (unsigned i = 0; i < num_chunks[file_idx]; ++i)
        {
            const auto r = responder->pop();
            EXPECT_EQ(r.ret, common::ResponseCode::Success);
            EXPECT_EQ(expected_responses[file_idx].count(r.index), 1);
            expected_responses[file_idx].erase(r.index);
        }
        EXPECT_TRUE(expected_responses[file_idx].empty());
    }

    auto r = responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::FinishedError);

    // verify read
    size_t offset = 0;

    for (unsigned file_idx = 0; file_idx < num_files; ++file_idx)
    {
        const auto & expected_content = data[file_idx];
        for (unsigned i = 0; i < expected_content.size(); ++i)
        {
            EXPECT_EQ(static_cast<char>(expected_content[i]), buffer[offset + i]);
        }
        offset += expected_content.size();
    }
}

TEST(Workload, Stopped)
{
    auto num_files = utils::random::number(1, 10);

    common::s3::S3ClientWrapper::Params s3_params;

    std::atomic<bool> stopped(false);
    std::vector<std::string> paths;
    std::vector<std::vector<size_t>> chunks(num_files);
    std::vector<std::vector<uint8_t>> data(num_files);
    std::vector<unsigned> num_chunks(num_files);
    std::vector<utils::temp::File> file;
    std::vector<std::set<int>> expected_responses(num_files);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    // concurrency 1: coalescing must still merge each file's ranges into a single read, since it runs
    // before any work is divided between workers
    auto config = std::make_shared<Config>(1, 1, utils::random::number<size_t>(1, 1024), chunk_size, false /* do not force minimum chunk size */);
    auto responder = std::make_shared<common::Responder>(0);

    size_t total_bytes = 0;
    for (unsigned i = 0; i < num_files; ++i)
    {
        auto size = utils::random::number(1000, 100000);
        num_chunks[i] = utils::random::number(1, 20);
        EXPECT_LT(num_chunks[i], size);
        responder->increment(num_chunks[i]);
        total_bytes += size;
        data[i] = utils::random::buffer(size);
        file.push_back(utils::temp::File(data[i]));
        paths.push_back(file[i].path);

        chunks[i] = utils::random::chunks(size, num_chunks[i]);
        EXPECT_EQ(chunks[i].size(), num_chunks[i]);
        EXPECT_EQ(std::accumulate(chunks[i].begin(), chunks[i].end(), static_cast<size_t>(0)), size);

        for (unsigned j = 0; j < num_chunks[i]; ++j)
        {
            expected_responses[i].insert(j);
        }
    }

    std::vector<char> buffer(total_bytes);
    const auto request = build_contiguous_request(paths, chunks, buffer.data());

    Assigner assigner(request, config);

    EXPECT_EQ(assigner.transfers().size(), num_files);

    Workload workload;

    for (const auto & transfer : assigner.transfers())
    {
        // one worker, so a transfer is never split
        EXPECT_EQ(transfer.tasks.size(), 1);

        Batches batches(utils::random::number(), transfer.file_index, transfer.tasks, config, responder,
                        request[transfer.file_index].path, s3_params,
                        transfer.range_sizes, transfer.first_range_index, common::Device::host());

        for (size_t j = 0; j < batches.size(); ++j)
        {
            auto & batch = batches[j];
            workload.add_batch(std::move(batch));
        }
    }

    // execute workloads
    auto thread = utils::Thread([&]()
    {
        workload.execute(stopped);
    });

    ::usleep(utils::random::number(300));

    common::s3::S3ClientWrapper::stop();
    stopped = true;

    // wait for all the requests to finish

    for (unsigned file_idx = 0; file_idx < num_files; ++file_idx)
    {
        for (unsigned i = 0; i < num_chunks[file_idx]; ++i)
        {
            const auto r = responder->pop();
            EXPECT_TRUE(r.ret == common::ResponseCode::Success || r.ret == common::ResponseCode::FinishedError);
            EXPECT_LT(r.file_index, num_files);
            EXPECT_LT(r.index, num_chunks[r.file_index]);
            EXPECT_EQ(expected_responses[r.file_index].count(r.index), 1);
            expected_responses[r.file_index].erase(r.index);
        }
    }
    for (unsigned file_idx = 0; file_idx < num_files; ++file_idx)
    {
        EXPECT_TRUE(expected_responses[file_idx].empty());
    }

    auto r = responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::FinishedError);
}

// A worker reads the destination off whichever batch a completed chunk belongs to, so a workload has
// to be homogeneous in it - the same rule the backend kind already has.
TEST(Workload, Refuses_Batches_For_Different_Devices)
{
    auto config = std::make_shared<Config>(false /* do not force minimum */);
    auto responder = std::make_shared<common::Responder>(0, common::QueueMode::PERSISTENT);
    common::s3::S3ClientWrapper::Params s3_params;

    auto batch_for = [&](common::Device device)
    {
        return Batch(1 /* submission */, 0 /* workload */, 0 /* file */, "/tmp/file", s3_params,
                     Tasks{}, responder, config, 4096, device);
    };

    Workload workload;
    ASSERT_EQ(workload.add_batch(batch_for(common::Device::cuda(0))), common::ResponseCode::Success);
    EXPECT_EQ(workload.device(), common::Device::cuda(0));

    EXPECT_EQ(workload.add_batch(batch_for(common::Device::cuda(1))),
              common::ResponseCode::InvalidParameterError);
    EXPECT_EQ(workload.add_batch(batch_for(common::Device::host())),
              common::ResponseCode::InvalidParameterError);
    EXPECT_EQ(workload.add_batch(batch_for(common::Device::cuda(0))), common::ResponseCode::Success);

    EXPECT_EQ(workload.size(), 2u) << "only the agreeing batches were kept";
}

}; // namespace runai::llm::streamer::impl
