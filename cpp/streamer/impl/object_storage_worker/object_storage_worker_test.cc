/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/object_storage_worker/object_storage_worker.h"
#include "common/device/device.h"
#include "device/mock/mock_device.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "streamer/impl/assigner/assigner.h"
#include "streamer/impl/batches/batches.h"

#include "common/exception/exception.h"
#include "common/s3_wrapper/s3_wrapper.h"

#include "utils/threadpool/threadpool.h"
#include "utils/random/random.h"
#include "utils/scope_guard/scope_guard.h"
#include "utils/thread/thread.h"
#include "utils/dylib/dylib.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

namespace
{

// One object-storage submission spread over several files, built the way Streamer::async_request builds it:
// one submission_id for the whole submission, one Batches per file. Bundles the config/responder/paths so a
// test can wait on the responder and assert per-file, per-request completion.
struct Submission
{
    Submission(SubmissionId submission_id,
               unsigned num_files,
               std::shared_ptr<Config> config,
               std::shared_ptr<common::Responder> responder,
               unsigned ranges_per_file = 0) :
        submission_id(submission_id),
        config(config),
        responder(responder),
        num_chunks(num_files),
        expected(num_files)
    {
        const std::string bucket = "test-bucket";
        std::vector<std::vector<size_t>> chunks(num_files);

        for (unsigned i = 0; i < num_files; ++i)
        {
            const auto size = utils::random::number(1000, 100000);
            num_chunks[i] = ranges_per_file == 0 ? utils::random::number(1, 20) : ranges_per_file;
            EXPECT_LT(num_chunks[i], size);
            responder->increment(num_chunks[i]);
            total_bytes += size;

            paths.push_back("s3://" + bucket + "/" + utils::random::string());

            // the ranges must exist before the Assigner is built - it coalesces them
            chunks[i] = utils::random::chunks(size, num_chunks[i]);
            EXPECT_EQ(chunks[i].size(), num_chunks[i]);
        }

        buffer.resize(total_bytes);

        // each file's ranges tile it contiguously, and the files are packed consecutively into the one
        // buffer, so every file coalesces to exactly one transfer
        request.resize(num_files);
        char * dst = buffer.data();
        for (unsigned i = 0; i < num_files; ++i)
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
    }

    // build the workloads (one Assigner over the request, Batches per contiguous transfer) ready to push
    // to the pool
    std::vector<Workload> build(common::Device device = common::Device::host())
    {
        Assigner assigner(request, config);
        std::vector<Workload> workloads(assigner.num_workloads());

        const common::s3::Credentials credentials;
        for (const auto & transfer : assigner.transfers())
        {
            const auto file_idx = transfer.file_index;

            auto uri = std::make_shared<common::s3::StorageUri>(paths[file_idx]);
            common::s3::S3ClientWrapper::Params params(uri, credentials, config->s3_block_bytesize, config->s3_concurrency);

            Batches batches(submission_id, file_idx, transfer.tasks, config, responder, paths[file_idx], params,
                            transfer.range_sizes, transfer.first_range_index, device);
            for (size_t j = 0; j < batches.size(); ++j)
            {
                workloads[batches[j].workload_index].add_batch(std::move(batches[j]));
            }

            for (unsigned i = 0; i < num_chunks[file_idx]; ++i)
            {
                expected[file_idx].insert(i);
            }
        }
        return workloads;
    }

    unsigned total_requests() const
    {
        return std::accumulate(num_chunks.begin(), num_chunks.end(), 0u);
    }

    SubmissionId submission_id;
    std::shared_ptr<Config> config;
    std::shared_ptr<common::Responder> responder;
    std::vector<std::string> paths;
    std::vector<FileRanges> request;
    std::vector<char> buffer;
    size_t total_bytes = 0;
    std::vector<unsigned> num_chunks;
    std::vector<std::set<int>> expected;
};



} // namespace

// Fixture: owns the s3 mock handle, resets its knobs before each test, and releases the plugin's clients +
// backend handle after each test (so state never leaks between tests). Provides build() to set up a
// submission, and small wrappers around the mock knobs.
class ObjectStorageWorkerTest : public ::testing::Test
{
 protected:
    ObjectStorageWorkerTest() : _dylib("libstreamers3.so") {}

    void SetUp() override
    {
        set_response_time(0);
        set_sentinel(false);
        fail_paths_containing("");   // no failures unless a test asks for them
    }

    void TearDown() override
    {
        _dylib.dlsym<void(*)()>("runai_mock_s3_cleanup")();
        common::s3::S3ClientWrapper::shutdown();
    }

    // create a Config/Responder and a (num_files) submission; returns the workloads ready to dispatch. Stores
    // the config/responder/submission as members so the test can build a pool and wait on the responder.
    std::vector<Workload> build(unsigned num_files, unsigned s3_concurrency, unsigned ranges_per_file = 0,
                                size_t s3_block_bytesize = 0, common::Device device = common::Device::host())
    {
        make_context(s3_concurrency);
        // Applied here, between the config and the cut: Batches divides the ranges using this value, so
        // setting it after build() would leave the chunks already cut at the random default.
        if (s3_block_bytesize != 0)
        {
            config->s3_block_bytesize = s3_block_bytesize;
        }
        submission = std::make_unique<Submission>(utils::random::number(), num_files, config, responder, ranges_per_file);
        return submission->build(device);
    }

    // Bigger than any file Submission generates (it tops out at 100000 bytes), so one range is one
    // chunk and a single injected failure lands on the only backend read.
    static constexpr size_t SingleChunkBlockBytesize = 1024 * 1024;

    // set up config + responder without a submission (for tests that build several submissions themselves)
    void make_context(unsigned s3_concurrency)
    {
        // (concurrency, s3_concurrency, fs_block, s3_block, force_min_chunk=false)
        config = std::make_shared<Config>(s3_concurrency, s3_concurrency, utils::random::number<size_t>(1, 1024), utils::random::number<size_t>(1, 1024), false);
        responder = std::make_shared<common::Responder>(0);
    }

    static utils::ThreadPool<Workload> make_pool(unsigned size,
                                                std::shared_ptr<DeviceWriter> writer = nullptr,
                                                std::shared_ptr<DeviceIssuer> issuer = nullptr)
    {
        return utils::ThreadPool<Workload>(
            [writer, issuer]() -> std::unique_ptr<utils::Worker<Workload>>
            {
                // the s3 mock client ignores credentials, so the provider returns an empty set
                return std::make_unique<ObjectStorageWorker>([]() { return common::s3::Credentials{}; },
                                                             writer, issuer);
            },
            size);
    }

    static void push_all(utils::ThreadPool<Workload> & pool, std::vector<Workload> & workloads)
    {
        for (auto & workload : workloads)
        {
            if (workload.size() > 0)
            {
                pool.push(std::move(workload));
            }
        }
    }

    void set_response_time(unsigned ms) { _dylib.dlsym<void(*)(unsigned)>("runai_mock_s3_set_response_time_ms")(ms); }
    void set_sentinel(bool on)          { _dylib.dlsym<void(*)(bool)>("runai_mock_s3_set_append_finished_sentinel")(on); }
    void set_window(size_t bytes)       { _dylib.dlsym<void(*)(size_t)>("runai_mock_s3_set_inflight_window")(bytes); }
    void fail_paths_containing(const char * substr) { _dylib.dlsym<void(*)(const char*)>("runai_mock_s3_set_failing_path")(substr); }
    void set_read_failures(unsigned count, common::ResponseCode code)
    {
        _dylib.dlsym<void(*)(unsigned, common::backend_api::ResponseCode_t)>("runai_mock_s3_set_read_failures")(count, code);
    }
    size_t total_read_requests()        { return _dylib.dlsym<size_t(*)()>("runai_mock_s3_total_read_requests")(); }
    size_t max_concurrent()             { return _dylib.dlsym<size_t(*)()>("runai_mock_s3_max_concurrent")(); }
    size_t requests()                   { return _dylib.dlsym<size_t(*)()>("runai_mock_s3_requests")(); }
    int clients()                       { return _dylib.dlsym<int(*)()>("runai_mock_s3_clients")(); }

    // The chunks the worker will submit: exactly Batch::chunks, one backend read each.
    //
    // Read from the workload rather than recomputed from s3_block_bytesize. The cut happens upstream
    // in Batches, and a test that re-derives it is asserting against its own copy of the rule instead
    // of against the thing under test.
    static size_t count_object_chunks(const std::vector<Workload> & workloads)
    {
        size_t result = 0;
        for (const auto & workload : workloads)
        {
            for (const auto & batch : workload.batches())
            {
                result += batch.chunks.size();
            }
        }
        return result;
    }

    std::shared_ptr<Config> config;
    std::shared_ptr<common::Responder> responder;
    std::unique_ptr<Submission> submission;

 private:
    utils::Dylib _dylib;
};

// All requests of a multi-file submission complete successfully through a pool of ObjectStorageWorkers,
// with the drained-responder sentinel randomly enabled to prove the worker tolerates it.
// THE step: an object storage read whose destination is a device lands in pinned host memory and is
// copied from there, and its ranges are answered only once the copy has retired.
TEST_F(ObjectStorageWorkerTest, Reads_A_Device_Submission_Through_Pinned_Buffers)
{
    constexpr unsigned Files = 2;
    constexpr unsigned RangesPerFile = 4;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    // One range is one chunk, because the block size is larger than any file the fixture generates.
    auto workloads = build(Files, 2 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));

    {
        auto pool = make_pool(config->s3_concurrency, writer, issuer);
        push_all(pool, workloads);

        // Let the reads land and their copies pile up, holding every buffer. The chunks behind them
        // have nowhere to read into and must be waiting, not failing.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        backend->opened(0)->release_copies();

        // INSIDE the pool's scope: its destructor stops the workers rather than waiting for what is
        // queued, so a response popped after it would be a response the workers never sent.
        for (unsigned i = 0; i < Files * RangesPerFile; ++i)
        {
            const auto response = responder->pop(5000);
            ASSERT_NE(response.ret, common::ResponseCode::TimedOut) << "range " << i << " was never answered";
            EXPECT_EQ(response.ret, common::ResponseCode::Success) << "range " << i;
        }
    }

    const auto device = backend->opened(0);
    ASSERT_NE(device, nullptr) << "no device was ever opened, so nothing was staged";

    // ONE COPY PER CHUNK, and a chunk is a span of whole ranges: every file here is smaller than the
    // block size, so its four ranges pack into one read and therefore one copy. A range answered
    // without a copy behind it would mean the plugin had written to the caller's device pointer -
    // the segmentation fault this whole path exists to avoid.
    EXPECT_EQ(device->copies.load(), Files) << "a chunk reached the device without a copy";
    EXPECT_GT(device->host_allocs.load(), 0u) << "nothing was pinned, so nothing was staged";

    // NOT a byte check: the s3 mock records reads and never writes into the destination, so no test
    // here can tell correct bytes from zeroed ones - a gap this path inherits rather than adds.
}

// A chunk that cannot have a staging buffer WAITS. The pool is the window plus a little, so it runs
// dry exactly when the link falls behind storage - which is a slow link, not a failed read. Failing
// the chunk there would turn a slow device into lost ranges.
// A chunk that cannot have a staging buffer WAITS. The pool is the window plus a little, so it runs
// dry exactly when the link falls behind storage - a slow device, not a failed read. Failing the chunk
// there would turn slowness into lost ranges.
// A chunk that cannot have a staging buffer WAITS. The pool is the window plus a little, so it runs
// dry exactly when the link falls behind storage - a slow device, not a failed read. Failing the chunk
// there would turn slowness into lost ranges.
//
// The worker is driven DIRECTLY rather than through a ThreadPool: the pool's destructor decides when
// its workers stop, and a teardown that lands mid-flight answers the ranges FinishedError - true, but
// it says nothing about whether a chunk waited.
// A backend with nothing in flight is not a backend that has finished.
//
// The plugin reports FinishedError whenever it has no ready event, and that is exactly what it reports
// while THIS worker is the one holding things up: chunks parked for a staging buffer were never
// submitted, so the plugin has nothing to say about them. Aborting there failed a whole submission for
// being slower at copying than at reading.
//
// Deterministic where A_Chunk_Waits_For_A_Staging_Buffer caught it only 3 times in 40: waiting until
// every buffer is out guarantees the plugin has nothing left, so the next turn takes the path.
TEST_F(ObjectStorageWorkerTest, A_Drained_Backend_Does_Not_Abort_Parked_Chunks)
{
    constexpr unsigned Files = 6;
    constexpr unsigned RangesPerFile = 2;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    backend->opened(0)->hold_copies();

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u);

    ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer);

    // Declared AFTER the worker, so it runs BEFORE ~worker: a failed ASSERT below would otherwise
    // leave quiesce_copies waiting for a copy this test is still holding, and the test would time out
    // rather than fail.
    utils::ScopeGuard release([backend]() { backend->opened(0)->release_copies(); });
    std::atomic<bool> stopped{ false };

    worker.execute(std::move(workloads[0]), stopped);

    // Every buffer out and its copy held means every read the plugin was given has completed, so it has
    // nothing ready and answers FinishedError from here on.
    for (unsigned i = 0; i < 10000 && backend->opened(0)->copies.load() < ObjectStorageWorker::CopyDepth; ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ASSERT_EQ(backend->opened(0)->copies.load(), ObjectStorageWorker::CopyDepth)
        << "the pool never ran dry, so the plugin still has reads and this test proves nothing";

    // Turns taken against a plugin that reports FinishedError every time. Chunks are parked behind the
    // held copies, so none of them may be answered.
    for (unsigned i = 0; i < 20; ++i)
    {
        worker.drain(stopped);
    }

    EXPECT_EQ(responder->pop(50).ret, common::ResponseCode::TimedOut)
        << "a parked chunk was answered while the plugin was merely idle";

    // And the work still completes once the link catches up.
    backend->opened(0)->release_copies();

    for (unsigned i = 0; i < 10000 && !worker.idle(); ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    for (unsigned i = 0; i < Files * RangesPerFile; ++i)
    {
        const auto response = responder->pop(5000);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut) << "range " << i << " was never answered";
        EXPECT_EQ(response.ret, common::ResponseCode::Success)
            << "range " << i << " was aborted by an idle plugin";
    }
}

TEST_F(ObjectStorageWorkerTest, A_Chunk_Waits_For_A_Staging_Buffer)
{
    // The pool is the plugin's window plus CopyDepth, and this mock advertises no window - so the pool
    // is CopyDepth buffers and every file past that has to wait for one.
    constexpr unsigned Files = 6;
    constexpr unsigned RangesPerFile = 2;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    // Opened here so the mock device exists before the worker starts: holding its copies is what keeps
    // the staging buffers out and drives the pool dry.
    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    backend->opened(0)->hold_copies();

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u) << "this test drives one worker, so it wants one workload";

    ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer);

    // Declared AFTER the worker, so it runs BEFORE ~worker: a failed ASSERT below would otherwise
    // leave quiesce_copies waiting for a copy this test is still holding, and the test would time out
    // rather than fail.
    utils::ScopeGuard release([backend]() { backend->opened(0)->release_copies(); });
    std::atomic<bool> stopped{ false };

    worker.execute(std::move(workloads[0]), stopped);

    // execute() submits; the plugin completes asynchronously and the worker harvests on its own turns.
    // Give it those turns, until the first reads have landed and their copies are stuck holding the
    // buffers.
    // Generous: this runs alongside the rest of the suite, and a loaded machine can take a while to
    // complete a read and give the worker a turn to harvest it.
    for (unsigned i = 0; i < 10000 && backend->opened(0)->copies.load() == 0; ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Copies are held, so the buffers never came back and the chunks behind them are waiting. Nothing
    // is answered yet, and nothing has been failed.
    EXPECT_GT(backend->opened(0)->copies.load(), 0u) << "no copy was issued, so no buffer is held";
    EXPECT_EQ(responder->pop(50).ret, common::ResponseCode::TimedOut)
        << "a waiting chunk was answered - it was failed rather than parked";

    // The link catches up: every buffer comes back, and the waiting chunks read.
    backend->opened(0)->release_copies();

    for (unsigned i = 0; i < 10000 && !worker.idle(); ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_TRUE(worker.idle()) << "a chunk waited for good";

    for (unsigned i = 0; i < Files * RangesPerFile; ++i)
    {
        const auto response = responder->pop(5000);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut) << "range " << i << " was never answered";
        EXPECT_EQ(response.ret, common::ResponseCode::Success)
            << "range " << i << " was failed for want of a buffer rather than waiting for one";
    }

    const auto device = backend->opened(0);
    EXPECT_EQ(device->copies.load(), Files) << "one chunk per file, one copy each";

    // The whole point: six files went through a pool of two buffers.
    EXPECT_LE(device->host_allocs.load(), ObjectStorageWorker::CopyDepth) << "the pool grew past its ceiling";
}

// A RETRIED chunk gives its staging buffer back before the backoff. That path does not go through the
// completion accounting, so a buffer held there is lost for the life of the worker - and the pool is
// small, so a couple of retries would leave every later chunk waiting for a buffer that no longer
// exists. Holding it would also pin memory through a backoff that another chunk could be reading into.
TEST_F(ObjectStorageWorkerTest, A_Retried_Chunk_Gives_Its_Buffer_Back)
{
    constexpr unsigned Files = 4;
    constexpr unsigned RangesPerFile = 1;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u);

    // Retries on, and the first reads fail retryably - so several chunks take a buffer, fail, and are
    // scheduled again.
    config->object_storage_retry_timeout = std::chrono::seconds(5);
    set_read_failures(Files, common::ResponseCode::RetryableFileAccessError);

    ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer);
    std::atomic<bool> stopped{ false };

    worker.execute(std::move(workloads[0]), stopped);

    for (unsigned i = 0; i < 4000 && !worker.idle(); ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_TRUE(worker.idle()) << "a retried chunk never came back - its buffer was lost";

    for (unsigned i = 0; i < Files * RangesPerFile; ++i)
    {
        const auto response = responder->pop(5000);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut)
            << "range " << i << " was never answered after its retry";
        EXPECT_EQ(response.ret, common::ResponseCode::Success) << "range " << i;
    }

    // Every failed attempt returned its buffer, so the pool never had to grow past its ceiling even
    // though twice as many attempts were made as there are chunks.
    const auto device = backend->opened(0);
    ASSERT_NE(device, nullptr);
    EXPECT_LE(device->host_allocs.load(), ObjectStorageWorker::CopyDepth)
        << "a retry took a second buffer without giving the first one back";
}

TEST_F(ObjectStorageWorkerTest, Happy_Path)
{
    set_sentinel(utils::random::boolean());

    auto workloads = build(utils::random::number(1, 10), utils::random::number(1, 6));

    {
        auto pool = make_pool(config->s3_concurrency);
        push_all(pool, workloads);

        for (unsigned file_idx = 0; file_idx < submission->paths.size(); ++file_idx)
        {
            for (unsigned i = 0; i < submission->num_chunks[file_idx]; ++i)
            {
                const auto r = responder->pop();
                EXPECT_EQ(r.ret, common::ResponseCode::Success);
                EXPECT_EQ(submission->expected[r.file_index].count(r.index), 1);
                submission->expected[r.file_index].erase(r.index);
            }
        }
    }

    for (const auto & e : submission->expected)
    {
        EXPECT_TRUE(e.empty());
    }
}

// Small ranges that share a chunk become ONE backend read, not one each.
//
// Chunks come from Batch::chunks, built at s3_block_bytesize where the tasks were cut, so consecutive
// small tensors inside one chunk are fetched together. Before that, every task was chunked on its own
// and a 100-byte tensor meant a 100-byte GET - which is what this asserts is no longer true.
TEST_F(ObjectStorageWorkerTest, Small_Ranges_Are_Packed_Into_One_Read)
{
    constexpr size_t s3_chunk = 64 * 1024;
    constexpr unsigned num_ranges = 40;
    constexpr size_t range_size = 100;

    // s3_block_bytesize is the 3rd argument, and the 4th is fs_sync_read_block_bytesize, which this
    // test does not care about. enforce_minimum=false is what keeps s3_chunk from being floored at
    // 5 MiB.
    config = std::make_shared<Config>(1, 1, s3_chunk, 1024, false);
    responder = std::make_shared<common::Responder>(0);

    const std::string path = "s3://" + utils::random::string() + "/" + utils::random::string();
    std::vector<char> buffer(num_ranges * range_size);

    FileRanges ranges;
    ranges.path = path;
    for (unsigned i = 0; i < num_ranges; ++i)
    {
        ranges.ranges.push_back(ReadRange{ i * range_size, range_size, buffer.data() + i * range_size });
    }
    responder->increment(num_ranges);

    std::vector<FileRanges> request = { ranges };
    Assigner assigner(request, config);

    std::vector<Workload> workloads(assigner.num_workloads());
    const common::s3::Credentials credentials;
    for (const auto & transfer : assigner.transfers())
    {
        auto uri = std::make_shared<common::s3::StorageUri>(path);
        common::s3::S3ClientWrapper::Params params(uri, credentials, config->s3_block_bytesize, config->s3_concurrency);

        Batches batches(utils::random::number(), transfer.file_index, transfer.tasks, config, responder,
                        path, params, transfer.range_sizes, transfer.first_range_index, common::Device::host());
        for (size_t j = 0; j < batches.size(); ++j)
        {
            workloads[batches[j].workload_index].add_batch(std::move(batches[j]));
        }
    }

    {
        auto pool = make_pool(1);
        push_all(pool, workloads);

        // every range still owes exactly one response, however few reads served them
        for (unsigned i = 0; i < num_ranges; ++i)
        {
            EXPECT_EQ(responder->pop().ret, common::ResponseCode::Success);
        }
    }

    // 40 x 100 bytes all sit inside one 64 KiB chunk.
    EXPECT_EQ(requests(), 1u) << num_ranges << " small ranges should be one read, not one each";
}

// A failing file still owes a response for every one of its ranges, and must not take the others
// with it.
//
// This is the path where a response can be lost without anything looking wrong: a failed chunk does
// NOT answer its tasks - complete_chunk only records the code in error_by_file_index - so the response
// is pushed later, by finalize -> report_workload -> handle_error. A workload that failed to finalize
// would deliver every successful range and silently drop every failed one, and the caller would wait
// forever for responses that never come.
TEST_F(ObjectStorageWorkerTest, Failing_File_Still_Answers_Every_Range)
{
    // Several files, so the failure has neighbours it must not affect.
    auto workloads = build(4 /* files */, 2 /* s3_concurrency */);

    // Fail exactly one of them, by a substring of its path.
    const auto & doomed = submission->paths[utils::random::number<unsigned>(0, 3)];
    const auto key = doomed.substr(doomed.rfind('/') + 1);
    fail_paths_containing(key.c_str());

    unsigned failed = 0;
    unsigned succeeded = 0;

    {
        auto pool = make_pool(config->s3_concurrency);
        push_all(pool, workloads);

        // EVERY range of EVERY file, failing or not, owes exactly one response.
        for (unsigned file_idx = 0; file_idx < submission->paths.size(); ++file_idx)
        {
            for (unsigned i = 0; i < submission->num_chunks[file_idx]; ++i)
            {
                const auto r = responder->pop();

                // identity, not a count: the right range of the right file, exactly once
                EXPECT_EQ(submission->expected[r.file_index].count(r.index), 1)
                    << "response for file " << r.file_index << " range " << r.index
                    << " was unexpected or already seen";
                submission->expected[r.file_index].erase(r.index);

                const bool is_doomed = submission->paths[r.file_index] == doomed;
                if (is_doomed)
                {
                    EXPECT_NE(r.ret, common::ResponseCode::Success) << "the failing file must not report success";
                    ++failed;
                }
                else
                {
                    EXPECT_EQ(r.ret, common::ResponseCode::Success) << "one file's failure must not touch another";
                    ++succeeded;
                }
            }
        }
    }

    // Nothing left owing - which is what a lost response would show up as.
    for (const auto & e : submission->expected)
    {
        EXPECT_TRUE(e.empty());
    }

    EXPECT_GT(failed, 0u) << "the failure never fired, so this asserted nothing";
    EXPECT_GT(succeeded, 0u) << "every file failed, so the isolation was not tested";
}

// Tearing the pool down while reads are in flight: every request still gets exactly one response (Success
// or FinishedError) - none is lost or double-counted - and the pool joins without hanging.
TEST_F(ObjectStorageWorkerTest, Stopped_Mid_Stream)
{
    set_response_time(1000);   // slow completions -> reads stay in flight

    auto workloads = build(utils::random::number(1, 10), 1);

    {
        auto pool = make_pool(config->s3_concurrency);
        push_all(pool, workloads);

        ::usleep(utils::random::number(100));

        // unblock the workers parked in async_response, then let ~pool set stopped and join
        common::s3::S3ClientWrapper::stop();

        for (unsigned processed = 0; processed < submission->total_requests(); ++processed)
        {
            const auto r = responder->pop();
            EXPECT_TRUE(r.ret == common::ResponseCode::Success || r.ret == common::ResponseCode::FinishedError);
            EXPECT_LT(r.file_index, submission->paths.size());
            EXPECT_LT(r.index, submission->num_chunks[r.file_index]);
            EXPECT_EQ(submission->expected[r.file_index].count(r.index), 1);
            submission->expected[r.file_index].erase(r.index);
        }
    }

    for (const auto & e : submission->expected)
    {
        EXPECT_TRUE(e.empty());
    }
}

// The worker never exceeds the plugin's in-flight window: with one worker (one client) and a bounded
// window, the mock's peak per-client in-flight stays within the window's chunk count.
TEST_F(ObjectStorageWorkerTest, Window_Bounded)
{
    auto workloads = build(utils::random::number(3, 10), 1);

    const size_t window_bytes = config->s3_block_bytesize * utils::random::number<size_t>(2, 6);
    set_window(window_bytes);
    const size_t window_chunks = std::max<size_t>(1, window_bytes / config->s3_block_bytesize);

    {
        auto pool = make_pool(config->s3_concurrency);
        push_all(pool, workloads);

        for (unsigned processed = 0; processed < submission->total_requests(); ++processed)
        {
            EXPECT_EQ(responder->pop().ret, common::ResponseCode::Success);
        }
    }

    const auto peak = max_concurrent();
    EXPECT_LE(peak, window_chunks);
    EXPECT_GT(peak, 0u);
}

// A terminal backend error marked retryable requeues only that ObjectChunk. Every already-completed chunk
// stays complete: total backend submissions grow by exactly one, and all public responses remain Success.
TEST_F(ObjectStorageWorkerTest, RetryableChunkIsRequeuedWithoutRestartingWorkload)
{
    auto workloads = build(1, 1);
    const size_t initial_chunks = count_object_chunks(workloads);
    config->object_storage_retry_timeout = std::chrono::seconds(5);
    set_read_failures(1, common::ResponseCode::RetryableFileAccessError);

    {
        auto pool = make_pool(1);
        push_all(pool, workloads);
        for (unsigned i = 0; i < submission->total_requests(); ++i)
        {
            EXPECT_EQ(responder->pop().ret, common::ResponseCode::Success);
        }
    }

    EXPECT_EQ(total_read_requests(), initial_chunks + 1);
}

// With the application retry timeout disabled, even an internal retryable marker is converted to the
// original public FileAccessError and the worker does not submit another backend request.
TEST_F(ObjectStorageWorkerTest, RetryableChunkFailsWithoutRetryWhenTimeoutDisabled)
{
    auto workloads = build(1, 1, 1, SingleChunkBlockBytesize);   // exactly one backend chunk
    const size_t initial_chunks = count_object_chunks(workloads);
    ASSERT_EQ(initial_chunks, 1u);
    ASSERT_EQ(config->object_storage_retry_timeout.count(), 0);
    set_read_failures(1, common::ResponseCode::RetryableFileAccessError);

    {
        auto pool = make_pool(1);
        push_all(pool, workloads);
        EXPECT_EQ(responder->pop().ret, common::ResponseCode::FileAccessError);
    }

    EXPECT_EQ(total_read_requests(), initial_chunks);
}

// A permanent storage error bypasses the application retry loop even when a retry deadline is configured.
TEST_F(ObjectStorageWorkerTest, PermanentChunkErrorFailsWithoutRetry)
{
    auto workloads = build(1, 1);
    const size_t initial_chunks = count_object_chunks(workloads);
    config->object_storage_retry_timeout = std::chrono::seconds(5);
    set_read_failures(1, common::ResponseCode::FileAccessError);

    bool saw_file_access_error = false;
    {
        auto pool = make_pool(1);
        push_all(pool, workloads);
        for (unsigned i = 0; i < submission->total_requests(); ++i)
        {
            const auto ret = responder->pop().ret;
            saw_file_access_error = saw_file_access_error || ret == common::ResponseCode::FileAccessError;
            EXPECT_TRUE(ret == common::ResponseCode::Success || ret == common::ResponseCode::FileAccessError);
        }
    }

    EXPECT_TRUE(saw_file_access_error);
    EXPECT_EQ(total_read_requests(), initial_chunks);
}

// Once a chunk's retry deadline (started at its first backend submission) has expired, the internal
// retryable marker is converted to the public FileAccessError and no new backend attempt is submitted.
TEST_F(ObjectStorageWorkerTest, ExpiredChunkRetryDeadlineReturnsFileAccessError)
{
    auto workloads = build(1, 1, 1, SingleChunkBlockBytesize);   // exactly one backend chunk
    config->object_storage_retry_timeout = std::chrono::seconds(1);
    const size_t initial_chunks = count_object_chunks(workloads);
    ASSERT_EQ(initial_chunks, 1u);
    set_response_time(1100);   // the first attempt completes after its per-chunk deadline
    set_read_failures(1, common::ResponseCode::RetryableFileAccessError);

    bool saw_file_access_error = false;
    {
        auto pool = make_pool(1);
        push_all(pool, workloads);
        for (unsigned i = 0; i < submission->total_requests(); ++i)
        {
            const auto ret = responder->pop().ret;
            saw_file_access_error = saw_file_access_error || ret == common::ResponseCode::FileAccessError;
            EXPECT_NE(ret, common::ResponseCode::RetryableFileAccessError);
        }
    }

    EXPECT_TRUE(saw_file_access_error);
    EXPECT_EQ(total_read_requests(), initial_chunks);
}

// Time spent before ObjectStorageWorker first submits a chunk does not consume its retry budget. Even after
// waiting longer than the configured timeout, the first retryable completion is requeued and succeeds.
TEST_F(ObjectStorageWorkerTest, RetryDeadlineStartsAtFirstChunkSubmission)
{
    auto workloads = build(1, 1, 1, SingleChunkBlockBytesize);   // exactly one backend chunk
    config->object_storage_retry_timeout = std::chrono::seconds(1);
    const size_t initial_chunks = count_object_chunks(workloads);
    ASSERT_EQ(initial_chunks, 1u);
    set_read_failures(1, common::ResponseCode::RetryableFileAccessError);

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    {
        auto pool = make_pool(1);
        push_all(pool, workloads);
        for (unsigned i = 0; i < submission->total_requests(); ++i)
        {
            EXPECT_EQ(responder->pop().ret, common::ResponseCode::Success);
        }
    }

    EXPECT_EQ(total_read_requests(), initial_chunks + 1);
}

// New submissions pushed WHILE the consumer is draining earlier ones (from a separate thread) all complete:
// the interleaving worker picks up newly arrived workloads without waiting for in-flight ones to finish.
// Every response is checked against its expected (submission, file, request) - not just the total count.
TEST_F(ObjectStorageWorkerTest, Concurrent_Submissions)
{
    make_context(utils::random::number(2, 6));

    // build several submissions up front (each grows the shared responder's expected count). Distinct
    // submission ids (the loop index) let us verify every response back to its exact request.
    const unsigned num_submissions = utils::random::number(2, 5);
    std::vector<std::unique_ptr<Submission>> submissions;
    std::vector<std::vector<Workload>> workloads(num_submissions);
    std::map<std::pair<SubmissionId, unsigned>, std::set<int>> outstanding;   // (submission_id, file_index) -> indices
    unsigned total = 0;
    for (unsigned s = 0; s < num_submissions; ++s)
    {
        submissions.push_back(std::make_unique<Submission>(/*submission_id=*/s, utils::random::number(1, 6), config, responder));
        workloads[s] = submissions[s]->build();
        total += submissions[s]->total_requests();
        for (unsigned f = 0; f < submissions[s]->expected.size(); ++f)
        {
            outstanding[{s, f}] = submissions[s]->expected[f];
        }
    }

    {
        auto pool = make_pool(config->s3_concurrency);
        push_all(pool, workloads[0]);   // first submission on this thread

        // the rest are dispatched concurrently, while the loop below is popping responses
        auto pusher = utils::Thread([&]()
        {
            for (unsigned s = 1; s < num_submissions; ++s)
            {
                ::usleep(utils::random::number(200));
                push_all(pool, workloads[s]);
            }
        });

        for (unsigned processed = 0; processed < total; ++processed)
        {
            const auto r = responder->pop();
            EXPECT_EQ(r.ret, common::ResponseCode::Success);

            auto it = outstanding.find({ r.submission_id, r.file_index });
            ASSERT_NE(it, outstanding.end()) << "response for unknown submission " << r.submission_id << " file " << r.file_index;
            EXPECT_EQ(it->second.erase(r.index), 1u) << "unexpected/duplicate request index " << r.index;
        }

        pusher.join();
    }

    // every expected response arrived exactly once, so nothing is left outstanding and the responder drained
    for (const auto & [key, indices] : outstanding)
    {
        EXPECT_TRUE(indices.empty()) << "submission " << key.first << " file " << key.second << " missing responses";
    }
    EXPECT_EQ(responder->pop().ret, common::ResponseCode::FinishedError);
}

// An empty workload (no batches) is a no-op: the worker logs a warning, builds no client, issues no reads,
// and stays idle so the pool joins cleanly. Production never dispatches empty workloads (they are skipped
// at dispatch), but the worker must not crash or hang on one.
TEST_F(ObjectStorageWorkerTest, Empty_Workload)
{
    {
        auto pool = make_pool(utils::random::number(1, 4));
        pool.push(Workload{});
        pool.push(Workload{});
    }

    // no client was ever created and nothing was read
    EXPECT_EQ(clients(), 0);
}

// An abort does not return while a copy is still in flight.
//
// abort_all fails every workload and drops them, but a copy already handed to the issuer is NOT
// cancelled. It completes later on the waiter's thread and calls back into THIS worker, pushing onto a
// member deque - so a worker torn down first is written through after it is gone. The copy is also
// still writing into the caller's device memory, and a response promises that nothing will write to
// that range again.
//
// Before the fix this returned at once, having set the in-flight count to zero.
TEST_F(ObjectStorageWorkerTest, An_Abort_Waits_For_A_Copy_In_Flight)
{
    constexpr unsigned Files = 2;
    constexpr unsigned RangesPerFile = 2;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    backend->opened(0)->hold_copies();

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u) << "this test drives one worker, so it wants one workload";

    ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer);

    // Declared AFTER the worker, so it runs BEFORE ~worker: a failed ASSERT below would otherwise
    // leave quiesce_copies waiting for a copy this test is still holding, and the test would time out
    // rather than fail.
    utils::ScopeGuard release([backend]() { backend->opened(0)->release_copies(); });
    std::atomic<bool> stopped{ false };

    worker.execute(std::move(workloads[0]), stopped);

    // Turns until a read has landed and its copy is stuck in the issuer, holding its buffer.
    for (unsigned i = 0; i < 10000 && backend->opened(0)->copies.load() == 0; ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GT(backend->opened(0)->copies.load(), 0u) << "no copy was issued, so nothing is in flight";

    // Abort on another thread, because it is supposed to block.
    std::atomic<bool> returned{ false };
    std::thread aborting([&]()
        {
            stopped = true;
            worker.drain(stopped);
            returned = true;
        });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(returned.load())
        << "the abort returned with a copy still in flight: the worker may now be destroyed, and the"
        << " copy's completion would call back into freed memory";

    backend->opened(0)->release_copies();
    aborting.join();

    EXPECT_TRUE(returned.load());
    EXPECT_TRUE(worker.idle()) << "the abort left work behind";
}


// An abort does not release a staging buffer while the backend is still reading into it.
//
// async_read is given buffer.data and the plugin fills it asynchronously. abort_all used to release
// every staging buffer straight back to the pool, so the next chunk of this same worker could acquire
// memory the plugin was still writing into - and at teardown the pool frees that pinned memory
// outright. The reads are not cancelled, so the only safe order is to wait for them first.
//
// Before the fix this returned at once, having cleared the in-flight credit.
TEST_F(ObjectStorageWorkerTest, An_Abort_Waits_For_Reads_Still_At_The_Backend)
{
    constexpr unsigned Files = 2;
    constexpr unsigned RangesPerFile = 2;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    // The plugin takes its time, so a read is still outstanding when the abort lands. The mock spends
    // about this long inside each obj_wait_for_completions.
    constexpr unsigned HarvestRoundMs = 50;
    set_response_time(HarvestRoundMs);

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u) << "this test drives one worker, so it wants one workload";

    ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer);
    std::atomic<bool> stopped{ false };

    worker.execute(std::move(workloads[0]), stopped);
    ASSERT_GT(requests(), 0u) << "no read was submitted, so nothing is at the backend";

    // Abort while the backend still holds reads. It must not return until they have reported, and the
    // mock makes every harvest round cost real time - so an abort that waits cannot be instant.
    stopped = true;

    const auto start = std::chrono::steady_clock::now();
    worker.drain(stopped);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    EXPECT_GE(elapsed, HarvestRoundMs / 2)
        << "the abort returned in " << elapsed << " ms, without waiting for a harvest round: its reads"
        << " were still at the backend and their staging buffers went back to the pool while the plugin"
        << " was still writing into them";

    EXPECT_TRUE(worker.idle()) << "the abort left reads outstanding";

    set_response_time(0);
}

// A staging pool is not freed with the worker that made it.
//
// The plugin is given the pool's pinned memory and fills it asynchronously. A sent request cannot be
// cancelled, and removing a client only parks it - the client's destructor, which is what waits for
// the SDK, runs later. So the pool has to outlive the worker, and the streamer holds it until the
// backend has been cleaned up.
//
// Measured here as "no host_free while the worker is destroyed", because freeing the slab is exactly
// what would hand the plugin freed memory.
TEST_F(ObjectStorageWorkerTest, A_Staging_Pool_Outlives_Its_Worker)
{
    constexpr unsigned Files = 2;
    constexpr unsigned RangesPerFile = 2;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);
    auto retainer = std::make_shared<StagingPoolRetainer>();

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u);

    {
        ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer, retainer);
        std::atomic<bool> stopped{ false };

        worker.execute(std::move(workloads[0]), stopped);

        for (unsigned i = 0; i < 10000 && !worker.idle(); ++i)
        {
            worker.drain(stopped);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        ASSERT_GT(backend->opened(0)->host_allocs.load(), 0u) << "nothing was pinned, so nothing is retained";
    }

    const auto device = backend->opened(0);

    // THE POINT. The worker is gone and its _pool member with it, but the retainer still holds the
    // pool - so not one slab has been freed.
    EXPECT_EQ(device->host_frees.load(), 0u)
        << "a staging slab was freed with its worker: the plugin's client is only parked at that"
        << " point, so a read still at the backend would be writing into freed pinned memory";

    // And it is the retainer holding it: dropping that is what frees them.
    retainer.reset();

    EXPECT_EQ(device->host_frees.load(), device->host_allocs.load())
        << "dropping the retainer did not free the slabs, so something else still holds the pool";
}

// A staging buffer whose read never reported is NOT given back to the pool.
//
// THE BUG THIS REPRODUCES: quiesce_reads() counted every event the plugin handed it, including the
// FinishedError marker that azure and gcs append when their ready queue empties. A stopped responder
// produces nothing but that marker, so the loop drove the in-flight credit to zero without a single
// read landing, declared itself drained, and abort_all handed the buffers back. A later chunk could
// then acquire memory the plugin was still writing into.
//
// Driven with the sentinel enabled, because that is the shape that returns Success WITH a marker
// event - the case the old code miscounted. The worker keeps running afterwards, which is what makes
// a recycled buffer reachable.
TEST_F(ObjectStorageWorkerTest, A_Buffer_Whose_Read_Never_Reported_Is_Not_Reused)
{
    constexpr unsigned Files = 2;
    constexpr unsigned RangesPerFile = 1;

    set_sentinel(true);
    set_response_time(60000);   // no read completes during this test

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);
    auto retainer = std::make_shared<StagingPoolRetainer>();

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u);

    ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer, retainer);
    std::atomic<bool> stopped{ false };

    worker.execute(std::move(workloads[0]), stopped);
    ASSERT_GT(requests(), 0u) << "no read was submitted, so no buffer is held by one";

    // The reads are now at the backend, holding their staging buffers. Stopping the client makes the
    // responder answer with the marker and nothing else - exactly what teardown does, but with this
    // worker still alive.
    common::s3::S3ClientWrapper::stop();

    // NOT the stopped flag: this is a mid-life abort, so the worker survives and keeps its pool.
    worker.drain(stopped);

    std::shared_ptr<StagingPool> pool;
    ASSERT_TRUE(retainer->try_pop(pool)) << "no staging pool was ever built";
    ASSERT_NE(pool, nullptr);

    EXPECT_GT(pool->retired(), 0u)
        << "a staging buffer went back to the pool although its read never reported: the plugin may"
        << " still be writing into it, and the next chunk to acquire it would read torn bytes";

    EXPECT_EQ(pool->retired(), pool->created())
        << "every buffer this pool made was held by an unreported read, so none may be handed out again";

    // AND THE CALLER IS TOLD. A retirement is never silent: the ranges whose buffers were taken out of
    // circulation are failed in the same abort, so capacity is never lost behind the caller's back.
    for (unsigned i = 0; i < Files * RangesPerFile; ++i)
    {
        const auto response = responder->pop(5000);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut)
            << "range " << i << " was never answered, so the retirement was silent";
        // THE ABORT'S OWN CODE, not a device one: what failed here is the object backend, which stopped
        // answering while its reads were outstanding. DeviceDriverError is what the two device paths
        // report, because there the thing that may still be writing is a copy.
        EXPECT_EQ(response.ret, common::ResponseCode::FinishedError)
            << "range " << i << " was not failed with the code the abort carried";
    }

    set_response_time(0);
}

// idle() must stay FALSE while this worker still owes something. The pool reads it to decide a worker
// has nothing left, so an idle() that lies lets teardown start under live work.
//
// THE SUBTLE CASE IS THE COPY. issue_copy gives the window slot back as soon as the read lands, so the
// capacity queue reports idle while the copy is still in flight. Only has_deferred_work() keeps the
// answer honest. A reader who assumes "queue idle means worker idle" would get this wrong.
TEST_F(ObjectStorageWorkerTest, Idle_Is_False_While_Work_Is_Outstanding)
{
    constexpr unsigned Files = 2;
    constexpr unsigned RangesPerFile = 1;

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend](common::DeviceType) -> DeviceWriter::BackendFactory { return [backend]() { return backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    backend->opened(0)->hold_copies();

    auto workloads = build(Files, 1 /* s3 concurrency */, RangesPerFile, SingleChunkBlockBytesize,
                           common::Device::cuda(0));
    ASSERT_EQ(workloads.size(), 1u);

    ObjectStorageWorker worker([]() { return common::s3::Credentials{}; }, writer, issuer);

    // Declared AFTER the worker so it runs BEFORE ~worker: a failed ASSERT below would otherwise leave
    // quiesce_copies waiting for a copy this test is still holding.
    utils::ScopeGuard release([backend]() { backend->opened(0)->release_copies(); });

    std::atomic<bool> stopped{ false };

    EXPECT_TRUE(worker.idle()) << "a worker with no window yet owes nothing";

    worker.execute(std::move(workloads[0]), stopped);

    EXPECT_FALSE(worker.idle()) << "reads are in flight, so this worker still owes responses";

    // Turn until a copy is held. The read has landed by then and its window slot is back, so the
    // capacity queue alone would say idle.
    for (unsigned i = 0; i < 10000 && backend->opened(0)->copies.load() == 0; ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GT(backend->opened(0)->copies.load(), 0u) << "no copy was issued, so the point is untested";

    EXPECT_FALSE(worker.idle())
        << "a copy is still in flight: its range is unanswered and its completion will call back into"
        << " this worker, so the pool must not treat it as finished";

    backend->opened(0)->release_copies();

    for (unsigned i = 0; i < 10000 && !worker.idle(); ++i)
    {
        worker.drain(stopped);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_TRUE(worker.idle()) << "everything completed, so the worker owes nothing";
}

}; // namespace runai::llm::streamer::impl
