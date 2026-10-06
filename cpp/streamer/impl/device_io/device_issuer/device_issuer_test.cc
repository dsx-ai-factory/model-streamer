/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/device_io/device_issuer/device_issuer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "device/mock/mock_device.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr size_t Buffer = 4096;

bool eventually(const std::function<bool()> & holds)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (holds())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

class DeviceIssuerTest : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        _writer = std::make_shared<DeviceWriter>([this](common::DeviceType) -> DeviceWriter::BackendFactory { return [this]() { return _backend; }; });
    }

    // A reading thread's own pool: small, and its own, which is the whole point of the split.
    std::shared_ptr<StagingPool> pool_for(unsigned ordinal, unsigned buffers)
    {
        DeviceWriter::Channel channel = nullptr;
        EXPECT_EQ(_writer->open(common::Device::cuda(ordinal), channel), common::ResponseCode::Success);

        StagingPool::Params params;
        params.buffer_bytesize = Buffer;
        params.slab_bytesize = buffers * Buffer;
        params.max_buffers = buffers;
        return std::make_shared<StagingPool>(_writer->device(channel), params);
    }

    std::shared_ptr<device::MockBackend> _backend = std::make_shared<device::MockBackend>();
    std::shared_ptr<DeviceWriter> _writer;
};

} // namespace

// A load whose destinations are all host memory never submits, so the thread is never started and
// the driver is never reached.
TEST_F(DeviceIssuerTest, Starts_Nothing_Until_The_First_Submit)
{
    DeviceIssuer issuer(_writer);

    EXPECT_FALSE(issuer.running());
    EXPECT_EQ(issuer.issued(), 0u);
    EXPECT_EQ(issuer.devices(), 0u);
    EXPECT_EQ(_backend->opens, 0u);
}

// The reader hands over a full buffer and goes back to reading; the copy happens on the issuer's
// thread and the buffer comes back to the reader's own pool.
TEST_F(DeviceIssuerTest, Copies_And_Returns_The_Buffer_To_Its_Own_Pool)
{
    auto pool = pool_for(0, 1);
    DeviceIssuer issuer(_writer);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());
    std::memset(buffer.data, 0x5a, Buffer);

    std::vector<char> destination(Buffer, 0);
    std::atomic<int> reported{-1};

    issuer.submit(common::Device::cuda(0), pool, buffer, Buffer, destination.data(),
                  [&](common::ResponseCode ret) { reported.store(static_cast<int>(ret)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);
    EXPECT_EQ(destination[0], 0x5a);
    EXPECT_EQ(destination[Buffer - 1], 0x5a);

    // One buffer in that pool, so getting it again proves it went back to the RIGHT pool.
    StagingBuffer again;
    ASSERT_TRUE(eventually([&]()
        {
            return pool->try_acquire(again) == common::ResponseCode::Success && again.valid();
        }));
    EXPECT_EQ(again.data, buffer.data);
}

// THE arrangement this class exists for: a pool per reading thread, one issuer, one stream. Every
// buffer must come back to the pool it came from, whichever thread submitted it.
TEST_F(DeviceIssuerTest, Many_Readers_Each_With_Its_Own_Pool)
{
    constexpr unsigned Readers = 8;
    constexpr unsigned PerReader = 50;
    constexpr unsigned Buffers = 3;   // the sync reader's pipeline: fill, copy, in flight

    DeviceIssuer issuer(_writer);

    std::vector<std::shared_ptr<StagingPool>> pools;
    for (unsigned r = 0; r < Readers; ++r)
    {
        pools.push_back(pool_for(0, Buffers));
    }

    // One destination per reader, so a copy landing on the wrong bytes is visible.
    std::vector<std::vector<char>> destinations(Readers, std::vector<char>(Buffer, 0));
    std::atomic<unsigned> done{0};

    std::vector<std::thread> readers;
    for (unsigned r = 0; r < Readers; ++r)
    {
        readers.emplace_back([&, r]()
            {
                for (unsigned i = 0; i < PerReader; ++i)
                {
                    StagingBuffer buffer;
                    // Blocking, because a reading thread has nothing else to do - unlike an async
                    // engine, which has I/O to wait for.
                    ASSERT_EQ(pools[r]->acquire(buffer), common::ResponseCode::Success);
                    ASSERT_TRUE(buffer.valid()) << "reader " << r << " round " << i;

                    std::memset(buffer.data, static_cast<int>('a' + r), Buffer);
                    issuer.submit(common::Device::cuda(0), pools[r], buffer, Buffer, destinations[r].data(),
                                  [&](common::ResponseCode ret)
                                  {
                                      EXPECT_EQ(ret, common::ResponseCode::Success);
                                      ++done;
                                  });
                }
            });
    }
    for (auto & reader : readers)
    {
        reader.join();
    }

    ASSERT_TRUE(eventually([&]() { return done.load() == Readers * PerReader; }));

    // AFTER the completions, not with them: the lane counts a copy as issued once its handler
    // returns, and the completion fires from the StreamWaiter inside that handler. The last one can
    // still be un-counted here.
    ASSERT_TRUE(eventually([&]() { return issuer.issued() == Readers * PerReader; }));
    EXPECT_EQ(_backend->opened(0)->streams_created, 1u) << "one device, one stream, one issuer";
    EXPECT_EQ(_backend->opened(0)->copies, Readers * PerReader);

    for (unsigned r = 0; r < Readers; ++r)
    {
        EXPECT_EQ(destinations[r][0], 'a' + static_cast<int>(r)) << "reader " << r;
        EXPECT_EQ(pools[r]->created(), Buffers) << "reader " << r << " stayed inside its own pool";
    }
}

// A copy that cannot be issued still reports. The reader has already moved on, so the completion is
// the only way it can ever learn - unlike DeviceWriter::write, which answers its caller directly.
TEST_F(DeviceIssuerTest, A_Copy_That_Cannot_Be_Issued_Still_Reports)
{
    auto pool = pool_for(0, 1);
    DeviceIssuer issuer(_writer);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    std::vector<char> destination(Buffer, 0);
    std::atomic<int> reported{-1};

    // Ordinal 9 does not exist - the mock backend has four devices - so no lane can be built.
    issuer.submit(common::Device::cuda(9), pool, buffer, Buffer, destination.data(),
                  [&](common::ResponseCode ret) { reported.store(static_cast<int>(ret)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_NE(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);
    EXPECT_EQ(issuer.devices(), 0u) << "a device that cannot be opened leaves no lane behind";

    // And the buffer came back, so the reader is not left short.
    StagingBuffer again;
    ASSERT_EQ(pool->try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid());
}

// A RETIRED BUFFER IS STILL REPORTED. DeviceWriter::write keeps the buffer when it cannot drain the
// stream, and answers its caller - but the reader moved on the moment this was queued, so the issuer
// is the only one left to deliver that answer.
//
// The point for the API: capacity is never lost behind the caller's back. Every retirement costs a
// range, and that range is failed with the code that says its destination may still be written.
TEST_F(DeviceIssuerTest, A_Copy_Whose_Buffer_Is_Retired_Still_Reports)
{
    auto pool = pool_for(0, 1);
    DeviceIssuer issuer(_writer);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    // The copy is enqueued, its event cannot be recorded, and the stream cannot be drained - the one
    // branch in write() that retires rather than releases.
    auto device = _backend->opened(0);
    ASSERT_NE(device, nullptr);
    device->fail_event_record = true;
    device->fail_stream_synchronize = true;

    std::vector<char> destination(Buffer, 0);
    std::atomic<int> reported{-1};

    issuer.submit(common::Device::cuda(0), pool, buffer, Buffer, destination.data(),
                  [&](common::ResponseCode ret) { reported.store(static_cast<int>(ret)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::DeviceDriverError);

    // And the buffer did NOT come back, which is what the report is paying for.
    EXPECT_EQ(pool->retired(), 1u);

    device->fail_event_record = false;
    device->fail_stream_synchronize = false;
}

// Teardown waits for what is queued, so no copy is dropped and no completion is lost.
TEST_F(DeviceIssuerTest, Teardown_Issues_What_Is_Already_Queued)
{
    auto pool = pool_for(0, 4);
    std::atomic<unsigned> done{0};

    // OUTSIDE the scope, deliberately. The destructor below issues the copies that are still queued,
    // so a destination declared after the issuer would be destroyed first and written into after it
    // was freed. Production has the same rule the other way round: a caller must not free a
    // destination with a copy in flight, which is what quiesce() exists to guarantee.
    std::vector<char> destination(4 * Buffer, 0);

    {
        DeviceIssuer issuer(_writer);

        for (unsigned i = 0; i < 4; ++i)
        {
            StagingBuffer buffer;
            ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
            issuer.submit(common::Device::cuda(0), pool, buffer, Buffer, destination.data() + i * Buffer,
                          [&](common::ResponseCode) { ++done; });
        }
    }

    // ~DeviceIssuer drained its queue; the writer's waiter still has to retire the copies.
    ASSERT_TRUE(eventually([&]() { return done.load() == 4u; }));
    EXPECT_EQ(_backend->opened(0)->copies, 4u);
}

// THE reason the lanes are per device: checkpoint/restore runs one streamer over every GPU on the
// node. Each device gets its own thread, so no thread alternates between contexts and no device
// waits behind another's enqueues.
TEST_F(DeviceIssuerTest, One_Thread_Per_Device)
{
    constexpr unsigned Devices = 4;
    constexpr unsigned PerDevice = 25;

    DeviceIssuer issuer(_writer);

    // One reader with one pool, submitting to every device in turn - the interleaving that would
    // make a single shared thread switch context on every copy.
    auto pool = pool_for(0, 4);

    std::vector<std::vector<char>> destinations(Devices, std::vector<char>(Buffer, 0));
    std::atomic<unsigned> done{0};

    for (unsigned round = 0; round < PerDevice; ++round)
    {
        for (unsigned d = 0; d < Devices; ++d)
        {
            StagingBuffer buffer;
            ASSERT_EQ(pool->acquire(buffer), common::ResponseCode::Success);
            ASSERT_TRUE(buffer.valid());

            std::memset(buffer.data, static_cast<int>('a' + d), Buffer);
            issuer.submit(common::Device::cuda(d), pool, buffer, Buffer, destinations[d].data(),
                          [&](common::ResponseCode ret)
                          {
                              EXPECT_EQ(ret, common::ResponseCode::Success);
                              ++done;
                          });
        }
    }

    ASSERT_TRUE(eventually([&]() { return done.load() == Devices * PerDevice; }));

    EXPECT_EQ(issuer.devices(), Devices) << "a lane, and a thread, for each";
    ASSERT_TRUE(eventually([&]() { return issuer.issued() == Devices * PerDevice; }));

    for (unsigned d = 0; d < Devices; ++d)
    {
        EXPECT_EQ(destinations[d][0], 'a' + static_cast<int>(d)) << "device " << d;
        EXPECT_EQ(_backend->opened(d)->copies, PerDevice) << "device " << d;
        EXPECT_EQ(_backend->opened(d)->streams_created, 1u) << "device " << d;

        // One bind per thread that reaches this device - the opener, its stream waiter and its lane -
        // and none per copy. A shared lane would bind again on every copy that changed device.
        EXPECT_LE(_backend->opened(d)->bind_calls, 3u) << "device " << d << " rebound per copy";
    }

    EXPECT_EQ(pool->created(), 4u) << "one reader, one pool, whatever devices it named";
}

} // namespace runai::llm::streamer::impl
