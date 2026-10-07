/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/device_io/device_writer_client/device_writer_client.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <functional>
#include <vector>

#include "device/mock/mock_device.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr size_t Buffer = 4096;

DeviceWriterClient::Buffers geometry(size_t slab = Buffer)
{
    DeviceWriterClient::Buffers b;
    b.buffer_bytesize = Buffer;
    b.slab_bytesize = slab;
    return b;
}

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

class DeviceWriterClientTest : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        _writer = std::make_shared<DeviceWriter>([this](common::DeviceType) -> DeviceWriter::BackendFactory { return [this]() { ++_factory_calls; return _backend; }; });
    }

    std::shared_ptr<device::MockBackend> _backend = std::make_shared<device::MockBackend>();
    std::shared_ptr<DeviceWriter> _writer;
    std::atomic<unsigned> _factory_calls{0};
};

} // namespace

// Every reader owns one of these, so a load whose destinations are all host memory must pay nothing
// for it: no pinned memory, no stream, no thread, and no driver.
TEST_F(DeviceWriterClientTest, Costs_Nothing_Until_The_First_Take)
{
    DeviceWriterClient client(_writer, geometry(), 8);

    EXPECT_EQ(_factory_calls.load(), 0u) << "the driver was reached before any device was named";
    EXPECT_EQ(_backend->opens, 0u);
    EXPECT_EQ(client.devices(), 0u);
    EXPECT_EQ(client.buffers(), 0u);
}

// take() is the first touch: it opens the device and builds the pool, because pinned memory needs a
// context and this is the first call that knows one.
TEST_F(DeviceWriterClientTest, The_First_Take_Opens_The_Device_And_Builds_The_Pool)
{
    DeviceWriterClient client(_writer, geometry(4 * Buffer), 4);

    StagingBuffer buffer;
    ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    EXPECT_EQ(client.devices(), 1u);
    EXPECT_EQ(client.buffers(), 4u) << "one slab, carved whole";
    EXPECT_EQ(_backend->opened(0)->host_allocs, 1u);
    EXPECT_EQ(_backend->opened(0)->streams_created, 1u);
}

// THE reason this class exists: a reader has ONE pool however many devices it names. Its window did
// not grow, and pinned memory reaches every context.
TEST_F(DeviceWriterClientTest, One_Reader_Has_One_Pool_Across_Devices)
{
    DeviceWriterClient client(_writer, geometry(4 * Buffer), 4);

    StagingBuffer first;
    StagingBuffer second;
    ASSERT_EQ(client.take(common::Device::cuda(0), first), common::ResponseCode::Success);
    ASSERT_EQ(client.take(common::Device::cuda(1), second), common::ResponseCode::Success);
    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());

    EXPECT_EQ(client.devices(), 2u);
    EXPECT_EQ(client.buffers(), 4u) << "the second device did not bring its own buffers";
    EXPECT_EQ(_backend->opened(0)->host_allocs, 1u);
    EXPECT_EQ(_backend->opened(1)->host_allocs, 0u);
}

// Two readers of one device share its stream and its waiter, and have pools of their own - each
// sized by its own in-flight window.
TEST_F(DeviceWriterClientTest, Two_Readers_Share_The_Device_And_Not_The_Buffers)
{
    DeviceWriterClient small(_writer, geometry(4 * Buffer), 1);
    DeviceWriterClient large(_writer, geometry(4 * Buffer), 4);

    StagingBuffer buffer;
    ASSERT_EQ(small.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);
    ASSERT_EQ(large.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);

    EXPECT_EQ(small.buffers(), 1u) << "held to its own window, not the other reader's";
    EXPECT_EQ(large.buffers(), 4u);
    EXPECT_EQ(_writer->devices(), 1u);
    EXPECT_EQ(_backend->opened(0)->streams_created, 1u) << "one device, one stream";
    EXPECT_EQ(_backend->opened(0)->host_allocs, 2u) << "but one slab each";
    EXPECT_EQ(_factory_calls.load(), 1u) << "and one driver, for the whole streamer";
}

// The reader passes the device it got from the Batch, and nothing else.
TEST_F(DeviceWriterClientTest, Write_Copies_And_Returns_The_Buffer)
{
    DeviceWriterClient client(_writer, geometry(), 1);

    StagingBuffer buffer;
    ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());
    std::memset(buffer.data, 0x5a, Buffer);

    std::vector<char> destination(Buffer, 0);
    std::atomic<int> reported{-1};
    ASSERT_EQ(client.write(common::Device::cuda(0), buffer, Buffer, destination.data(),
                           [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);
    EXPECT_EQ(destination[0], 0x5a);
    EXPECT_EQ(destination[Buffer - 1], 0x5a);

    StagingBuffer next;
    ASSERT_TRUE(eventually([&]()
        {
            return client.take(common::Device::cuda(0), next) == common::ResponseCode::Success && next.valid();
        }));
}

// Buffers are anonymous, so a buffer taken for one device can be written to another. The pool is the
// same one either way.
TEST_F(DeviceWriterClientTest, A_Buffer_Can_Go_To_Any_Device_The_Reader_Has_Opened)
{
    DeviceWriterClient client(_writer, geometry(4 * Buffer), 4);

    StagingBuffer opened;
    ASSERT_EQ(client.take(common::Device::cuda(1), opened), common::ResponseCode::Success);

    StagingBuffer buffer;
    ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);
    std::memset(buffer.data, 0x33, Buffer);

    std::vector<char> destination(Buffer, 0);
    std::atomic<bool> done{false};
    ASSERT_EQ(client.write(common::Device::cuda(1), buffer, Buffer, destination.data(),
                           [&](common::ResponseCode) { done.store(true); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return done.load(); }));
    EXPECT_EQ(destination[0], 0x33);
    EXPECT_EQ(_backend->opened(1)->copies, 1u);
    EXPECT_EQ(_backend->opened(0)->copies, 0u);
}

// The ceiling is the reader's window, so it is reached only when the reader has as many buffers out
// as it is allowed reads in flight - a state its own window already prevents.
TEST_F(DeviceWriterClientTest, The_Ceiling_Is_The_Readers_Window)
{
    DeviceWriterClient client(_writer, geometry(), 1);

    StagingBuffer held;
    ASSERT_EQ(client.take(common::Device::cuda(0), held), common::ResponseCode::Success);
    ASSERT_TRUE(held.valid());

    StagingBuffer none;
    EXPECT_EQ(client.take(common::Device::cuda(0), none), common::ResponseCode::Success);
    EXPECT_FALSE(none.valid());
}

// Writing to a device this reader never read for is a routing bug. The buffer still comes back -
// there is only one pool, so it is reachable whatever ordinal was named.
TEST_F(DeviceWriterClientTest, Write_To_An_Unopened_Device_Is_Reported_And_The_Buffer_Returned)
{
    DeviceWriterClient client(_writer, geometry(), 1);

    StagingBuffer buffer;
    ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);

    std::vector<char> destination(Buffer, 0);
    EXPECT_EQ(client.write(common::Device::cuda(3), buffer, Buffer, destination.data(), nullptr),
              common::ResponseCode::InvalidParameterError);
    EXPECT_EQ(_writer->devices(), 1u) << "and the device was not opened on the way";

    StagingBuffer again;
    ASSERT_EQ(client.take(common::Device::cuda(0), again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid()) << "the buffer came back";
}

TEST_F(DeviceWriterClientTest, An_Unopenable_Device_Is_Reported)
{
    _backend->fail_open_device_at = 1;

    DeviceWriterClient client(_writer, geometry(), 4);

    StagingBuffer buffer;
    EXPECT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::InvalidDevice);
    EXPECT_FALSE(buffer.valid());
    EXPECT_EQ(client.buffers(), 0u) << "nothing was pinned for a device that could not be opened";
}

TEST_F(DeviceWriterClientTest, No_Device_On_This_Machine_Is_Reported)
{
    auto writer = std::make_shared<DeviceWriter>([](common::DeviceType) -> DeviceWriter::BackendFactory { return []() { return std::shared_ptr<device::Backend>(); }; });
    DeviceWriterClient client(writer, geometry(), 4);

    StagingBuffer buffer;
    EXPECT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::DeviceUnavailable);
    EXPECT_FALSE(buffer.valid());
    EXPECT_EQ(client.buffers(), 0u);
}

// The client is dropped before the writer - a reader stops before the streamer does - and its pool
// goes with it. Every buffer must already be back by then.
TEST_F(DeviceWriterClientTest, Dropping_The_Client_Returns_Everything_It_Pinned)
{
    {
        DeviceWriterClient client(_writer, geometry(4 * Buffer), 4);

        std::vector<char> destination(4 * Buffer, 0);
        for (unsigned i = 0; i < 4; ++i)
        {
            StagingBuffer buffer;
            ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);
            ASSERT_TRUE(buffer.valid());
            ASSERT_EQ(client.write(common::Device::cuda(0), buffer, Buffer, destination.data() + i * Buffer, nullptr),
                      common::ResponseCode::Success);
        }

        ASSERT_TRUE(eventually([&]() { return _backend->opened(0)->event_syncs.load() == 4u; }));
    }

    const auto device = _backend->opened(0);
    EXPECT_EQ(device->host_frees, 1u) << "the slab was freed";
    EXPECT_TRUE(device->live_host.empty()) << "nothing left pinned";
    EXPECT_EQ(device->streams_destroyed, 0u) << "the stream belongs to the writer, which is still alive";
}

// The real arrangement: several workers of one backend, each with its own client, all sharing one
// DeviceWriter. Nothing in a client is shared, so nothing in it is locked - the only shared state is
// the writer's map of devices, which is touched once per worker rather than once per buffer.
//
// Run this under --config=tsan. It is the test that says the split actually removed the sharing,
// rather than moving it somewhere quieter.
TEST_F(DeviceWriterClientTest, Every_Worker_Has_Its_Own_Client)
{
    constexpr unsigned Workers = 8;
    constexpr unsigned PerWorker = 300;
    constexpr unsigned Window = 4;

    std::vector<std::unique_ptr<DeviceWriterClient>> clients;
    for (unsigned w = 0; w < Workers; ++w)
    {
        clients.push_back(std::make_unique<DeviceWriterClient>(_writer, geometry(Window * Buffer), Window));
    }

    // One destination per worker. A shared one would have several copies landing on the same bytes,
    // which is a race in the test rather than in what it is testing.
    std::vector<std::vector<char>> destinations(Workers, std::vector<char>(Buffer, 0));
    std::atomic<unsigned> written{0};
    std::atomic<unsigned> issued{0};

    std::vector<std::thread> threads;
    for (unsigned w = 0; w < Workers; ++w)
    {
        threads.emplace_back([&, w]()
            {
                auto & client = *clients[w];
                for (unsigned i = 0; i < PerWorker; ++i)
                {
                    StagingBuffer buffer;
                    ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);
                    if (!buffer.valid())
                    {
                        continue;   // this worker's window is full - not an error
                    }
                    ++issued;
                    ASSERT_EQ(client.write(common::Device::cuda(0), buffer, Buffer, destinations[w].data(),
                                           [&](common::ResponseCode) { ++written; }),
                              common::ResponseCode::Success);
                }
            });
    }
    for (auto & thread : threads)
    {
        thread.join();
    }

    ASSERT_TRUE(eventually([&]() { return written.load() == issued.load(); }));

    EXPECT_EQ(_writer->devices(), 1u) << "one device, opened once";
    EXPECT_EQ(_backend->opened(0)->streams_created, 1u) << "and one stream for all of them";
    EXPECT_EQ(_factory_calls.load(), 1u) << "and one driver, for the whole streamer";

    // Each worker registered its own slab, and none of them went past its own window.
    EXPECT_EQ(_backend->opened(0)->host_allocs, Workers);
    for (const auto & client : clients)
    {
        EXPECT_EQ(client->buffers(), Window);
    }
}

// A slab is registered once and carved into several buffers, so pinning is batched.
TEST_F(DeviceWriterClientTest, One_Slab_Yields_Many_Buffers)
{
    DeviceWriterClient client(_writer, geometry(4 * Buffer), 16);

    StagingBuffer buffer;
    ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);

    EXPECT_EQ(client.buffers(), 4u) << "one slab, not the whole window";
    EXPECT_EQ(_backend->opened(0)->host_allocs, 1u);
    EXPECT_EQ(_backend->opened(0)->host_alloc_sizes.at(0), 4 * Buffer);
}

// THE reason a slab is smaller than the window: the first take() waits for its slab, so a slab the
// size of the window would pin everything before the first read is issued. The rest is registered on
// demand, while reads already submitted are in the kernel.
TEST_F(DeviceWriterClientTest, The_Pool_Grows_On_Demand_Rather_Than_All_At_Once)
{
    DeviceWriterClient client(_writer, geometry(4 * Buffer), 16);

    std::vector<StagingBuffer> held;
    for (unsigned i = 0; i < 16; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        held.push_back(buffer);

        // A slab every four buffers, and never more than the buffers asked for so far.
        EXPECT_EQ(_backend->opened(0)->host_allocs, i / 4 + 1) << "after " << (i + 1) << " buffers";
    }

    EXPECT_EQ(client.buffers(), 16u);
    EXPECT_EQ(_backend->opened(0)->host_allocs, 4u) << "four slabs, spread across sixteen takes";
}

// A slab larger than the window is cut down to it, so a shared geometry cannot make a small worker
// pin more than it asked for.
TEST_F(DeviceWriterClientTest, A_Slab_Never_Overshoots_The_Window)
{
    DeviceWriterClient client(_writer, geometry(64 * Buffer), 3);

    StagingBuffer buffer;
    ASSERT_EQ(client.take(common::Device::cuda(0), buffer), common::ResponseCode::Success);

    EXPECT_EQ(client.buffers(), 3u);
    EXPECT_EQ(_backend->opened(0)->host_alloc_sizes.at(0), 3 * Buffer);
}

// A worker holds workloads from several submissions at once, so its copies interleave across devices
// out of one pool. Each must land on the device its own write named, and every buffer must come back
// whichever device it went to.
TEST_F(DeviceWriterClientTest, Copies_Interleave_Across_Devices)
{
    constexpr unsigned Devices = 3;
    constexpr unsigned Rounds = 42;   // a whole number of rounds, so each device gets the same count

    DeviceWriterClient client(_writer, geometry(4 * Buffer), 4);

    // One destination per device, so a copy landing on the wrong one is visible in the bytes and not
    // only in a counter.
    std::vector<std::vector<char>> destinations(Devices, std::vector<char>(Buffer, 0));
    std::atomic<unsigned> done{0};
    unsigned issued = 0;

    for (unsigned round = 0; round < Rounds; ++round)
    {
        const unsigned ordinal = round % Devices;

        StagingBuffer buffer;
        ASSERT_EQ(client.take(common::Device::cuda(ordinal), buffer), common::ResponseCode::Success);
        if (!buffer.valid())
        {
            // The window is full; let the copies retire and try this round again.
            ASSERT_TRUE(eventually([&]() { return done.load() == issued; }));
            ASSERT_EQ(client.take(common::Device::cuda(ordinal), buffer), common::ResponseCode::Success);
            ASSERT_TRUE(buffer.valid());
        }

        std::memset(buffer.data, static_cast<int>('a' + ordinal), Buffer);
        ASSERT_EQ(client.write(common::Device::cuda(ordinal), buffer, Buffer, destinations[ordinal].data(),
                               [&](common::ResponseCode) { ++done; }),
                  common::ResponseCode::Success);
        ++issued;
    }

    ASSERT_TRUE(eventually([&]() { return done.load() == issued; }));

    for (unsigned ordinal = 0; ordinal < Devices; ++ordinal)
    {
        EXPECT_EQ(destinations[ordinal][0], 'a' + static_cast<int>(ordinal)) << "device " << ordinal;
        EXPECT_EQ(_backend->opened(ordinal)->copies, Rounds / Devices) << "device " << ordinal;
        EXPECT_EQ(_backend->opened(ordinal)->streams_created, 1u);
    }

    EXPECT_EQ(client.devices(), Devices);
    EXPECT_EQ(client.buffers(), 4u) << "one pool for all three devices";
    EXPECT_EQ(_backend->opened(0)->host_allocs, 1u) << "built through the first device named";

    // Every buffer is back, whichever device it went to.
    unsigned reacquired = 0;
    StagingBuffer buffer;
    while (client.take(common::Device::cuda(0), buffer) == common::ResponseCode::Success && buffer.valid())
    {
        ++reacquired;
    }
    EXPECT_EQ(reacquired, 4u);
}

} // namespace runai::llm::streamer::impl
