#include "streamer/impl/device_io/device_writer/device_writer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "device/mock/mock_device.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr size_t Buffer = 4096;

StagingPool::Params params(unsigned max_buffers, size_t slab = Buffer)
{
    StagingPool::Params p;
    p.buffer_bytesize = Buffer;
    p.slab_bytesize = slab;
    p.max_buffers = max_buffers;
    return p;
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

class DeviceWriterTest : public ::testing::Test
{
 protected:
    std::shared_ptr<device::MockBackend> _backend = std::make_shared<device::MockBackend>();
};

} // namespace

// A load whose destinations are all host memory never opens anything, so it must cost nothing: no
// device, no pinned memory, no stream, no thread.
TEST_F(DeviceWriterTest, CostsNothingUntilTheFirstOpen)
{
    DeviceWriter writer(_backend, params(8));

    EXPECT_EQ(_backend->opens, 0u);
    EXPECT_EQ(writer.devices(), 0u);
    EXPECT_EQ(writer.buffers(), 0u);
}

TEST_F(DeviceWriterTest, OpenBuildsTheStreamAndThePool)
{
    DeviceWriter writer(_backend, params(4, 4 * Buffer));

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);
    ASSERT_NE(channel, nullptr);

    EXPECT_EQ(writer.devices(), 1u);
    EXPECT_EQ(writer.buffers(), 0u) << "the pool exists but has not grown yet";

    StagingBuffer buffer;
    ASSERT_EQ(writer.take(channel, buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());
    EXPECT_EQ(writer.buffers(), 4u) << "one slab, carved whole, on the first take";
    EXPECT_EQ(_backend->opened(0)->host_allocs, 1u);
}

// A device is opened once. Asking again gives the same channel, not a second stream and waiter.
TEST_F(DeviceWriterTest, OpeningTwiceGivesTheSameChannel)
{
    DeviceWriter writer(_backend, params(4));

    DeviceWriter::Channel first = nullptr;
    DeviceWriter::Channel again = nullptr;
    ASSERT_EQ(writer.open(0, first), common::ResponseCode::Success);
    ASSERT_EQ(writer.open(0, again), common::ResponseCode::Success);

    EXPECT_EQ(first, again);
    EXPECT_EQ(writer.devices(), 1u);
}

// The reader hands over a buffer and a place; everything else is the writer's business.
TEST_F(DeviceWriterTest, WriteCopiesAndReturnsTheBuffer)
{
    DeviceWriter writer(_backend, params(1));

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);

    StagingBuffer buffer;
    ASSERT_EQ(writer.take(channel, buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    std::memset(buffer.data, 0x5a, Buffer);
    std::vector<char> destination(Buffer, 0);

    std::atomic<int> reported{-1};
    ASSERT_EQ(writer.write(channel, buffer, Buffer, destination.data(),
                           [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);
    EXPECT_EQ(destination[0], 0x5a) << "the bytes arrived";
    EXPECT_EQ(destination[Buffer - 1], 0x5a);

    // Only one buffer exists, so getting one again proves it was returned.
    StagingBuffer next;
    ASSERT_EQ(writer.take(channel, next), common::ResponseCode::Success);
    EXPECT_TRUE(next.valid());
}

// One read has one contiguous destination, so one buffer is one copy. Nothing is merged, nothing
// is split, and nothing depends on what the bytes contain.
TEST_F(DeviceWriterTest, OneBufferIsOneCopy)
{
    DeviceWriter writer(_backend, params(4, 4 * Buffer));

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);

    std::vector<char> destination(4 * Buffer, 0);
    std::atomic<unsigned> done{0};

    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(writer.take(channel, buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        std::memset(buffer.data, static_cast<int>('a' + i), Buffer);
        ASSERT_EQ(writer.write(channel, buffer, Buffer, destination.data() + i * Buffer,
                               [&](common::ResponseCode) { ++done; }),
                  common::ResponseCode::Success);
    }

    ASSERT_TRUE(eventually([&]() { return done.load() == 4u; }));
    EXPECT_EQ(_backend->opened(0)->copies, 4u) << "one per buffer, not one per range inside it";
    for (unsigned i = 0; i < 4; ++i)
    {
        EXPECT_EQ(destination[i * Buffer], 'a' + static_cast<int>(i));
    }
}

// A partly filled buffer - a short read - copies only what was read.
TEST_F(DeviceWriterTest, OnlyTheBytesReadAreCopied)
{
    DeviceWriter writer(_backend, params(1));

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);

    StagingBuffer buffer;
    ASSERT_EQ(writer.take(channel, buffer), common::ResponseCode::Success);
    std::memset(buffer.data, 0x7f, Buffer);

    std::vector<char> destination(Buffer, 0);
    std::atomic<bool> done{false};
    ASSERT_EQ(writer.write(channel, buffer, 16, destination.data(),
                           [&](common::ResponseCode) { done.store(true); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return done.load(); }));
    EXPECT_EQ(destination[15], 0x7f);
    EXPECT_EQ(destination[16], 0) << "nothing past the bytes that were read";
}

// Buffers are shared across devices: pinned memory reaches every context, so a second device gets a
// stream and a waiter of its own but no second pool.
TEST_F(DeviceWriterTest, ASecondDeviceSharesTheBuffers)
{
    DeviceWriter writer(_backend, params(4, 4 * Buffer));

    DeviceWriter::Channel first = nullptr;
    DeviceWriter::Channel second = nullptr;
    ASSERT_EQ(writer.open(0, first), common::ResponseCode::Success);
    ASSERT_EQ(writer.open(1, second), common::ResponseCode::Success);
    ASSERT_NE(first, second);

    // Either channel yields the same buffers - there is one pool.
    StagingBuffer buffer;
    ASSERT_EQ(writer.take(first, buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    std::vector<char> destination(Buffer, 0);
    std::atomic<bool> done{false};
    ASSERT_EQ(writer.write(second, buffer, Buffer, destination.data(),
                           [&](common::ResponseCode) { done.store(true); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return done.load(); }));

    EXPECT_EQ(writer.devices(), 2u);
    EXPECT_EQ(writer.buffers(), 4u) << "one pool, not one per device";
    EXPECT_EQ(_backend->opened(1)->host_allocs, 0u) << "device 1 allocated nothing";
    EXPECT_EQ(_backend->opened(1)->copies, 1u) << "but it did the copy";
}

// Everything in flight is not an error: the reader waits for I/O instead of submitting.
TEST_F(DeviceWriterTest, TakeYieldsNothingWhenEveryBufferIsOut)
{
    DeviceWriter writer(_backend, params(1));

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);

    StagingBuffer held;
    ASSERT_EQ(writer.take(channel, held), common::ResponseCode::Success);
    ASSERT_TRUE(held.valid());

    StagingBuffer none;
    EXPECT_EQ(writer.take(channel, none), common::ResponseCode::Success);
    EXPECT_FALSE(none.valid());
}

// A null channel is a caller bug, and the only remaining way to ask for a buffer without a device.
// "Open before you take" is otherwise not expressible: take() needs a channel, and only open()
// hands one out.
TEST_F(DeviceWriterTest, TakeWithoutAChannelIsReported)
{
    DeviceWriter writer(_backend, params(4));

    StagingBuffer buffer;
    EXPECT_EQ(writer.take(nullptr, buffer), common::ResponseCode::InvalidParameterError);
    EXPECT_FALSE(buffer.valid());
}

TEST_F(DeviceWriterTest, AnUnopenableDeviceIsReported)
{
    _backend->fail_open_device_at = 1;

    DeviceWriter writer(_backend, params(4));

    DeviceWriter::Channel channel = nullptr;
    EXPECT_EQ(writer.open(0, channel), common::ResponseCode::InvalidDevice);
    EXPECT_EQ(channel, nullptr);
    EXPECT_EQ(writer.buffers(), 0u) << "nothing was pinned for a device that could not be opened";
}

// No backend means no device at all. Not a crash, and not silence.
TEST_F(DeviceWriterTest, NoBackendIsReported)
{
    DeviceWriter writer(nullptr, params(4));

    DeviceWriter::Channel channel = nullptr;
    EXPECT_EQ(writer.open(0, channel), common::ResponseCode::DeviceUnavailable);
    EXPECT_EQ(channel, nullptr);
}

// The pool's teardown assumes every buffer is back, so the waiters must stop before it goes.
TEST_F(DeviceWriterTest, TeardownDrainsBeforeThePoolIsDestroyed)
{
    auto mock = std::make_shared<device::MockBackend>();
    {
        DeviceWriter writer(mock, params(4, 4 * Buffer));

        DeviceWriter::Channel channel = nullptr;
        ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);

        std::vector<char> destination(4 * Buffer, 0);
        for (unsigned i = 0; i < 4; ++i)
        {
            StagingBuffer buffer;
            ASSERT_EQ(writer.take(channel, buffer), common::ResponseCode::Success);
            ASSERT_TRUE(buffer.valid());
            ASSERT_EQ(writer.write(channel, buffer, Buffer, destination.data() + i * Buffer, nullptr),
                      common::ResponseCode::Success);
        }
    }

    const auto device = mock->opened(0);
    ASSERT_NE(device, nullptr);
    EXPECT_EQ(device->events_destroyed, 4u);
    EXPECT_EQ(device->host_frees, 1u) << "the slab was freed";
    EXPECT_TRUE(device->live_host.empty()) << "nothing left pinned";
}

// A stream is a raw handle with no owner of its own, so nothing but Target's destructor will ever
// free it.
//
// InstantTensor gets this one right - destroy_threads() joins its executors and then destroys both
// streams - and misses the events instead: cudaEventDestroy is not even in its binding table, so
// every open leaks io_depth of them. Different resource, same shape, and the same reason it hides:
// a one-shot loader exits before it matters, and so does a benchmark.
TEST_F(DeviceWriterTest, EveryStreamIsDestroyed)
{
    auto mock = std::make_shared<device::MockBackend>();
    {
        DeviceWriter writer(mock, params(4));

        DeviceWriter::Channel first = nullptr;
        DeviceWriter::Channel second = nullptr;
        ASSERT_EQ(writer.open(0, first), common::ResponseCode::Success);
        ASSERT_EQ(writer.open(1, second), common::ResponseCode::Success);

        EXPECT_EQ(mock->opened(0)->streams_created, 1u);
        EXPECT_EQ(mock->opened(1)->streams_created, 1u);
        EXPECT_EQ(mock->opened(0)->streams_destroyed, 0u);
    }

    EXPECT_EQ(mock->opened(0)->streams_destroyed, 1u);
    EXPECT_EQ(mock->opened(1)->streams_destroyed, 1u);
}

// Targets are moved into a map. A move that left the source holding the handle would have the
// moved-from destructor free a stream that is still in use.
TEST_F(DeviceWriterTest, MovingATargetDoesNotDestroyItsStreamTwice)
{
    auto mock = std::make_shared<device::MockBackend>();
    {
        DeviceWriter writer(mock, params(4));
        for (unsigned ordinal = 0; ordinal < 4; ++ordinal)
        {
            DeviceWriter::Channel channel = nullptr;
            ASSERT_EQ(writer.open(ordinal, channel), common::ResponseCode::Success);
        }
        for (unsigned ordinal = 0; ordinal < 4; ++ordinal)
        {
            EXPECT_EQ(mock->opened(ordinal)->streams_destroyed, 0u) << "ordinal " << ordinal;
        }
    }

    for (unsigned ordinal = 0; ordinal < 4; ++ordinal)
    {
        EXPECT_EQ(mock->opened(ordinal)->streams_created, 1u) << "ordinal " << ordinal;
        EXPECT_EQ(mock->opened(ordinal)->streams_destroyed, 1u) << "ordinal " << ordinal;
    }
}

// Copying more than the buffer holds would read past the end of pinned memory. The buffer still
// comes back: a caller cannot know to return one itself, and one lost here is a deadlock later.
TEST_F(DeviceWriterTest, CopyingMoreThanTheBufferHoldsIsRefusedAndTheBufferReturned)
{
    DeviceWriter writer(_backend, params(1));

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);

    StagingBuffer buffer;
    ASSERT_EQ(writer.take(channel, buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    std::vector<char> destination(2 * Buffer, 0);
    std::atomic<bool> told{false};
    EXPECT_EQ(writer.write(channel, buffer, Buffer + 1, destination.data(),
                           [&](common::ResponseCode) { told.store(true); }),
              common::ResponseCode::InvalidParameterError);

    EXPECT_EQ(_backend->opened(0)->copies, 0u) << "refused before anything was enqueued";
    EXPECT_FALSE(told.load()) << "an error is reported by the return value, not also by the callback";

    // The pool holds one buffer, so getting one again proves it came back.
    StagingBuffer again;
    ASSERT_EQ(writer.take(channel, again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid());
    EXPECT_EQ(again.data, buffer.data);
}

// A null channel is the one path that cannot return the buffer - there is no pool to reach - so the
// caller keeps it. Saying so is the point: silence here would look like a leak.
TEST_F(DeviceWriterTest, WriteWithoutAChannelLeavesTheBufferWithTheCaller)
{
    DeviceWriter writer(_backend, params(1));

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(0, channel), common::ResponseCode::Success);

    StagingBuffer buffer;
    ASSERT_EQ(writer.take(channel, buffer), common::ResponseCode::Success);

    std::vector<char> destination(Buffer, 0);
    EXPECT_EQ(writer.write(nullptr, buffer, Buffer, destination.data(), nullptr),
              common::ResponseCode::InvalidParameterError);

    // Still out: the writer did not take it back.
    StagingBuffer none;
    ASSERT_EQ(writer.take(channel, none), common::ResponseCode::Success);
    EXPECT_FALSE(none.valid());
}

} // namespace runai::llm::streamer::impl
