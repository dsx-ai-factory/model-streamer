#include "streamer/impl/device_io/device_writer/device_writer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <map>

#include "device/mock/mock_device.h"
#include "streamer/impl/device_io/event_pool/event_pool.h"

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

class DeviceWriterTest : public ::testing::Test
{
 protected:
    // The writer asks for a backend rather than holding one, so the count below is what a test
    // watches to see whether the driver was reached at all.
    // One backend, whatever the type is asked for - every test here names a single device type.
    DeviceWriter::BackendLookup lookup()
    {
        return [this](common::DeviceType) -> DeviceWriter::BackendFactory
            {
                return [this]() { ++_factory_calls; return _backend; };
            };
    }

    // The writer owns neither buffers nor events, so a test brings both - as a client does.
    std::shared_ptr<StagingPool> pool_for(DeviceWriter & writer, DeviceWriter::Channel channel, unsigned max_buffers)
    {
        StagingPool::Params params;
        params.buffer_bytesize = Buffer;
        params.slab_bytesize = max_buffers * Buffer;
        params.max_buffers = max_buffers;
        return std::make_shared<StagingPool>(writer.device(channel), params);
    }

    std::shared_ptr<EventPool> events_for(DeviceWriter & writer, DeviceWriter::Channel channel, unsigned max_events)
    {
        return std::make_shared<EventPool>(writer.device(channel), max_events);
    }

    // One copy's worth: a buffer from `pool` and an event from the channel's own pool.
    DeviceWriter::Copy copy_of(DeviceWriter & writer, DeviceWriter::Channel channel,
                               const std::shared_ptr<StagingPool> & pool, const StagingBuffer & buffer)
    {
        auto & events = _events[channel];
        if (events == nullptr)
        {
            events = events_for(writer, channel, 64);
        }

        DeviceWriter::Copy copy;
        copy.pool = pool;
        copy.buffer = buffer;
        copy.events = events;
        EXPECT_EQ(events->acquire(copy.event), common::ResponseCode::Success);
        return copy;
    }

    std::map<DeviceWriter::Channel, std::shared_ptr<EventPool>> _events;

    std::shared_ptr<device::MockBackend> _backend = std::make_shared<device::MockBackend>();
    std::atomic<unsigned> _factory_calls{0};
};

} // namespace

// A load whose destinations are all host memory never opens anything, so it must cost nothing: no
// device, no stream, no thread, no driver.
TEST_F(DeviceWriterTest, Costs_Nothing_Until_The_First_Open)
{
    DeviceWriter writer(lookup());

    EXPECT_EQ(_factory_calls.load(), 0u) << "the driver was reached before any device was asked for";
    EXPECT_EQ(_backend->opens, 0u);
    EXPECT_EQ(writer.devices(), 0u);
}

// One device, one stream, one waiter - however many readers name it.
TEST_F(DeviceWriterTest, Opening_Twice_Gives_The_Same_Channel)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel first = nullptr;
    DeviceWriter::Channel again = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), first), common::ResponseCode::Success);
    ASSERT_EQ(writer.open(common::Device::cuda(0), again), common::ResponseCode::Success);

    EXPECT_EQ(first, again);
    EXPECT_EQ(writer.devices(), 1u);
    EXPECT_EQ(_backend->opened(0)->streams_created, 1u);
}

// A client allocates pinned memory, and it needs a context to do that. This is the only reason a
// device leaves the writer.
TEST_F(DeviceWriterTest, The_Device_Behind_A_Channel_Is_Reachable)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(2), channel), common::ResponseCode::Success);

    EXPECT_EQ(writer.device(channel), _backend->opened(2));
    EXPECT_EQ(writer.device(nullptr), nullptr);
}

TEST_F(DeviceWriterTest, Write_Copies_And_Returns_The_Buffer)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    const auto pool = pool_for(writer, channel, 1);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    std::memset(buffer.data, 0x5a, Buffer);
    std::vector<char> destination(Buffer, 0);

    std::atomic<int> reported{-1};
    ASSERT_EQ(writer.write(channel, copy_of(writer, channel, pool, buffer), Buffer, destination.data(),
                           [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);
    EXPECT_EQ(destination[0], 0x5a) << "the bytes arrived";
    EXPECT_EQ(destination[Buffer - 1], 0x5a);

    // Only one buffer exists, so getting one again proves it was returned.
    StagingBuffer next;
    ASSERT_TRUE(eventually([&]()
        {
            return pool->try_acquire(next) == common::ResponseCode::Success && next.valid();
        }));
}

// One read has one contiguous destination, so one buffer is one copy. Nothing is merged, nothing
// is split, and nothing depends on what the bytes contain.
TEST_F(DeviceWriterTest, One_Buffer_Is_One_Copy)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    const auto pool = pool_for(writer, channel, 4);

    std::vector<char> destination(4 * Buffer, 0);
    std::atomic<unsigned> done{0};

    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        std::memset(buffer.data, static_cast<int>('a' + i), Buffer);
        ASSERT_EQ(writer.write(channel, copy_of(writer, channel, pool, buffer), Buffer, destination.data() + i * Buffer,
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
TEST_F(DeviceWriterTest, Only_The_Bytes_Read_Are_Copied)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    const auto pool = pool_for(writer, channel, 1);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
    std::memset(buffer.data, 0x7f, Buffer);

    std::vector<char> destination(Buffer, 0);
    std::atomic<bool> done{false};
    ASSERT_EQ(writer.write(channel, copy_of(writer, channel, pool, buffer), 16, destination.data(),
                           [&](common::ResponseCode) { done.store(true); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return done.load(); }));
    EXPECT_EQ(destination[15], 0x7f);
    EXPECT_EQ(destination[16], 0) << "nothing past the bytes that were read";
}

// A second device is a second stream and a second waiter, so a slow copy on one cannot delay the
// other. One pool still serves both: pinned memory reaches every context.
TEST_F(DeviceWriterTest, A_Second_Device_Gets_Its_Own_Stream)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel first = nullptr;
    DeviceWriter::Channel second = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), first), common::ResponseCode::Success);
    ASSERT_EQ(writer.open(common::Device::cuda(1), second), common::ResponseCode::Success);
    ASSERT_NE(first, second);

    const auto pool = pool_for(writer, first, 4);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    std::vector<char> destination(Buffer, 0);
    std::atomic<bool> done{false};
    ASSERT_EQ(writer.write(second, copy_of(writer, second, pool, buffer), Buffer, destination.data(),
                           [&](common::ResponseCode) { done.store(true); }),
              common::ResponseCode::Success);

    ASSERT_TRUE(eventually([&]() { return done.load(); }));

    EXPECT_EQ(writer.devices(), 2u);
    EXPECT_EQ(_backend->opened(0)->streams_created, 1u);
    EXPECT_EQ(_backend->opened(1)->streams_created, 1u);
    EXPECT_EQ(_backend->opened(1)->copies, 1u);
    EXPECT_EQ(_backend->opened(0)->host_allocs, 1u) << "one pool, built through the first device";
    EXPECT_EQ(_backend->opened(1)->host_allocs, 0u);
}

TEST_F(DeviceWriterTest, An_Unopenable_Device_Is_Reported)
{
    _backend->fail_open_device_at = 1;

    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    EXPECT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::InvalidDevice);
    EXPECT_EQ(channel, nullptr);
}

// No backend means no device at all. Not a crash, and not silence. Two ways to have none: no
// factory, and a factory that finds no driver on this machine.
TEST_F(DeviceWriterTest, No_Backend_Is_Reported)
{
    DeviceWriter writer(nullptr);

    DeviceWriter::Channel channel = nullptr;
    EXPECT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::DeviceUnavailable);
    EXPECT_EQ(channel, nullptr);
}

TEST_F(DeviceWriterTest, A_Backend_That_Cannot_Be_Loaded_Is_Reported)
{
    DeviceWriter writer([](common::DeviceType) -> DeviceWriter::BackendFactory
        { return []() { return std::shared_ptr<device::Backend>(); }; });

    DeviceWriter::Channel channel = nullptr;
    EXPECT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::DeviceUnavailable);
    EXPECT_EQ(channel, nullptr);
}

// Every device after the first reuses the backend already loaded.
TEST_F(DeviceWriterTest, The_Backend_Is_Asked_For_Once)
{
    DeviceWriter writer(lookup());

    for (unsigned ordinal : { 0u, 1u, 0u, 2u })
    {
        DeviceWriter::Channel channel = nullptr;
        ASSERT_EQ(writer.open(common::Device::cuda(ordinal), channel), common::ResponseCode::Success);
    }

    EXPECT_EQ(_factory_calls.load(), 1u);
    EXPECT_EQ(writer.devices(), 3u);
}

// A pool's teardown assumes every buffer is back, so the waiters must stop before it goes. The
// writer's destructor is what stops them; the pool then goes when its reader drops it.
TEST_F(DeviceWriterTest, Teardown_Drains_Before_The_Pool_Is_Destroyed)
{
    auto mock = std::make_shared<device::MockBackend>();
    {
        std::shared_ptr<StagingPool> pool;
        {
            DeviceWriter writer([mock](common::DeviceType) -> DeviceWriter::BackendFactory
            { return [mock]() { return mock; }; });

            DeviceWriter::Channel channel = nullptr;
            ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
            pool = pool_for(writer, channel, 4);

            std::vector<char> destination(4 * Buffer, 0);
            for (unsigned i = 0; i < 4; ++i)
            {
                StagingBuffer buffer;
                ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
                ASSERT_TRUE(buffer.valid());
                ASSERT_EQ(writer.write(channel, copy_of(writer, channel, pool, buffer), Buffer, destination.data() + i * Buffer, nullptr),
                          common::ResponseCode::Success);
            }
        }   // ~DeviceWriter stops the waiter, so every buffer is back before the pool can go

        EXPECT_EQ(mock->opened(0)->host_frees, 0u) << "the reader still holds its pool";
    }

    const auto device = mock->opened(0);
    ASSERT_NE(device, nullptr);
    EXPECT_EQ(device->host_frees, 1u) << "the slab was freed";
    EXPECT_TRUE(device->live_host.empty()) << "nothing left pinned";

    // The events belong to this test's pool, not to the writer or the buffers - see EventPool. Their
    // destruction is that pool's business, and its own test.
    EXPECT_EQ(device->foreign_records, 0u) << "every event was recorded on its own device's stream";
}

// The waiter returns the buffer to the pool, so the pool must outlive the copy - whatever the reader
// does with its own share meanwhile. Checked inside the completion, which runs before the release.
TEST_F(DeviceWriterTest, The_Pool_Outlives_A_Reader_That_Drops_It)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    auto pool = pool_for(writer, channel, 1);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    const std::weak_ptr<StagingPool> watch = pool;

    std::vector<char> destination(Buffer, 0);
    std::atomic<int> alive{-1};
    ASSERT_EQ(writer.write(channel, copy_of(writer, channel, pool, buffer), Buffer, destination.data(),
                           [&](common::ResponseCode) { alive.store(watch.lock() != nullptr ? 1 : 0); }),
              common::ResponseCode::Success);

    pool.reset();   // the reader is done with it while the copy may still be in flight

    ASSERT_TRUE(eventually([&]() { return alive.load() >= 0; }));
    EXPECT_EQ(alive.load(), 1) << "the copy held the last share of its pool";
}

// A stream is a raw handle with no owner of its own, so nothing but Target's destructor will ever
// free it.
//
// InstantTensor gets this one right - destroy_threads() joins its executors and then destroys both
// streams - and misses the events instead: cudaEventDestroy is not even in its binding table, so
// every open leaks io_depth of them. Different resource, same shape, and the same reason it hides:
// a one-shot loader exits before it matters, and so does a benchmark.
TEST_F(DeviceWriterTest, Every_Stream_Is_Destroyed)
{
    auto mock = std::make_shared<device::MockBackend>();
    {
        DeviceWriter writer([mock](common::DeviceType) -> DeviceWriter::BackendFactory
            { return [mock]() { return mock; }; });

        DeviceWriter::Channel first = nullptr;
        DeviceWriter::Channel second = nullptr;
        ASSERT_EQ(writer.open(common::Device::cuda(0), first), common::ResponseCode::Success);
        ASSERT_EQ(writer.open(common::Device::cuda(1), second), common::ResponseCode::Success);

        EXPECT_EQ(mock->opened(0)->streams_created, 1u);
        EXPECT_EQ(mock->opened(1)->streams_created, 1u);
        EXPECT_EQ(mock->opened(0)->streams_destroyed, 0u);
    }

    EXPECT_EQ(mock->opened(0)->streams_destroyed, 1u);
    EXPECT_EQ(mock->opened(1)->streams_destroyed, 1u);
}

// Targets are moved into a map. A move that left the source holding the handle would have the
// moved-from destructor free a stream that is still in use.
TEST_F(DeviceWriterTest, Moving_A_Target_Does_Not_Destroy_Its_Stream_Twice)
{
    auto mock = std::make_shared<device::MockBackend>();
    {
        DeviceWriter writer([mock](common::DeviceType) -> DeviceWriter::BackendFactory
            { return [mock]() { return mock; }; });
        for (unsigned ordinal = 0; ordinal < 4; ++ordinal)
        {
            DeviceWriter::Channel channel = nullptr;
            ASSERT_EQ(writer.open(common::Device::cuda(ordinal), channel), common::ResponseCode::Success);
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
TEST_F(DeviceWriterTest, Copying_More_Than_The_Buffer_Holds_Is_Refused_And_The_Buffer_Returned)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    const auto pool = pool_for(writer, channel, 1);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    std::vector<char> destination(2 * Buffer, 0);
    std::atomic<bool> told{false};
    EXPECT_EQ(writer.write(channel, copy_of(writer, channel, pool, buffer), Buffer + 1, destination.data(),
                           [&](common::ResponseCode) { told.store(true); }),
              common::ResponseCode::InvalidParameterError);

    EXPECT_EQ(_backend->opened(0)->copies, 0u) << "refused before anything was enqueued";
    EXPECT_FALSE(told.load()) << "an error is reported by the return value, not also by the callback";

    // The pool holds one buffer, so getting one again proves it came back.
    StagingBuffer again;
    ASSERT_EQ(pool->try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid());
    EXPECT_EQ(again.data, buffer.data);
}

// No pool means the copy cannot be accepted AT ALL - not even successfully. The waiter gives the
// buffer back from its own thread, so enqueuing would move the fault to another thread and turn a
// bad call into a crash.
TEST_F(DeviceWriterTest, Write_Without_A_Pool_Or_Event_Is_Refused_Before_Anything_Is_Enqueued)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    const auto pool = pool_for(writer, channel, 1);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    // A copy with nowhere to put the buffer back.
    DeviceWriter::Copy orphan;
    orphan.buffer = buffer;

    std::vector<char> destination(Buffer, 0);
    EXPECT_EQ(writer.write(channel, orphan, Buffer, destination.data(), nullptr),
              common::ResponseCode::InvalidParameterError);
    EXPECT_EQ(_backend->opened(0)->copies, 0u) << "nothing was enqueued, so no thread can touch it";

    // The caller keeps the buffer - the one path where it does.
    StagingBuffer none;
    ASSERT_EQ(pool->try_acquire(none), common::ResponseCode::Success);
    EXPECT_FALSE(none.valid());
}

// A null channel is a caller bug - open() clears it on failure, so a caller that ignored the error
// reaches here. The buffer is returned, because the pool is right there in the call.
TEST_F(DeviceWriterTest, Write_Without_A_Channel_Still_Returns_The_Buffer)
{
    DeviceWriter writer(lookup());

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    const auto pool = pool_for(writer, channel, 1);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    std::vector<char> destination(Buffer, 0);
    EXPECT_EQ(writer.write(nullptr, copy_of(writer, channel, pool, buffer), Buffer, destination.data(), nullptr),
              common::ResponseCode::InvalidParameterError);

    StagingBuffer again;
    ASSERT_EQ(pool->try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid()) << "the buffer came back";
}


// The reason the channel map is keyed by DEVICE and not by ordinal. Two types both have an ordinal 0,
// and they are different devices served by different drivers - keying on the ordinal alone would hand
// the second one the first one's stream.
//
// A type the lookup was taught but the C header has not published yet stands in for a second vendor:
// nothing else can tell a keyed map from an unkeyed one while CUDA is the only backend.
TEST_F(DeviceWriterTest, Two_Types_With_The_Same_Ordinal_Are_Two_Devices)
{
    const auto second_type = static_cast<common::DeviceType>(99);

    auto cuda = std::make_shared<device::MockBackend>();
    auto other = std::make_shared<device::MockBackend>();

    DeviceWriter writer([cuda, other, second_type](common::DeviceType type) -> DeviceWriter::BackendFactory
        {
            if (type == common::DeviceType::Cuda) { return [cuda]() { return cuda; }; }
            if (type == second_type)              { return [other]() { return other; }; }
            return DeviceWriter::BackendFactory();
        });

    DeviceWriter::Channel first = nullptr;
    DeviceWriter::Channel second = nullptr;

    ASSERT_EQ(writer.open(common::Device::cuda(0), first), common::ResponseCode::Success);
    ASSERT_EQ(writer.open(common::Device{ second_type, 0 }, second), common::ResponseCode::Success);

    EXPECT_NE(first, second) << "one channel served two different devices";

    // Each driver was reached exactly once, for its own type.
    EXPECT_EQ(cuda->opens, 1u);
    EXPECT_EQ(other->opens, 1u);
}

// A type this build has no backend for is reported, not guessed at. Routed by TYPE, so a served type
// still works in the same writer.
TEST_F(DeviceWriterTest, A_Type_With_No_Backend_Is_Unavailable)
{
    DeviceWriter writer([this](common::DeviceType type) -> DeviceWriter::BackendFactory
        {
            if (type != common::DeviceType::Cuda) { return DeviceWriter::BackendFactory(); }
            return [this]() { ++_factory_calls; return _backend; };
        });

    DeviceWriter::Channel channel = nullptr;
    EXPECT_EQ(writer.open(common::Device{ static_cast<common::DeviceType>(99), 0 }, channel),
              common::ResponseCode::DeviceUnavailable);
    EXPECT_EQ(channel, nullptr);

    EXPECT_EQ(writer.open(common::Device::cuda(0), channel), common::ResponseCode::Success)
        << "a type with no backend must not spoil the one that has";
    EXPECT_NE(channel, nullptr);
}

} // namespace runai::llm::streamer::impl
