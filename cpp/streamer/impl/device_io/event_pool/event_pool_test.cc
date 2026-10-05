#include "streamer/impl/device_io/event_pool/event_pool.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "device/mock/mock_device.h"

namespace runai::llm::streamer::impl
{

namespace
{

class EventPoolTest : public ::testing::Test
{
 protected:
    std::shared_ptr<device::MockDevice> _mock = std::make_shared<device::MockDevice>();
};

} // namespace

TEST_F(EventPoolTest, Creates_Nothing_Until_Asked)
{
    EventPool pool(_mock, 4);

    EXPECT_EQ(pool.created(), 0u);
    EXPECT_EQ(_mock->events_created, 0u);
}

TEST_F(EventPoolTest, Grows_On_Demand_And_Stops_At_The_Ceiling)
{
    EventPool pool(_mock, 3);

    std::vector<device::EventHandle> held;
    for (unsigned i = 0; i < 3; ++i)
    {
        device::EventHandle event = nullptr;
        ASSERT_EQ(pool.acquire(event), common::ResponseCode::Success);
        ASSERT_NE(event, nullptr) << "event " << i;
        held.push_back(event);
    }
    EXPECT_EQ(pool.created(), 3u);

    // At the ceiling with everything in flight: not an error, and nothing new is created.
    device::EventHandle none = nullptr;
    EXPECT_EQ(pool.acquire(none), common::ResponseCode::Success);
    EXPECT_EQ(none, nullptr);
    EXPECT_EQ(_mock->events_created, 3u);

    // One back means one out again, and still no new event.
    pool.release(held.back());
    device::EventHandle again = nullptr;
    ASSERT_EQ(pool.acquire(again), common::ResponseCode::Success);
    EXPECT_EQ(again, held.back());
    EXPECT_EQ(_mock->events_created, 3u);
}

TEST_F(EventPoolTest, A_Failed_Create_Is_Reported)
{
    _mock->fail_event_create_from = 1;

    EventPool pool(_mock, 4);

    device::EventHandle event = nullptr;
    EXPECT_EQ(pool.acquire(event), common::ResponseCode::DeviceOutOfMemory);
    EXPECT_EQ(event, nullptr);
    EXPECT_EQ(pool.created(), 0u);
}

// Every event is destroyed, and through its owner - so a throw part-way could not leak one.
TEST_F(EventPoolTest, Teardown_Destroys_Every_Event)
{
    {
        EventPool pool(_mock, 4);

        // Held all at once, so four exist. Acquiring and releasing in turn would correctly reuse one.
        std::vector<device::EventHandle> held;
        for (unsigned i = 0; i < 4; ++i)
        {
            device::EventHandle event = nullptr;
            ASSERT_EQ(pool.acquire(event), common::ResponseCode::Success);
            held.push_back(event);
        }
        for (const auto event : held)
        {
            pool.release(event);
        }
        EXPECT_EQ(_mock->events_destroyed, 0u);
    }

    EXPECT_EQ(_mock->events_created, 4u);
    EXPECT_EQ(_mock->events_destroyed, 4u);
}

} // namespace runai::llm::streamer::impl
