#include "device/mock/mock_device.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>
#include <mutex>

namespace runai::llm::streamer::device
{

namespace
{

// Tokens, not addresses. Casting a counter keeps every handle distinct and obviously not a pointer
// to anything, so a test that dereferences one fails loudly instead of corrupting memory.
void * token(unsigned value)
{
    return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

} // namespace

common::ResponseCode MockDevice::bind_thread()
{
    ++bind_calls;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::get_attribute(Attribute, int & value) const
{
    value = 0;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::memory_info(size_t & free_bytes, size_t & total_bytes) const
{
    free_bytes = total_bytes = 0;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::host_alloc(size_t bytesize, void ** ptr)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    ++host_allocs;
    host_alloc_sizes.push_back(bytesize);

    if (fail_host_alloc_at != 0 && host_allocs == fail_host_alloc_at)
    {
        *ptr = nullptr;
        return common::ResponseCode::DeviceOutOfMemory;
    }

    *ptr = std::malloc(bytesize);
    if (*ptr == nullptr)
    {
        return common::ResponseCode::DeviceOutOfMemory;
    }

    live_host.push_back(*ptr);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::host_free(void * ptr)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    ++host_frees;
    const auto it = std::find(live_host.begin(), live_host.end(), ptr);
    if (it == live_host.end())
    {
        // Freeing something never allocated, or freeing twice.
        return common::ResponseCode::InvalidParameterError;
    }
    live_host.erase(it);
    std::free(ptr);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::device_alloc(size_t bytesize, void ** ptr)
{
    *ptr = std::malloc(bytesize);
    return *ptr != nullptr ? common::ResponseCode::Success : common::ResponseCode::DeviceOutOfMemory;
}

common::ResponseCode MockDevice::device_free(void * ptr)
{
    std::free(ptr);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::stream_create(StreamHandle & stream)
{
    ++streams_created;

    const std::lock_guard<std::mutex> guard(_mutex);
    stream = token(_next_token++);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::stream_destroy(StreamHandle)
{
    ++streams_destroyed;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::stream_synchronize(StreamHandle)
{
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::stream_query(StreamHandle, Status & status)
{
    status = Status::Ready;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::stream_wait_event(StreamHandle, EventHandle)
{
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::event_create(EventHandle & event)
{
    const std::lock_guard<std::mutex> guard(_mutex);

    if (fail_event_create_from != 0 && events_created + 1 >= fail_event_create_from)
    {
        return common::ResponseCode::DeviceOutOfMemory;
    }

    ++events_created;
    event = token(_next_token++);
    // A never-recorded event is an empty set of work, so it queries ready - which is the correct
    // initial state for a buffer nobody is copying out of. The real driver behaves the same way.
    _ready[event] = true;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::event_destroy(EventHandle event)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    ++events_destroyed;
    _ready.erase(event);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::event_record(EventHandle event, StreamHandle)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    _ready[event] = false;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::event_query(EventHandle event, Status & status)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    const auto it = _ready.find(event);
    status = (it != _ready.end() && it->second) ? Status::Ready : Status::NotReady;
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::event_synchronize(EventHandle event)
{
    ++event_syncs;

    const std::lock_guard<std::mutex> guard(_mutex);
    _ready[event] = true;

    return fail_event_synchronize.load()
        ? common::ResponseCode::DeviceTransferError
        : common::ResponseCode::Success;
}

common::ResponseCode MockDevice::memcpy_h2d_async(void * dst, const void * src, size_t bytesize, StreamHandle)
{
    ++copies;
    std::memcpy(dst, src, bytesize);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::memcpy_d2d_async(void * dst, const void * src, size_t bytesize, StreamHandle)
{
    ++copies;
    std::memcpy(dst, src, bytesize);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::memset_async(void * dst, unsigned char value, size_t bytesize, StreamHandle)
{
    std::memset(dst, value, bytesize);
    return common::ResponseCode::Success;
}

void MockDevice::set_ready(EventHandle event, bool ready)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    _ready[event] = ready;
}

void MockDevice::set_all_ready(bool ready)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    for (auto & entry : _ready)
    {
        entry.second = ready;
    }
}

MockBackend::MockBackend(unsigned device_count) :
    _count(device_count)
{
}

Capabilities MockBackend::capabilities() const
{
    Capabilities capabilities;
    capabilities.pinned_host_memory = true;
    capabilities.device_to_device = true;
    return capabilities;
}

common::ResponseCode MockBackend::device_count(unsigned & count) const
{
    count = _count;
    return common::ResponseCode::Success;
}

common::ResponseCode MockBackend::open_device(unsigned ordinal, std::shared_ptr<Device> & device)
{
    ++opens;

    if (fail_open_device_at != 0 && opens == fail_open_device_at)
    {
        return common::ResponseCode::InvalidDevice;
    }

    if (ordinal >= _count)
    {
        return common::ResponseCode::InvalidDevice;
    }

    auto & held = _devices[ordinal];
    if (held == nullptr)
    {
        held = std::make_shared<MockDevice>();
    }

    device = held;
    return common::ResponseCode::Success;
}

std::shared_ptr<MockDevice> MockBackend::opened(unsigned ordinal) const
{
    const auto it = _devices.find(ordinal);
    return it != _devices.end() ? it->second : nullptr;
}

} // namespace runai::llm::streamer::device
