#include "device/mock/mock_device.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>
#include <mutex>

#include "common/exception/exception.h"

namespace runai::llm::streamer::device
{

namespace
{

// Tokens, not addresses. Casting a counter keeps every handle distinct and obviously not a pointer
// to anything, so a test that dereferences one fails loudly instead of corrupting memory.
void * token(uintptr_t value)
{
    return reinterpret_cast<void *>(value);
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

    if (throw_host_alloc_at != 0 && host_allocs == throw_host_alloc_at)
    {
        throw common::Exception(common::ResponseCode::DeviceOutOfMemory);
    }

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

std::atomic<uintptr_t> MockDevice::_next_token{1};

common::ResponseCode MockDevice::stream_create(StreamHandle & stream)
{
    ++streams_created;

    const std::lock_guard<std::mutex> guard(_mutex);
    stream = token(_next_token++);
    _streams.insert(stream);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::stream_destroy(StreamHandle stream)
{
    ++streams_destroyed;

    const std::lock_guard<std::mutex> guard(_mutex);
    _streams.erase(stream);
    return common::ResponseCode::Success;
}

common::ResponseCode MockDevice::stream_synchronize(StreamHandle)
{
    ++stream_syncs;
    // AS CUDA DOES: cuStreamSynchronize failing reports DeviceTransferError (cuda_device.cc). The
    // mock reported DeviceDriverError, which is the code DeviceWriter decides on afterwards - so a
    // test could expect the right answer for the wrong reason.
    return fail_stream_synchronize ? common::ResponseCode::DeviceTransferError
                                   : common::ResponseCode::Success;
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

common::ResponseCode MockDevice::event_record(EventHandle event, StreamHandle stream)
{
    const std::lock_guard<std::mutex> guard(_mutex);

    // An event and a stream belong to the context that created them. Recording one device's event on
    // another's stream is CUDA_ERROR_INVALID_HANDLE, not a silent success.
    if (_ready.count(event) == 0 || _streams.count(stream) == 0)
    {
        ++foreign_records;
        return common::ResponseCode::InvalidParameterError;
    }

    if (fail_event_record)
    {
        // AS CUDA DOES: cuEventRecord failing reports DeviceTransferError (cuda_device.cc).
        return common::ResponseCode::DeviceTransferError;
    }

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

void MockDevice::hold_copies()
{
    const std::lock_guard<std::mutex> guard(_hold_mutex);
    _copies_held = true;
}

void MockDevice::release_copies()
{
    {
        const std::lock_guard<std::mutex> guard(_hold_mutex);
        _copies_held = false;
    }

    _hold.notify_all();
}

common::ResponseCode MockDevice::event_synchronize(EventHandle event)
{
    const unsigned nth = ++event_syncs;

    {
        // BEFORE the state lock. A held copy sleeps here, and sleeping under _mutex would stop the
        // test thread reading a counter or the pool handing a buffer back.
        std::unique_lock<std::mutex> hold(_hold_mutex);
        _hold.wait(hold, [this]() { return !_copies_held; });
    }

    const std::lock_guard<std::mutex> guard(_mutex);
    _ready[event] = true;

    const unsigned from = fail_event_synchronize_from.load();
    const bool fails = fail_event_synchronize.load() || (from != 0 && nth >= from);

    return fails ? common::ResponseCode::DeviceTransferError : common::ResponseCode::Success;
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
