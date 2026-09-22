#include "streamer/impl/device_io/staging_pool/staging_pool.h"

#include <algorithm>
#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

StagingPool::StagingPool(std::shared_ptr<device::Device> device, Params params) :
    _device(std::move(device)),
    _params(params),
    _free(params.max_buffers, 0)
{
}

StagingPool::~StagingPool()
{
    // Every event, not just those believed outstanding: a ready event returns at once, and tracking
    // which are in flight is the bookkeeping that would be racy.
    //
    // Only the WAIT is here. The events and the slabs are freed by their owners, in the member
    // order below - which is why this function cannot forget one.
    for (const auto & buffer : _buffers)
    {
        _device->event_synchronize(buffer.event);
    }
}

bool StagingPool::take(unsigned & index)
{
    const size_t head = _head.load(std::memory_order_relaxed);
    if (head == _tail.load(std::memory_order_acquire))
    {
        return false;
    }

    index = _free[head % _free.size()];
    _head.store(head + 1, std::memory_order_release);
    return true;
}

void StagingPool::give(unsigned index)
{
    const size_t tail = _tail.load(std::memory_order_relaxed);

    if (tail - _head.load(std::memory_order_acquire) >= _free.size())
    {
        // Only reachable by returning a buffer twice, or one that was never taken. Dropping it
        // leaks a buffer; writing it would corrupt an entry another thread is reading.
        LOG(ERROR) << "[RunAI Streamer] staging buffer " << index << " returned to a full pool";
        return;
    }

    _free[tail % _free.size()] = index;
    _tail.store(tail + 1, std::memory_order_release);
}

common::ResponseCode StagingPool::grow()
{
    const unsigned room = _params.max_buffers - static_cast<unsigned>(_buffers.size());
    if (room == 0)
    {
        return common::ResponseCode::Success;
    }

    unsigned per_slab = static_cast<unsigned>(_params.slab_bytesize / _params.buffer_bytesize);
    per_slab = std::min(std::max(per_slab, 1u), room);

    const size_t bytesize = static_cast<size_t>(per_slab) * _params.buffer_bytesize;

    void * base = nullptr;
    const auto code = _device->host_alloc(bytesize, &base);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // Owned from here, so every path out of this function frees it - including a throw.
    _slabs.emplace_back(base, device::PinnedDeleter{_device});

    for (unsigned i = 0; i < per_slab; ++i)
    {
        StagingBuffer buffer;
        buffer.data = static_cast<char *>(base) + static_cast<size_t>(i) * _params.buffer_bytesize;
        buffer.bytesize = _params.buffer_bytesize;
        buffer.index = static_cast<unsigned>(_buffers.size());

        const auto event = _device->event_create(buffer.event);
        if (event != common::ResponseCode::Success)
        {
            if (i != 0)
            {
                // Earlier buffers of this slab are usable, so it stays and the caller gets one.
                return common::ResponseCode::Success;
            }

            // Nothing in this slab can be used. Free it and SAY SO: reporting success here would
            // send acquire() round again, and grow() would allocate another slab that fails the
            // same way - pinned memory growing without bound under the very condition that made
            // the event fail.
            _slabs.pop_back();
            return event;
        }

        _events.emplace_back(buffer.event, device::EventDeleter{_device});
        _buffers.push_back(buffer);
    }

    return common::ResponseCode::Success;
}

common::ResponseCode StagingPool::try_acquire(StagingBuffer & out)
{
    out = StagingBuffer{};

    unsigned index = 0;
    if (take(index))
    {
        out = _buffers[index];
        return common::ResponseCode::Success;
    }

    if (_next_new == _buffers.size())
    {
        const auto code = grow();
        if (code != common::ResponseCode::Success)
        {
            return code;
        }
    }

    if (_next_new < _buffers.size())
    {
        out = _buffers[_next_new++];
    }

    // out stays invalid when the pool is at its ceiling and everything is in flight. Not an error.
    return common::ResponseCode::Success;
}

void StagingPool::release(const StagingBuffer & buffer)
{
    give(buffer.index);
}

unsigned StagingPool::created() const
{
    return static_cast<unsigned>(_buffers.size());
}

unsigned StagingPool::slabs() const
{
    return static_cast<unsigned>(_slabs.size());
}

common::ResponseCode SharedStagingPool::try_acquire(StagingBuffer & out)
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return StagingPool::try_acquire(out);
}

common::ResponseCode SharedStagingPool::acquire(StagingBuffer & out)
{
    std::unique_lock<std::mutex> lock(_mutex);

    while (true)
    {
        const auto code = StagingPool::try_acquire(out);
        if (code != common::ResponseCode::Success || out.valid() || _stopped)
        {
            return code;
        }

        // Everything is in flight and the pool is at its ceiling. These threads have nothing else
        // to do, unlike an async engine's worker, so they wait for the reaper rather than spin.
        _ready.wait(lock);
    }
}

void SharedStagingPool::release(const StagingBuffer & buffer)
{
    {
        const std::lock_guard<std::mutex> guard(_mutex);
        StagingPool::release(buffer);
    }
    _ready.notify_one();
}

void SharedStagingPool::stop()
{
    {
        const std::lock_guard<std::mutex> guard(_mutex);
        _stopped = true;
    }
    _ready.notify_all();
}

} // namespace runai::llm::streamer::impl
