#include "streamer/impl/staging_pool/staging_pool.h"

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
    for (const auto & buffer : _buffers)
    {
        _device->event_synchronize(buffer.event);
        _device->event_destroy(buffer.event);
    }

    for (const auto & slab : _slabs)
    {
        _device->host_free(slab.base);
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

    Slab slab;
    slab.bytesize = static_cast<size_t>(per_slab) * _params.buffer_bytesize;

    const auto code = _device->host_alloc(slab.bytesize, &slab.base);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    for (unsigned i = 0; i < per_slab; ++i)
    {
        StagingBuffer buffer;
        buffer.data = static_cast<char *>(slab.base) + static_cast<size_t>(i) * _params.buffer_bytesize;
        buffer.bytesize = _params.buffer_bytesize;
        buffer.index = static_cast<unsigned>(_buffers.size());

        const auto event = _device->event_create(buffer.event);
        if (event != common::ResponseCode::Success)
        {
            // The slab is kept: its earlier buffers are already usable, and freeing it now would
            // invalidate them.
            _slabs.push_back(slab);
            return _buffers.empty() ? event : common::ResponseCode::Success;
        }

        _buffers.push_back(buffer);
    }

    _slabs.push_back(slab);
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
