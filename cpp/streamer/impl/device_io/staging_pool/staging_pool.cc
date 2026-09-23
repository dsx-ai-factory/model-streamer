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

// Nothing to wait for: a buffer comes back only after its copy's event was synchronised, so every
// buffer being back - which is this class's precondition - already means no DMA is reading out of
// this memory. The slabs are freed by their owners.
StagingPool::~StagingPool() = default;

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

unsigned StagingPool::plan_slab(size_t & bytesize) const
{
    const unsigned room = _params.max_buffers - static_cast<unsigned>(_buffers.size());
    if (room == 0)
    {
        bytesize = 0;
        return 0;
    }

    unsigned per_slab = static_cast<unsigned>(_params.slab_bytesize / _params.buffer_bytesize);
    per_slab = std::min(std::max(per_slab, 1u), room);

    bytesize = static_cast<size_t>(per_slab) * _params.buffer_bytesize;
    return per_slab;
}

void StagingPool::publish_slab(void * base, unsigned per_slab)
{
    // Owned from here, so every path out of this function frees it - including a throw.
    _slabs.emplace_back(base, device::PinnedDeleter{_device});

    for (unsigned i = 0; i < per_slab; ++i)
    {
        StagingBuffer buffer;
        buffer.data = static_cast<char *>(base) + static_cast<size_t>(i) * _params.buffer_bytesize;
        buffer.bytesize = _params.buffer_bytesize;
        buffer.index = static_cast<unsigned>(_buffers.size());

        _buffers.push_back(buffer);
    }
}

common::ResponseCode StagingPool::grow()
{
    size_t bytesize = 0;
    const unsigned per_slab = plan_slab(bytesize);
    if (per_slab == 0)
    {
        return common::ResponseCode::Success;
    }

    void * base = nullptr;
    const auto code = _device->host_alloc(bytesize, &base);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    publish_slab(base, per_slab);
    return common::ResponseCode::Success;
}

bool StagingPool::hand_out(StagingBuffer & out)
{
    unsigned index = 0;
    if (take(index))
    {
        out = _buffers[index];
        return true;
    }

    if (_next_new < _buffers.size())
    {
        out = _buffers[_next_new++];
        return true;
    }

    return false;
}

common::ResponseCode StagingPool::try_acquire(StagingBuffer & out)
{
    out = StagingBuffer{};

    if (hand_out(out))
    {
        return common::ResponseCode::Success;
    }

    const auto code = grow();
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    hand_out(out);

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

common::ResponseCode SharedStagingPool::grow_unlocked(std::unique_lock<std::mutex> & lock, bool & grew)
{
    grew = false;

    size_t bytesize = 0;
    const unsigned per_slab = plan_slab(bytesize);
    if (per_slab == 0)
    {
        return common::ResponseCode::Success;   // at the ceiling
    }

    grew = true;

    _growing = true;
    lock.unlock();

    // THE POINT OF ALL THIS. Pinning a slab is a driver call of several milliseconds, and the
    // StreamWaiter takes this same lock to hand buffers back - so holding it here would stall every
    // copy that lands while the pool is still filling.
    void * base = nullptr;
    const auto code = _device->host_alloc(bytesize, &base);

    lock.lock();
    _growing = false;

    if (code == common::ResponseCode::Success)
    {
        publish_slab(base, per_slab);
    }

    // Both paths: a waiter must not sleep through a slab that arrived, nor through one that failed.
    _ready.notify_all();
    return code;
}

common::ResponseCode SharedStagingPool::try_acquire(StagingBuffer & out)
{
    out = StagingBuffer{};

    std::unique_lock<std::mutex> lock(_mutex);

    if (hand_out(out))
    {
        return common::ResponseCode::Success;
    }

    // Another consumer is already pinning one. Growing again would take the pool past its ceiling, so
    // this reports what it honestly has: nothing free just now, which is not an error.
    if (_growing)
    {
        return common::ResponseCode::Success;
    }

    bool grew = false;
    const auto code = grow_unlocked(lock, grew);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    hand_out(out);
    return common::ResponseCode::Success;
}

common::ResponseCode SharedStagingPool::acquire(StagingBuffer & out)
{
    out = StagingBuffer{};

    std::unique_lock<std::mutex> lock(_mutex);

    while (true)
    {
        if (hand_out(out) || _stopped)
        {
            return common::ResponseCode::Success;
        }

        bool grew = false;
        if (!_growing)
        {
            const auto code = grow_unlocked(lock, grew);
            if (code != common::ResponseCode::Success)
            {
                return code;
            }
        }

        if (grew)
        {
            // The lock was released and retaken, so the pool may have changed either way. Go round.
            continue;
        }

        // Everything is in flight and the pool is at its ceiling - or another thread is pinning the
        // next slab. These threads have nothing else to do, unlike an async engine's worker, so they
        // wait for the StreamWaiter rather than spin.
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
