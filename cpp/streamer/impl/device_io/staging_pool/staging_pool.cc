#include "streamer/impl/device_io/staging_pool/staging_pool.h"

#include <algorithm>
#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

StagingPool::StagingPool(std::shared_ptr<device::Device> device, Params params) :
    _device(std::move(device)),
    _params(params)
{
}

// Nothing to wait for: a buffer comes back only after its copy's event was synchronised, so every
// buffer being back - which is this class's precondition - already means no DMA is reading out of
// this memory. The slabs are freed by their owners.
StagingPool::~StagingPool() = default;

bool StagingPool::hand_out(StagingBuffer & out)
{
    if (_free.empty())
    {
        return false;
    }

    out = _buffers[_free.front()];
    _free.pop_front();
    return true;
}

unsigned StagingPool::plan_slab(size_t & bytesize) const
{
    bytesize = 0;

    const unsigned room = _params.max_buffers - static_cast<unsigned>(_buffers.size());
    if (room == 0)
    {
        return 0;
    }

    // A pool of zero sized buffers would divide by zero below, which is SIGFPE rather than an error
    // the caller can see. Refused as if it were at its ceiling, so the pool hands out nothing.
    if (_params.buffer_bytesize == 0)
    {
        LOG(ERROR) << "[RunAI Streamer] a staging pool of zero sized buffers holds nothing";
        return 0;
    }

    unsigned per_slab = static_cast<unsigned>(_params.slab_bytesize / _params.buffer_bytesize);
    per_slab = std::min(std::max(per_slab, 1u), room);

    bytesize = static_cast<size_t>(per_slab) * _params.buffer_bytesize;
    return per_slab;
}

common::ResponseCode StagingPool::add_slab(size_t bytesize, unsigned per_slab, StagingBuffer & out)
{
    void * base = nullptr;
    const auto code = _device->host_alloc(bytesize, &base);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // Owned from here, so every path out frees it - including the one below that decides this slab
    // is no longer wanted, and a throw.
    device::OwnedPinned slab(base, device::PinnedDeleter{_device});

    const std::lock_guard<std::mutex> guard(_mutex);

    // Another consumer may have registered a slab while this one was pinning. The window is a
    // promise about how much memory is pinned at once, so the late slab is freed rather than
    // published. Nothing in the streamer shares a pool between consumers, so this costs nothing.
    if (_buffers.size() + per_slab <= _params.max_buffers)
    {
        _slabs.push_back(std::move(slab));

        for (unsigned i = 0; i < per_slab; ++i)
        {
            StagingBuffer buffer;
            buffer.data = static_cast<char *>(base) + static_cast<size_t>(i) * _params.buffer_bytesize;
            buffer.bytesize = _params.buffer_bytesize;
            buffer.index = static_cast<unsigned>(_buffers.size());

            _buffers.push_back(buffer);
            _free.push_back(buffer.index);
        }
    }

    hand_out(out);
    return common::ResponseCode::Success;
}

common::ResponseCode StagingPool::try_acquire(StagingBuffer & out)
{
    out = StagingBuffer{};

    size_t bytesize = 0;
    unsigned per_slab = 0;

    {
        const std::lock_guard<std::mutex> guard(_mutex);

        if (_stopped || hand_out(out))
        {
            return common::ResponseCode::Success;
        }

        per_slab = plan_slab(bytesize);
        if (per_slab == 0)
        {
            // At the ceiling with everything in flight. out stays invalid, which is not an error.
            return common::ResponseCode::Success;
        }
    }

    return add_slab(bytesize, per_slab, out);
}

common::ResponseCode StagingPool::acquire(StagingBuffer & out)
{
    out = StagingBuffer{};

    while (true)
    {
        size_t bytesize = 0;
        unsigned per_slab = 0;

        {
            std::unique_lock<std::mutex> lock(_mutex);

            if (_stopped || hand_out(out))
            {
                return common::ResponseCode::Success;
            }

            per_slab = plan_slab(bytesize);
            if (per_slab == 0)
            {
                // Everything is in flight and the pool is at its ceiling, so no slab is coming: the
                // only way forward is a buffer coming back. These threads have nothing else to do,
                // unlike an async engine's worker, so they wait rather than spin.
                _ready.wait(lock);
                continue;
            }
        }

        const auto code = add_slab(bytesize, per_slab, out);
        if (code != common::ResponseCode::Success)
        {
            return code;
        }

        if (out.valid())
        {
            return common::ResponseCode::Success;
        }

        // The slab was dropped because another consumer filled the window first. Go round: either
        // one of its buffers is free, or this thread waits for one.
    }
}

void StagingPool::release(const StagingBuffer & buffer)
{
    {
        const std::lock_guard<std::mutex> guard(_mutex);

        if (_free.size() >= _buffers.size())
        {
            // Only reachable by returning a buffer twice, or one that was never taken. Dropping it
            // loses a buffer; handing it out twice would give two readers the same memory.
            LOG(ERROR) << "[RunAI Streamer] staging buffer " << buffer.index << " returned to a full pool";
            return;
        }

        _free.push_back(buffer.index);
    }

    _ready.notify_one();
}

void StagingPool::stop()
{
    {
        const std::lock_guard<std::mutex> guard(_mutex);
        _stopped = true;
    }

    _ready.notify_all();
}

size_t StagingPool::buffer_bytesize() const
{
    return _params.buffer_bytesize;
}

unsigned StagingPool::created() const
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return static_cast<unsigned>(_buffers.size());
}

unsigned StagingPool::slabs() const
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return static_cast<unsigned>(_slabs.size());
}

} // namespace runai::llm::streamer::impl
