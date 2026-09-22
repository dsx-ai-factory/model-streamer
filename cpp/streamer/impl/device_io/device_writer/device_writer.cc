#include "streamer/impl/device_io/device_writer/device_writer.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

DeviceWriter::DeviceWriter(BackendFactory backend, StagingPool::Params params) :
    _backend(std::move(backend)),
    _params(params)
{
}

DeviceWriter::~DeviceWriter() = default;

DeviceWriter::Channels::~Channels()
{
    // Waiters first: each drains what it holds and returns those buffers. The pool then goes when
    // the last share of it is dropped, which is after every waiter has stopped - by reference
    // count rather than by the order these members happen to be declared in.
    _targets.clear();
}

common::ResponseCode DeviceWriter::Channels::open(const BackendFactory & backend,
                                                  const StagingPool::Params & params,
                                                  unsigned ordinal,
                                                  Target ** out)
{
    const std::lock_guard<std::mutex> guard(_mutex);

    const auto existing = _targets.find(ordinal);
    if (existing != _targets.end())
    {
        *out = &existing->second;
        return common::ResponseCode::Success;
    }

    if (_opened == nullptr)
    {
        _opened = backend != nullptr ? backend() : nullptr;
        if (_opened == nullptr)
        {
            return common::ResponseCode::DeviceUnavailable;
        }
    }

    Target target;
    auto code = _opened->open_device(ordinal, target.device);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    device::StreamHandle stream = nullptr;
    code = target.device->stream_create(stream);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }
    target.stream = device::OwnedStream(stream, device::StreamDeleter{target.device});

    // The pool is built against the first device opened. Pinned memory is allocated through one
    // context and reachable from every other, so later devices share these buffers.
    if (_pool == nullptr)
    {
        _pool = std::make_shared<SharedStagingPool>(target.device, params);
    }

    target.pool = _pool;
    target.waiter = std::make_unique<StreamWaiter>(target.device, _pool);

    *out = &_targets.emplace(ordinal, std::move(target)).first->second;
    return common::ResponseCode::Success;
}

unsigned DeviceWriter::Channels::created() const
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return _pool != nullptr ? _pool->created() : 0;
}

unsigned DeviceWriter::Channels::count() const
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return static_cast<unsigned>(_targets.size());
}

common::ResponseCode DeviceWriter::open(unsigned ordinal, Channel & out)
{
    out = nullptr;

    Target * target = nullptr;
    const auto code = _channels.open(_backend, _params, ordinal, &target);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    out = target;
    return common::ResponseCode::Success;
}

common::ResponseCode DeviceWriter::take(Channel channel, StagingBuffer & out)
{
    out = StagingBuffer{};

    const Target * const target = static_cast<const Target *>(channel);
    if (target == nullptr)
    {
        return common::ResponseCode::InvalidParameterError;
    }

    // No lock of ours and no atomic: the pool hangs off the handle, and the handle came from
    // open(), which held the lock while it wrote this.
    return target->pool->try_acquire(out);
}

common::ResponseCode DeviceWriter::write(Channel channel,
                                         const StagingBuffer & buffer,
                                         size_t bytesize,
                                         void * destination,
                                         Completion on_done)
{
    // Const-cast because the handle is opaque to the caller and mutable to us. The target it names
    // was created under the lock before the handle was handed out, and targets are never removed.
    Target * const target = const_cast<Target *>(static_cast<const Target *>(channel));
    if (target == nullptr)
    {
        // A caller bug, not a runtime case: a channel comes only from open(), which fills it on
        // success alone. Loud, because the buffer cannot be returned without one and would
        // otherwise look like a leak from the pool's side.
        // LOG rather than ASSERT: ASSERT throws, and every other error here is a return code. One
        // function reporting two ways makes every call site handle both.
        LOG(ERROR) << "[RunAI Streamer] write() called with a null channel; the buffer was not returned";
        return common::ResponseCode::InvalidParameterError;
    }

    if (bytesize > buffer.bytesize)
    {
        LOG(ERROR) << "[RunAI Streamer] asked to copy " << bytesize << " bytes from a staging buffer of "
                   << buffer.bytesize;
        target->pool->release(buffer);
        return common::ResponseCode::InvalidParameterError;
    }

    // Nothing between the copy and the record, so the event marks this copy and not whatever
    // another thread enqueued on the same stream in between.
    auto code = target->device->memcpy_h2d_async(destination, buffer.data, bytesize, target->stream.get());
    if (code == common::ResponseCode::Success)
    {
        code = target->device->event_record(buffer.event, target->stream.get());
    }

    if (code != common::ResponseCode::Success)
    {
        LOG(ERROR) << "[RunAI Streamer] failed to enqueue a copy of " << bytesize
                   << " bytes: " << code;

        // on_done is NOT called: the return value is the report. Doing both would tell a caller
        // twice, and a caller that watched only the callback would miss the other error paths.
        target->pool->release(buffer);
        return code;
    }

    // From here the buffer belongs to the waiter, which returns it once the copy has landed.
    target->waiter->enqueue(buffer, std::move(on_done));
    return common::ResponseCode::Success;
}

unsigned DeviceWriter::devices() const
{
    return _channels.count();
}

unsigned DeviceWriter::buffers() const
{
    return _channels.created();
}

} // namespace runai::llm::streamer::impl
