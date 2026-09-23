#include "streamer/impl/device_io/device_writer/device_writer.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

DeviceWriter::DeviceWriter(BackendFactory backend) :
    _backend(std::move(backend))
{
}

DeviceWriter::~DeviceWriter() = default;

DeviceWriter::Channels::~Channels()
{
    // Waiters first: each drains what it holds and returns those buffers. A pool then goes with the
    // last share of it - by reference count, not by any declaration order.
    _targets.clear();
}

common::ResponseCode DeviceWriter::Channels::open(const BackendFactory & backend, unsigned ordinal, Target ** out)
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

    // open_device RETAINS the primary context; it does not make it current on this thread. Every
    // driver call below - and every one the caller makes afterwards - needs a current context, so the
    // first one binds it here.
    code = target.device->bind_thread();
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

    target.waiter = std::make_unique<StreamWaiter>(target.device);

    *out = &_targets.emplace(ordinal, std::move(target)).first->second;
    return common::ResponseCode::Success;
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
    const auto code = _channels.open(_backend, ordinal, &target);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    out = target;
    return common::ResponseCode::Success;
}

std::shared_ptr<device::Device> DeviceWriter::device(Channel channel) const
{
    const Target * const target = static_cast<const Target *>(channel);
    return target != nullptr ? target->device : nullptr;
}

common::ResponseCode DeviceWriter::write(Channel channel,
                                         Copy copy,
                                         size_t bytesize,
                                         void * destination,
                                         Completion on_done)
{
    const auto & buffer = copy.buffer;
    const auto & pool = copy.pool;

    // Checked before anything is enqueued: the waiter returns the buffer from its own thread, so a
    // missing pool would otherwise crash there rather than fail here.
    //
    // LOG rather than ASSERT, which throws: one function reporting two ways makes every call site
    // handle both.
    if (pool == nullptr || copy.events == nullptr || copy.event == nullptr)
    {
        // The only path that keeps them - there is nowhere to give them back to.
        LOG(ERROR) << "[RunAI Streamer] write() called without a pool or an event; nothing was returned";
        return common::ResponseCode::InvalidParameterError;
    }

    // Const-cast because the handle is opaque to the caller and mutable to us. Its target was created
    // under the lock before the handle was handed out, and targets are never removed.
    Target * const target = const_cast<Target *>(static_cast<const Target *>(channel));
    if (target == nullptr)
    {
        // A caller bug: a channel comes only from open(), which clears it on failure.
        LOG(ERROR) << "[RunAI Streamer] write() called with a null channel";
        pool->release(buffer);
        copy.events->release(copy.event);
        return common::ResponseCode::InvalidParameterError;
    }

    if (bytesize > buffer.bytesize)
    {
        LOG(ERROR) << "[RunAI Streamer] asked to copy " << bytesize << " bytes from a staging buffer of "
                   << buffer.bytesize;
        pool->release(buffer);
        copy.events->release(copy.event);
        return common::ResponseCode::InvalidParameterError;
    }

    // Nothing between the copy and the record, so the event marks this copy and not what another
    // thread enqueued on the same stream.
    auto code = target->device->memcpy_h2d_async(destination, buffer.data, bytesize, target->stream.get());
    if (code == common::ResponseCode::Success)
    {
        code = target->device->event_record(copy.event, target->stream.get());
    }

    if (code != common::ResponseCode::Success)
    {
        LOG(ERROR) << "[RunAI Streamer] failed to enqueue a copy of " << bytesize
                   << " bytes: " << code;

        // on_done is NOT called: the return value is the report, and reporting both ways would tell
        // a caller twice.
        pool->release(buffer);
        copy.events->release(copy.event);
        return code;
    }

    // From here both belong to the waiter, which returns them once the copy has landed.
    target->waiter->enqueue(std::move(copy), std::move(on_done));
    return common::ResponseCode::Success;
}

unsigned DeviceWriter::devices() const
{
    return _channels.count();
}

} // namespace runai::llm::streamer::impl
