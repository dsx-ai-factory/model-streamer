/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/device_io/device_writer/device_writer.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

DeviceWriter::DeviceWriter(BackendLookup backends) :
    _backends(std::move(backends))
{
}

DeviceWriter::~DeviceWriter() = default;

DeviceWriter::Channels::~Channels()
{
    // Waiters first: each drains what it holds and returns those buffers. A pool then goes with the
    // last share of it - by reference count, not by any declaration order.
    _targets.clear();
}

common::ResponseCode DeviceWriter::Channels::open(const BackendLookup & backends, common::Device device, Target ** out)
{
    const std::lock_guard<std::mutex> guard(_mutex);

    const auto existing = _targets.find(device);
    if (existing != _targets.end())
    {
        *out = &existing->second;
        return common::ResponseCode::Success;
    }

    auto & opened = _opened[device.type];
    if (opened == nullptr)
    {
        const auto factory = backends ? backends(device.type) : BackendFactory();
        opened = factory ? factory() : nullptr;
        if (opened == nullptr)
        {
            return common::ResponseCode::DeviceUnavailable;
        }
    }

    Target target;
    auto code = opened->open_device(device.id, target.device);
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

    *out = &_targets.emplace(device, std::move(target)).first->second;
    return common::ResponseCode::Success;
}

unsigned DeviceWriter::Channels::count() const
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return static_cast<unsigned>(_targets.size());
}

common::ResponseCode DeviceWriter::open(common::Device device, Channel & out)
{
    out = nullptr;

    Target * target = nullptr;
    const auto code = _channels.open(_backends, device, &target);
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

    // These two are NOT atomic. Two workers can share a device, so another thread may enqueue between
    // them and this event then marks a later point on the stream. A stream is FIFO, so the event fires
    // late and never early: the buffer is held longer, the bytes are still right.
    auto code = target->device->memcpy_h2d_async(destination, buffer.data, bytesize, target->stream.get());

    // Which of the two failed decides whether this buffer is safe to give back.
    const bool enqueued = code == common::ResponseCode::Success;
    if (enqueued)
    {
        code = target->device->event_record(copy.event, target->stream.get());
    }

    if (code != common::ResponseCode::Success)
    {
        LOG(ERROR) << "[RunAI Streamer] failed to enqueue a copy of " << bytesize
                   << " bytes: " << code;

        if (enqueued)
        {
            // THE COPY IS ON THE STREAM and still reading out of this buffer; only its event failed,
            // so nothing marks when it ends and the waiter has nothing to wait on. Returning the
            // buffer now lets the next read overwrite the source mid-transfer.
            //
            // Synchronising the whole stream is heavier than waiting for one event, which is why it is
            // not the normal path - but with no event there is nothing finer to wait for.
            //
            // WHAT IT ACTUALLY SAVES is narrow. A record fails either because the event is bad, which
            // leaves the context healthy and this wait able to confirm the copy, or because the context
            // is broken, which fails this wait too. Our events cannot be from the wrong context -
            // EventPool is per device - so the first case means an invalid handle, which would be our
            // own bug. Kept because it costs one call on a path that has already failed, and because
            // the alternative is retiring a buffer whose copy had in fact finished.
            //
            // StreamWaiter has no equivalent fallback, and that is deliberate: by the time it runs the
            // record has already succeeded, so only a broken context can fail its wait.
            const auto drained = target->device->stream_synchronize(target->stream.get());
            if (drained != common::ResponseCode::Success)
            {
                // The copy may still be reading, and nothing left can say when it stops. Retiring is
                // what keeps it out of the next reader's hands; the pool reports once every buffer has
                // gone this way, so no one waits for it.
                //
                // The event is safe to return: the record failed, so it was never put on the stream.
                //
                // DeviceDriverError, NOT the code event_record gave us. That is DeviceTransferError,
                // which tells a caller the destination is free - and the copy is on the stream here
                // and may still land in it.
                pool->retire(buffer, common::ResponseCode::DeviceDriverError);
                copy.events->release(copy.event);
                return common::ResponseCode::DeviceDriverError;
            }
        }

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
