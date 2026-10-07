/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/device_io/stream_waiter/stream_waiter.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

StreamWaiter::StreamWaiter(std::shared_ptr<device::Device> device) :
    _device(std::move(device)),
    _worker([this](Entry && entry) { wait_for(std::move(entry)); })
{
}

StreamWaiter::~StreamWaiter() = default;

void StreamWaiter::enqueue(Copy copy, Completion on_done)
{
    _worker.push(Entry{std::move(copy), std::move(on_done)});
}

void StreamWaiter::wait_for(Entry && entry)
{
    auto code = common::ResponseCode::Success;

    // Driver calls need a context and a new thread inherits none. Done on the first message rather
    // than at construction, so a waiter that is never used touches no driver.
    if (!_thread_bound)
    {
        code = _device->bind_thread();
        if (code == common::ResponseCode::Success)
        {
            _thread_bound = true;
        }
        else if (!_bind_failure_logged)
        {
            // Once. Without a context every copy below fails the same way, and one line per copy
            // would bury the reason in its own repetition.
            LOG(ERROR) << "[RunAI Streamer] stream waiter could not bind a device context: " << code
                       << ". Copies cannot be waited for, and their staging buffers are retained";
            _bind_failure_logged = true;
        }
    }

    // SKIPPED when the bind failed, rather than called and left to fail: the answer would be the
    // same, and the log would name the wait instead of the context that was never bound.
    if (code == common::ResponseCode::Success)
    {
        code = _device->event_synchronize(entry.copy.event);

        // Here rather than in the branch below, which a failed bind also reaches without ever
        // waiting - and before the code is replaced, which loses what the driver said.
        if (code != common::ResponseCode::Success)
        {
            LOG(ERROR) << "[RunAI Streamer] could not wait for a copy to land (" << code
                       << "); its staging buffer is retained because the copy may still be reading it";
        }
    }

    if (code != common::ResponseCode::Success)
    {
        // NOTHING SAYS THE COPY STOPPED, so the buffer is not handed to the next chunk and the
        // caller is told its destination may still be written.
        //
        // NO stream_synchronize FALLBACK, unlike DeviceWriter::write. By the time a copy reaches this
        // thread its event RECORD has already succeeded - write() returns instead of enqueueing
        // otherwise - so only a broken context can fail this wait, and that fails a stream wait too.
        //
        // DeviceDriverError, not the driver's DeviceTransferError: that code promises the destination
        // is free. Same decision as DeviceWriter::write when it cannot drain the stream.
        code = common::ResponseCode::DeviceDriverError;

        // The event IS given back. A later copy records on it, and a record supersedes the pending
        // one, so a stale signal cannot reach the next waiter. Worth saying because the buffer beside
        // it is treated the opposite way.
        entry.copy.events->release(entry.copy.event);
        entry.copy.pool->retire(entry.copy.buffer, code);
    }
    else
    {
        // RETURNED BEFORE THE COMPLETION IS REPORTED. on_done is what lets the worker free its window
        // slot and submit the next chunk, and that chunk immediately asks for a buffer - so reporting
        // first hands out the slot while this buffer is still ours, and the pool comes back empty with
        // the window not full.
        //
        // The event before the buffer, for the same reason one step further in: a copy holds one of
        // each and the two pools share a ceiling.
        entry.copy.events->release(entry.copy.event);
        entry.copy.pool->release(entry.copy.buffer);
    }

    if (entry.on_done)
    {
        entry.on_done(code);
    }
}

void StreamWaiter::stop()
{
    _worker.stop();
}

bool StreamWaiter::running() const
{
    return _worker.running();
}

unsigned StreamWaiter::completed() const
{
    return _worker.handled();
}

} // namespace runai::llm::streamer::impl
