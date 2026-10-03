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
    // Driver calls need a context and a new thread inherits none. Done on the first message rather
    // than at construction, so a waiter that is never used touches no driver.
    if (!_thread_bound)
    {
        const auto code = _device->bind_thread();
        if (code != common::ResponseCode::Success)
        {
            LOG(ERROR) << "[RunAI Streamer] stream waiter could not bind a device context: " << code;
        }
        _thread_bound = true;
    }

    const auto code = _device->event_synchronize(entry.copy.event);

    // RETURNED BEFORE THE COMPLETION IS REPORTED. on_done is what lets the worker free its window
    // slot and submit the next chunk, and that chunk immediately asks for a buffer - so reporting
    // first hands out the slot while this buffer is still ours, and the pool comes back empty with
    // the window not full.
    //
    // The event before the buffer, for the same reason one step further in: a copy holds one of each
    // and the two pools share a ceiling.
    entry.copy.events->release(entry.copy.event);
    entry.copy.pool->release(entry.copy.buffer);

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
