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

void StreamWaiter::enqueue(std::shared_ptr<StagingPool> pool, const StagingBuffer & buffer, Completion on_done)
{
    _worker.push(Entry{std::move(pool), buffer, std::move(on_done)});
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

    const auto code = _device->event_synchronize(entry.buffer.event);

    if (entry.on_done)
    {
        entry.on_done(code);
    }

    // Returned even when the copy failed. A buffer lost on an error path is a deadlock that arrives
    // later.
    entry.pool->release(entry.buffer);
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
