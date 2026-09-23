#include "streamer/impl/device_io/event_pool/event_pool.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

EventPool::EventPool(std::shared_ptr<device::Device> device, unsigned max_events) :
    _device(std::move(device)),
    _max(max_events)
{
}

EventPool::~EventPool() = default;

common::ResponseCode EventPool::acquire(device::EventHandle & out)
{
    out = nullptr;

    const std::lock_guard<std::mutex> guard(_mutex);

    if (!_free.empty())
    {
        out = _free.back();
        _free.pop_back();
        return common::ResponseCode::Success;
    }

    if (_owned.size() >= _max)
    {
        return common::ResponseCode::Success;   // all in flight; not an error
    }

    device::EventHandle event = nullptr;
    const auto code = _device->event_create(event);
    if (code != common::ResponseCode::Success)
    {
        LOG(ERROR) << "[RunAI Streamer] could not create a copy event: " << code;
        return code;
    }

    _owned.emplace_back(event, device::EventDeleter{_device});
    out = event;
    return common::ResponseCode::Success;
}

void EventPool::release(device::EventHandle event)
{
    if (event == nullptr)
    {
        return;
    }

    const std::lock_guard<std::mutex> guard(_mutex);
    _free.push_back(event);
}

unsigned EventPool::created() const
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return static_cast<unsigned>(_owned.size());
}

} // namespace runai::llm::streamer::impl
