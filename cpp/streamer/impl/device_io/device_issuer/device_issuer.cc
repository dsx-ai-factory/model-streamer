#include "streamer/impl/device_io/device_issuer/device_issuer.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

namespace
{

// A lane's client never takes a buffer, so its buffer geometry describes a pool it will not build.
// Zero says that plainly: a pool sized for no buffers cannot be grown into by accident. The CEILING
// is not zero, because it also sizes this device's event pool, which the lane does use.
DeviceWriterClient::Buffers no_buffers()
{
    return DeviceWriterClient::Buffers{};
}

} // namespace

DeviceIssuer::DeviceIssuer(std::shared_ptr<DeviceWriter> writer) :
    _writer(std::move(writer))
{
}

DeviceIssuer::~DeviceIssuer() = default;

DeviceIssuer::Lane * DeviceIssuer::lane_for(common::Device device, common::ResponseCode & code)
{
    code = common::ResponseCode::Success;

    const std::lock_guard<std::mutex> guard(_mutex);

    const auto existing = _lanes.find(device);
    if (existing != _lanes.end())
    {
        return &existing->second;
    }

    auto client = std::make_shared<DeviceWriterClient>(_writer, no_buffers(), MaxCopiesInFlight);

    // Opened HERE rather than on the lane's thread so a device that cannot be reached is reported to
    // the reader while it is still listening, instead of through a completion.
    code = client->open(device);
    if (code != common::ResponseCode::Success)
    {
        return nullptr;
    }

    auto & lane = _lanes[device];
    lane.device = device;
    lane.client = client;

    // Started on its first message, and it binds this device's context then - once, never again,
    // because a lane only ever serves one device.
    //
    // The handler holds the client, not the lane: the thread that calls into it owns a share of it,
    // so no destruction order can take it away while there is still queued work to drain.
    lane.worker = std::make_unique<utils::DrainingWorker<Request>>(
        [this, device, client](Request && request) { issue(device, *client, std::move(request)); });

    return &lane;
}

void DeviceIssuer::submit(common::Device device,
                          std::shared_ptr<StagingPool> pool,
                          const StagingBuffer & buffer,
                          size_t bytesize,
                          void * destination,
                          Completion on_done)
{
    auto code = common::ResponseCode::Success;
    Lane * const lane = lane_for(device, code);

    if (lane == nullptr)
    {
        LOG(ERROR) << "[RunAI Streamer] no copy path to device " << device << ": " << code;

        // Returned before the report, as everywhere else on this path.
        pool->release(buffer);
        if (on_done)
        {
            on_done(code);
        }
        return;
    }

    // Outside the lock: the lane is stable, and a push must not wait behind another device's first
    // submission.
    lane->worker->push(Request{ std::move(pool), buffer, bytesize, destination, std::move(on_done) });
}

void DeviceIssuer::issue(common::Device device, DeviceWriterClient & client, Request && request)
{
    const auto code = client.write(device, request.pool, request.buffer,
                                   request.bytesize, request.destination, request.on_done);

    if (code != common::ResponseCode::Success)
    {
        // write() returned the buffer itself, but nobody is left to read that code: the reader moved
        // on the moment this was queued. So the completion is the only report, and it must fire here
        // because the waiter never saw this copy.
        LOG(ERROR) << "[RunAI Streamer] could not issue a copy to device " << device
                   << ": " << code;

        if (request.on_done)
        {
            request.on_done(code);
        }
    }
}

bool DeviceIssuer::running() const
{
    const std::lock_guard<std::mutex> guard(_mutex);

    for (const auto & entry : _lanes)
    {
        if (entry.second.worker->running())
        {
            return true;
        }
    }
    return false;
}

unsigned DeviceIssuer::issued() const
{
    const std::lock_guard<std::mutex> guard(_mutex);

    unsigned total = 0;
    for (const auto & entry : _lanes)
    {
        total += entry.second.worker->handled();
    }
    return total;
}

unsigned DeviceIssuer::devices() const
{
    const std::lock_guard<std::mutex> guard(_mutex);
    return static_cast<unsigned>(_lanes.size());
}

} // namespace runai::llm::streamer::impl
