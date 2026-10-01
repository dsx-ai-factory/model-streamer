#include "streamer/impl/device_io/device_writer_client/device_writer_client.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

DeviceWriterClient::DeviceWriterClient(std::shared_ptr<DeviceWriter> writer, Buffers buffers, unsigned max_buffers) :
    _writer(std::move(writer)),
    _buffers(buffers),
    _max_buffers(max_buffers)
{
}

DeviceWriterClient::~DeviceWriterClient() = default;

common::ResponseCode DeviceWriterClient::channel_for(common::Device device, DeviceWriter::Channel & out)
{
    out = nullptr;

    const auto existing = _channels.find(device);
    if (existing != _channels.end())
    {
        out = existing->second;
        return common::ResponseCode::Success;
    }

    if (_writer == nullptr)
    {
        return common::ResponseCode::DeviceUnavailable;
    }

    DeviceWriter::Channel channel = nullptr;
    const auto code = _writer->open(device, channel);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // Per device, because an event belongs to the context that created it. Created here, where the
    // device is known, and sized by the same window as the buffers - a copy needs one of each.
    _events.emplace(device, std::make_shared<EventPool>(_writer->device(channel), _max_buffers));

    _channels.emplace(device, channel);
    out = channel;
    return common::ResponseCode::Success;
}

common::ResponseCode DeviceWriterClient::bind(common::Device device, const DeviceWriter::Channel & channel)
{
    if (_bound && _bound_device == device)
    {
        return common::ResponseCode::Success;
    }

    const auto opened = _writer->device(channel);
    ASSERT(opened != nullptr) << "binding a channel that names no device";

    const auto code = opened->bind_thread();
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    _bound = true;
    _bound_device = device;
    return common::ResponseCode::Success;
}

common::ResponseCode DeviceWriterClient::open(common::Device device)
{
    DeviceWriter::Channel channel = nullptr;
    return channel_for(device, channel);
}

common::ResponseCode DeviceWriterClient::take(common::Device device, StagingBuffer & out)
{
    out = StagingBuffer{};

    DeviceWriter::Channel channel = nullptr;
    auto code = channel_for(device, channel);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // Pinning and creating an event are driver calls like any other.
    code = bind(device, channel);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // Built on the FIRST TAKE, against the first device this client names, and serving every later
    // one: pinned memory is allocated through a context and reachable from all of them.
    //
    // Here rather than when a channel opens, so a client that never takes owns no pool at all - which
    // is what the synchronous reader's issuer is, since there the buffers belong to the reading
    // threads and only the copy is shared.
    if (_pool == nullptr)
    {
        StagingPool::Params params;
        params.buffer_bytesize = _buffers.buffer_bytesize;
        params.slab_bytesize = _buffers.slab_bytesize;
        params.max_buffers = _max_buffers;

        _pool = std::make_shared<StagingPool>(_writer->device(channel), params);
    }

    return _pool->try_acquire(out);
}

common::ResponseCode DeviceWriterClient::write(common::Device device,
                                               const StagingBuffer & buffer,
                                               size_t bytesize,
                                               void * destination,
                                               Completion on_done)
{
    return write(device, _pool, buffer, bytesize, destination, std::move(on_done));
}

common::ResponseCode DeviceWriterClient::write(common::Device device,
                                               const std::shared_ptr<StagingPool> & pool,
                                               const StagingBuffer & buffer,
                                               size_t bytesize,
                                               void * destination,
                                               Completion on_done)
{
    const auto existing = _channels.find(device);
    if (existing == _channels.end())
    {
        // take() opens the device it is asked for, so reaching this means the bytes are going to a
        // device this worker never read for. The buffer still comes back: one pool, any device.
        LOG(ERROR) << "[RunAI Streamer] write to device " << device << " which was never opened";
        if (pool != nullptr)
        {
            pool->release(buffer);
        }
        return common::ResponseCode::InvalidParameterError;
    }

    // The copy, the event record and creating an event all run in this device's context.
    auto code = bind(device, existing->second);
    if (code != common::ResponseCode::Success)
    {
        pool->release(buffer);
        return code;
    }

    DeviceWriter::Copy copy;
    copy.pool = pool;
    copy.buffer = buffer;
    copy.events = _events.at(device);

    code = copy.events->acquire(copy.event);
    if (code != common::ResponseCode::Success || copy.event == nullptr)
    {
        // The event pool has the same ceiling as the buffers and a copy takes one of each, so running
        // out is not a state the window allows.
        LOG(ERROR) << "[RunAI Streamer] no copy event for device " << device << ": " << code;
        pool->release(buffer);
        return code != common::ResponseCode::Success ? code : common::ResponseCode::UnknownError;
    }

    return _writer->write(existing->second, std::move(copy), bytesize, destination, std::move(on_done));
}

StagingPool::Slab DeviceWriterClient::slab_at(unsigned index) const
{
    return _pool == nullptr ? StagingPool::Slab{} : _pool->slab_at(index);
}

void DeviceWriterClient::release(const StagingBuffer & buffer)
{
    ASSERT(_pool != nullptr) << "releasing a buffer to a client that never took one";
    _pool->release(buffer);
}

unsigned DeviceWriterClient::devices() const
{
    return static_cast<unsigned>(_channels.size());
}

unsigned DeviceWriterClient::events(common::Device device) const
{
    const auto it = _events.find(device);
    return it != _events.end() ? it->second->created() : 0;
}

unsigned DeviceWriterClient::buffers() const
{
    return _pool != nullptr ? _pool->created() : 0;
}

} // namespace runai::llm::streamer::impl
