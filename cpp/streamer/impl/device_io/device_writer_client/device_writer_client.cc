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

common::ResponseCode DeviceWriterClient::channel_for(unsigned ordinal, DeviceWriter::Channel & out)
{
    out = nullptr;

    const auto existing = _channels.find(ordinal);
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
    const auto code = _writer->open(ordinal, channel);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // Built against the FIRST device this worker names, and serving every later one: pinned memory
    // is allocated through a context and reachable from all of them.
    if (_pool == nullptr)
    {
        StagingPool::Params params;
        params.buffer_bytesize = _buffers.buffer_bytesize;
        params.slab_bytesize = _buffers.slab_bytesize;
        params.max_buffers = _max_buffers;

        _pool = std::make_shared<SharedStagingPool>(_writer->device(channel), params);
    }

    _channels.emplace(ordinal, channel);
    out = channel;
    return common::ResponseCode::Success;
}

common::ResponseCode DeviceWriterClient::take(unsigned ordinal, StagingBuffer & out)
{
    out = StagingBuffer{};

    DeviceWriter::Channel channel = nullptr;
    const auto code = channel_for(ordinal, channel);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    return _pool->try_acquire(out);
}

common::ResponseCode DeviceWriterClient::write(unsigned ordinal,
                                               const StagingBuffer & buffer,
                                               size_t bytesize,
                                               void * destination,
                                               Completion on_done)
{
    const auto existing = _channels.find(ordinal);
    if (existing == _channels.end())
    {
        // take() opens the ordinal it is asked for, so reaching this means the bytes are going to a
        // device this worker never read for. The buffer still comes back: one pool, any ordinal.
        LOG(ERROR) << "[RunAI Streamer] write to device " << ordinal << " which was never opened";
        if (_pool != nullptr)
        {
            _pool->release(buffer);
        }
        return common::ResponseCode::InvalidParameterError;
    }

    return _writer->write(existing->second, _pool, buffer, bytesize, destination, std::move(on_done));
}

unsigned DeviceWriterClient::devices() const
{
    return static_cast<unsigned>(_channels.size());
}

unsigned DeviceWriterClient::buffers() const
{
    return _pool != nullptr ? _pool->created() : 0;
}

} // namespace runai::llm::streamer::impl
