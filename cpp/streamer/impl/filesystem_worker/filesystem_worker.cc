#include "streamer/impl/filesystem_worker/filesystem_worker.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

FileSystemWorker::FileSystemWorker(std::shared_ptr<DeviceWriter> writer,
                                   std::shared_ptr<DeviceIssuer> issuer,
                                   size_t block_bytesize) :
    _writer(std::move(writer)),
    _issuer(std::move(issuer)),
    _block_bytesize(block_bytesize)
{
}

FileSystemWorker::~FileSystemWorker() = default;

common::ResponseCode FileSystemWorker::staging_for(const Workload & workload, DeviceStaging & out)
{
    out = DeviceStaging{};

    if (_writer == nullptr || _issuer == nullptr)
    {
        LOG(ERROR) << "This reader has no copy path, so it cannot serve " << workload.device();
        return common::ResponseCode::DeviceUnavailable;
    }

    if (_block_bytesize == 0)
    {
        // A buffer of no bytes reads no bytes, so the batch would take one per turn and never
        // advance. The streamer's config never allows it; only a caller that built this reader
        // directly can.
        LOG(ERROR) << "This reader has no block size, so it cannot stage for " << workload.device();
        return common::ResponseCode::InvalidParameterError;
    }

    if (_pool == nullptr)
    {
        // Through the device this workload names. Opening it here rather than in the issuer is what
        // gives this thread a context, which pinning needs.
        DeviceWriter::Channel channel = nullptr;
        auto code = _writer->open(workload.device().id, channel);
        if (code != common::ResponseCode::Success)
        {
            return code;
        }

        const auto device = _writer->device(channel);
        code = device->bind_thread();
        if (code != common::ResponseCode::Success)
        {
            return code;
        }

        StagingPool::Params params;
        params.buffer_bytesize = _block_bytesize;
        params.slab_bytesize = _block_bytesize;   // one call per buffer: three of them, once
        params.max_buffers = BuffersPerThread;

        _pool = std::make_shared<StagingPool>(device, params);
    }

    out.pool = _pool;
    out.issuer = _issuer.get();
    return common::ResponseCode::Success;
}

void FileSystemWorker::execute(Workload && workload, std::atomic<bool> & stopped)
{
    if (workload.device().is_host())
    {
        workload.execute(stopped);
        return;
    }

    DeviceStaging staging;
    const auto code = staging_for(workload, staging);
    if (code != common::ResponseCode::Success)
    {
        // Failed rather than read: pread cannot target device memory, so reading anyway would be a
        // segmentation fault. The ranges are still owed a response.
        workload.fail(code);
        return;
    }

    workload.execute(stopped, &staging);
}

void FileSystemWorker::drain(std::atomic<bool> & /* stopped */)
{
}

bool FileSystemWorker::idle() const
{
    return true;
}

}; // namespace runai::llm::streamer::impl
