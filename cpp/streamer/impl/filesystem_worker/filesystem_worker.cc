#include "streamer/impl/filesystem_worker/filesystem_worker.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

void FileSystemWorker::execute(Workload && workload, std::atomic<bool> & stopped)
{
    // This reader writes straight into the caller's destination with pread, which for a device
    // pointer is a segmentation fault rather than an error. Refused until it can stage through pinned
    // memory like the async readers do.
    if (!workload.device().is_host())
    {
        LOG(ERROR) << "The synchronous reader cannot write to " << workload.device()
                   << " yet; this submission needs an asynchronous file system strategy";

        workload.fail(common::ResponseCode::UnsupportedDeviceType);
        return;
    }

    workload.execute(stopped);
}

void FileSystemWorker::drain(std::atomic<bool> & /* stopped */)
{
}

bool FileSystemWorker::idle() const
{
    return true;
}

}; // namespace runai::llm::streamer::impl
