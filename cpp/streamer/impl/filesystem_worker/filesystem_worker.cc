#include "streamer/impl/filesystem_worker/filesystem_worker.h"

#include <utility>

namespace runai::llm::streamer::impl
{

void FileSystemWorker::execute(Workload && workload, std::atomic<bool> & stopped)
{
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
