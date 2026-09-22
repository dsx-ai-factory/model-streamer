#pragma once

#include <atomic>

#include "streamer/impl/workload/workload.h"
#include "utils/threadpool/threadpool.h"

namespace runai::llm::streamer::impl
{

// The synchronous filesystem reader: one per thread of the filesystem pool.
//
// A Worker rather than a stateless Handler, though reading needs no state, because a worker will own
// a DeviceWriterClient - which is not thread safe, so it needs a per-thread object to live in.
//
// The pool behaves the same either way: a read returns when it is done, so there is nothing to drain
// and the worker is always idle.
class FileSystemWorker : public utils::Worker<Workload>
{
 public:
    void execute(Workload && workload, std::atomic<bool> & stopped) override;
    void drain(std::atomic<bool> & stopped) override;
    bool idle() const override;
};

}; // namespace runai::llm::streamer::impl
