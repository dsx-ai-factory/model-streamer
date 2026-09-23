#pragma once

#include <atomic>
#include <memory>

#include "streamer/impl/batch/batch.h"
#include "streamer/impl/device_io/device_issuer/device_issuer.h"
#include "streamer/impl/device_io/staging_pool/staging_pool.h"
#include "streamer/impl/workload/workload.h"
#include "utils/threadpool/threadpool.h"

namespace runai::llm::streamer::impl
{

// The synchronous filesystem reader: one per thread of the filesystem pool.
//
// A Worker rather than a stateless Handler because of what it owns: its OWN pinned buffers. One pool
// per thread means a single consumer and no contention for buffers; a stateless handler shared by
// every thread has nowhere to put them.
//
// THREE BUFFERS. The pipeline is read, copy, in flight - so one is being filled while another is
// being copied and a third is on the link. Fewer serialises the read against the copy; more buys
// nothing this thread can use, because it reads one block at a time.
//
// The COPY is not issued here. Every reading thread hands its full buffer to one shared DeviceIssuer,
// which binds a context once and enqueues for all of them - see DeviceIssuer for why.
//
// Nothing is allocated until a device batch actually arrives: a host-only load pays for none of it.
class FileSystemWorker : public utils::Worker<Workload>
{
 public:
    // Buffers per reading thread. Not a knob: it is the depth of a three-stage pipeline, and this
    // thread can never use a fourth.
    static constexpr unsigned BuffersPerThread = 3;

    // `issuer` is shared by every thread of this pool. Null in tests that read to the host only.
    explicit FileSystemWorker(std::shared_ptr<DeviceWriter> writer = nullptr,
                              std::shared_ptr<DeviceIssuer> issuer = nullptr,
                              size_t block_bytesize = 0);

    ~FileSystemWorker() override;

    void execute(Workload && workload, std::atomic<bool> & stopped) override;
    void drain(std::atomic<bool> & stopped) override;
    bool idle() const override;

 private:
    // This thread's buffers, built the first time a device workload arrives. Against the device that
    // workload names - pinned memory is reachable from every context, so a later device reuses them.
    common::ResponseCode staging_for(const Workload & workload, DeviceStaging & out);

    const std::shared_ptr<DeviceWriter> _writer;
    const std::shared_ptr<DeviceIssuer> _issuer;
    const size_t _block_bytesize;

    std::shared_ptr<SharedStagingPool> _pool;
};

}; // namespace runai::llm::streamer::impl
