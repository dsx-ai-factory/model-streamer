#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <sys/types.h>

#include "posix_io/mount_capabilities/mount_capabilities.h"
#include "posix_io/strategy/strategy.h"

#include "streamer/impl/device_io/device_writer/device_writer.h"
#include "streamer/impl/async_io/async_io_worker/async_io_worker.h"
#include "streamer/impl/config/fs_queue_depth/fs_queue_depth.h"
#include "streamer/impl/request/request.h"
#include "streamer/impl/strategy_resolver/strategy_resolver.h"
#include "streamer/impl/workload/workload.h"

#include "utils/threadpool/threadpool.h"

namespace runai::llm::streamer::impl
{

// Which reader serves each file of a filesystem submission, and on which engine.
class FsAsyncRouter
{
 public:
    // Must throw when a path cannot be read, as MountCapabilities::of_path does.
    using MountProbe = std::function<posix_io::MountCapability(const std::string &)>;

    // 0 means the mount serves no direct reads.
    using DirectProbe = std::function<posix_io::DirectSupport(dev_t, const std::string &)>;
    using DirectBlockProbe = std::function<size_t(dev_t, const std::string &)>;

    // Test seams: empty in production, and empty means ask the machine.
    struct Environment
    {
        MountProbe mount;
        DirectProbe direct;
        DirectBlockProbe direct_block;
        AsyncIoWorker::EngineFactory engine;
        StrategyResolver::Availability availability;
    };

    FsAsyncRouter(const std::string & candidates, Environment environment);

    common::ResponseCode set_candidates(const std::string & candidates);
    common::ResponseCode resolve();
    posix_io::Strategy strategy() const;

    // Dense with each other: the group serving each file (-1 for the synchronous reader), and each
    // group's device, direct-I/O block and queue depth.
    struct Groups
    {
        std::vector<int> by_file;
        std::vector<dev_t> devices;
        std::vector<size_t> blocks;
        std::vector<unsigned> depths;

        // Whether each group's reads may use registered buffers. Per mount, because it pays on some
        // file systems and costs on others.
        std::vector<bool> registers;
    };

    // Answers, for a file system type, whether its reads may use registered buffers. A PREDICATE
    // rather than the list, so the matching rule lives in Config with its tests.
    using RegisterPolicy = std::function<bool(const std::string & fs_type)>;

    // One group per MOUNT, and a workload goes to one group. Groups are not engines: several mounts
    // can share an engine once the engine cap is reached (BackendPools), which is why every per-mount
    // answer is carried here rather than assumed to be the engine's.
    //
    // Stats once per directory, not once per file. Never fails a submission - an unreadable directory
    // sends its file to the synchronous reader.
    //
    // A null `registers` means every mount registers, which is what a caller that does not care wants.
    Groups groups(const std::vector<FileRanges> & request, const FsQueueDepth & depth,
                  const RegisterPolicy & registers = nullptr);

    // Captures the shared state by value and never `this`, so the workers it builds outlive this
    // object whatever the destruction order.
    using WorkerFactory = std::function<std::unique_ptr<utils::Worker<Workload>>(dev_t, size_t, unsigned, bool)>;

    // `writer` is the streamer's copy path onto a device, shared by every worker it builds. Captured
    // by value like everything else here, so a worker outlives this object whatever the order.
    WorkerFactory worker_factory(std::shared_ptr<DeviceWriter> writer) const;

    // The largest direct-I/O block any of these mounts requires. `paths` must exclude object-storage
    // URIs, which name no mount.
    //
    // UnknownError when nothing could be measured, and out_block is then the host page size so the
    // caller can still lay out its buffers.
    common::ResponseCode direct_block_for(const std::vector<std::string> & paths, size_t & out_block);

    AsyncIoCounters counters() const;

 private:
    // Asked only for libaio, which is asynchronous only with O_DIRECT: a file it cannot read directly
    // is read inline, one at a time, so it belongs on the synchronous reader. The choice must be made
    // here - once a workload is on the async pool the worker cannot hand it back.
    bool reads_directly(const FileRanges & file, dev_t device);

    size_t block_of(dev_t device, const std::string & path);

    // A dead engine belongs to ONE mount and its files are still readable, so groups() demotes that
    // mount to the synchronous reader. Written by a worker thread, read by whichever thread submits.
    class DeadMounts
    {
     public:
        void add(dev_t device);
        bool contains(dev_t device) const;

     private:
        mutable std::mutex _mutex;
        std::set<dev_t> _devices;
    };

    // shared_ptr because worker_factory() captures them by value.
    std::shared_ptr<StrategyResolver> _strategy_resolver;
    std::shared_ptr<AsyncIoWorkers> _workers = std::make_shared<AsyncIoWorkers>();
    std::shared_ptr<DeadMounts> _dead_mounts = std::make_shared<DeadMounts>();

    posix_io::MountCapabilities _mounts;
    Environment _environment;
};

}; // namespace runai::llm::streamer::impl
