#include "streamer/impl/async_io/fs_async_router/fs_async_router.h"

#include <unistd.h>
#include <sys/sysmacros.h>   // major/minor

#include <algorithm>
#include <map>
#include <utility>

#include "common/exception/exception.h"
#include "posix_io/alignment/alignment.h"
#include "posix_io/engine_factory/engine_factory.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

void FsAsyncRouter::DeadMounts::add(dev_t device)
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    _devices.insert(device);
}

bool FsAsyncRouter::DeadMounts::contains(dev_t device) const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _devices.count(device) != 0;
}

FsAsyncRouter::FsAsyncRouter(const std::string & candidates, Environment environment) :
    // Resolved on the first submission, so RUNAI_STREAMER_FS_STRATEGY is only the DEFAULT and
    // anything set before that request still takes effect.
    _strategy_resolver(std::make_shared<StrategyResolver>(candidates, environment.availability)),
    _environment(std::move(environment))
{}

common::ResponseCode FsAsyncRouter::set_candidates(const std::string & candidates)
{
    return _strategy_resolver->set_candidates(candidates);
}

common::ResponseCode FsAsyncRouter::resolve()
{
    return _strategy_resolver->resolve();
}

posix_io::Strategy FsAsyncRouter::strategy() const
{
    return _strategy_resolver->resolved();
}

AsyncIoCounters FsAsyncRouter::counters() const
{
    return _workers->total();
}

FsAsyncRouter::WorkerFactory FsAsyncRouter::worker_factory(std::shared_ptr<DeviceWriter> writer) const
{
    // Reading the strategy inside is safe: the factory runs when the pool is created, which is the
    // first push, which is after resolution.
    return [resolver = _strategy_resolver, workers = _workers, dead = _dead_mounts,
            engine = _environment.engine, writer = std::move(writer)]
           (dev_t device, size_t block, unsigned depth, bool register_buffers) -> std::unique_ptr<utils::Worker<Workload>>
    {
        auto worker = std::make_unique<AsyncIoWorker>(resolver->resolved(), block, depth,
                                                      engine ? engine : posix_io::make_io_engine,
                                                      [dead, device]() { dead->add(device); },
                                                      writer, register_buffers);

        // Registered here, the last point at which the concrete type is still known: the pool stores
        // it as a Worker<Workload>, which knows nothing of counters.
        workers->add(worker.get());
        return worker;
    };
}

size_t FsAsyncRouter::block_of(dev_t device, const std::string & path)
{
    if (_environment.direct_block)
    {
        return _environment.direct_block(device, path);
    }

    // The code is ignored on purpose: a mount that refuses O_DIRECT and one that could not be probed
    // both leave the block at 0, and both mean "no measurement to carry". What must NOT happen is
    // caching the failure, and direct_block already does not.
    size_t block = 0;
    (void)_mounts.direct_block(device, path, block);
    return block;
}

namespace
{

// The DIRECTORY, not the file: capability belongs to the mount, so the answer is the same for every
// shard beside it, and it works for a file that does not exist yet.
std::string directory_of(const std::string & path)
{
    const auto slash = path.find_last_of('/');
    return (slash == std::string::npos) ? std::string(".")
         : (slash == 0 ? std::string("/") : path.substr(0, slash));
}

} // namespace

FsAsyncRouter::Groups FsAsyncRouter::groups(const std::vector<FileRanges> & request,
                                            const FsQueueDepth & depth,
                                            const RegisterPolicy & registers)
{
    Groups out;
    out.by_file.assign(request.size(), -1);

    if (!posix_io::is_async(_strategy_resolver->resolved()))
    {
        return out;   // the synchronous reader serves everything
    }

    // Directory -> its group. MountCapabilities caches by st_dev, which saves the statfs but NOT the
    // stat that finds st_dev - so without this a 200-shard model would stat its directory 200 times.
    std::map<std::string, int> by_directory;

    // st_dev -> group id, so two directories on the same mount share a group and therefore an engine.
    std::map<dev_t, int> by_device;

    const auto group_of_directory = [&](const std::string & directory, const std::string & path)
    {
        const auto seen = by_directory.find(directory);
        if (seen != by_directory.end())
        {
            return seen->second;
        }

        int group = -1;
        try
        {
            const auto capability = _environment.mount ? _environment.mount(directory)
                                                       : _mounts.of_path(directory);

            // Tested before memory_backed and before any probe, because both cost syscalls to answer
            // a question already settled. Left at -1 rather than returned early, so the answer still
            // lands in by_directory below.
            if (!_dead_mounts->contains(capability.dev))
            {
                // tmpfs and ramfs are pure memcpy with no device to overlap, so depth buys nothing
                // and parallelism does (design 5.12).
                if (!capability.memory_backed)
                {
                    // Numbered in first-seen order, which is what makes them dense indices.
                    const auto device = by_device.find(capability.dev);
                    if (device != by_device.end())
                    {
                        group = device->second;
                    }
                    else
                    {
                        group = static_cast<int>(out.devices.size());
                        by_device.emplace(capability.dev, group);
                        out.devices.push_back(capability.dev);

                        // 0 until a file on this mount answers the probe - see the loop below.
                        out.blocks.push_back(0);

                        // Resolved here because this is the one place the file system type is known.
                        const auto resolved = depth.for_type(capability.fs_type);
                        out.depths.push_back(resolved);

                        const bool may_register = registers == nullptr || registers(capability.fs_type);
                        out.registers.push_back(may_register);

                        LOG(DEBUG) << "Mount " << major(capability.dev) << ":" << minor(capability.dev)
                                   << " is " << capability.fs_type << " and reads at a queue depth of "
                                   << resolved << (may_register ? " with" : " without")
                                   << " registered buffers";
                    }
                }
            }
        }
        catch (const common::Exception & e)
        {
            // Deliberately NOT fatal: failing here would turn a per-file problem into a
            // whole-submission one, which the missing file itself will report attributably.
            LOG(WARNING) << "Cannot probe the mount of " << directory << " (" << e.error()
                         << "); reading " << path << " with the synchronous reader";
        }

        // Remembered whatever the answer, so an unprobeable directory is not retried once per file.
        by_directory.emplace(directory, group);
        return group;
    };

    // libaio has no asynchronous buffered mode, so a file it cannot read directly must be routed away
    // before dispatch. Every other async strategy keeps its files - see reads_directly().
    const bool check_direct = _strategy_resolver->resolved() == posix_io::Strategy::LibaioDirect;

    for (size_t i = 0; i < request.size(); ++i)
    {
        const auto & path = request[i].path;

        // A file with no ranges reaches no storage, so probing its mount would fail on a path that
        // was never going to be read.
        if (request[i].ranges.empty())
        {
            continue;
        }

        const int group = group_of_directory(directory_of(path), path);
        if (group < 0)
        {
            continue;   // the synchronous reader serves it
        }

        // From the first file that ANSWERS, not the first file: a path that does not exist yet leaves
        // the block at 0, so asking only the first would freeze this mount's engine on a fallback
        // because of one missing shard.
        //
        // Measured for every direct strategy, not only libaio - io_uring needs the number just as
        // much, and has no other chance to learn it before its engine is built.
        if (out.blocks[group] == 0 && posix_io::is_direct(_strategy_resolver->resolved()))
        {
            out.blocks[group] = block_of(out.devices[group], path);
        }

        // PER FILE, so it cannot be answered from the directory cache above: congruence depends on
        // this file's own offsets and destinations, and two shards in one directory can differ.
        if (check_direct && !reads_directly(request[i], out.devices[group]))
        {
            continue;
        }

        out.by_file[i] = group;
    }

    return out;
}

bool FsAsyncRouter::reads_directly(const FileRanges & file, dev_t device)
{
    // THIS MOUNT's block, not the process-wide assumption: testing against a larger block rejects
    // files the mount would have served. On a mount that accepts 512, testing at 65536 makes
    // congruence 128 times harder.
    const auto measured = block_of(device, file.path);
    const auto block = measured != 0 ? measured : posix_io::direct_block_size();

    for (const auto & range : file.ranges)
    {
        // A zero-sized range is never read, so letting one decide the reader for the whole file would
        // be deciding on a range nobody reads.
        if (range.size == 0)
        {
            continue;
        }

        if (!posix_io::is_congruent(range.offset, range.dst, block))
        {
            // No PART of this range can be read directly, so the worker would open the file buffered.
            LOG(DEBUG) << "Reading " << file.path << " with the synchronous reader: offset "
                       << range.offset << " and its destination are not congruent for block " << block;
            return false;
        }
    }

    const auto support = _environment.direct ? _environment.direct(device, file.path)
                                             : _mounts.direct_support(device, file.path);

    // Unknown counts as yes: it means the probe could not open the file at all, which the read is
    // about to report properly, and refusing the mount would move every file beside it too.
    if (support == posix_io::DirectSupport::No)
    {
        LOG(DEBUG) << "Reading " << file.path << " with the synchronous reader: its mount cannot"
                   << " serve O_DIRECT, and libaio without O_DIRECT reads one file at a time";
        return false;
    }

    return true;
}

common::ResponseCode FsAsyncRouter::direct_block_for(const std::vector<std::string> & paths,
                                                     size_t & out_block)
{
    out_block = 0;

    size_t largest = 0;
    bool measured = false;

    for (const auto & path : paths)
    {
        try
        {
            const auto directory = directory_of(path);
            const auto capability = _environment.mount ? _environment.mount(directory)
                                                       : _mounts.of_path(directory);

            // A memory-backed mount has no device to bypass, so it imposes no alignment at all.
            if (capability.memory_backed)
            {
                continue;
            }

            // Non-zero is the answer, whichever side produced it: the injected probe reports 0 for
            // "serves no direct reads" and has no response code, and the real one only reports
            // Success with a block above 0.
            const auto block = block_of(capability.dev, path);
            if (block != 0)
            {
                largest = std::max(largest, block);
                measured = true;
            }
        }
        catch (const common::Exception &)
        {
            // An unstattable directory tells us nothing about alignment; another path may answer.
            continue;
        }
    }

    if (!measured)
    {
        // A LAYOUT value, so the caller can still place its buffers. UnknownError, not Success, so it
        // knows to ask again next submission instead of treating this as the mount's answer.
        out_block = static_cast<size_t>(::sysconf(_SC_PAGESIZE));

        LOG(INFO) << "No filesystem mount among " << paths.size() << " path(s) could be measured;"
                  << " laying out at the host page size of " << out_block << " bytes for now";
        return common::ResponseCode::UnknownError;
    }

    out_block = largest;

    LOG(INFO) << "Destinations for these " << paths.size() << " path(s) must be laid out at "
              << out_block << " bytes - the largest any of their mounts requires";
    return common::ResponseCode::Success;
}

}; // namespace runai::llm::streamer::impl
