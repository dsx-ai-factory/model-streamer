#include "streamer/impl/pools/backend_pools.h"

#include <algorithm>

#include <sys/sysmacros.h>   // major/minor - moved out of sys/types.h in glibc 2.28

#include <atomic>
#include <utility>

#include "streamer/impl/config/config/config.h"
#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

BackendPools::BackendPools(Handler filesystem_handler,
                           AsyncWorkerFactory filesystem_async_factory,
                           WorkerFactory object_storage_factory,
                           unsigned filesystem_size,
                           unsigned object_storage_size) :
    _filesystem_handler(std::move(filesystem_handler)),
    _filesystem_async_factory(std::move(filesystem_async_factory)),
    _object_storage_factory(std::move(object_storage_factory)),
    _filesystem_size(filesystem_size),
    _object_storage_size(object_storage_size),
    // Per queue depth and per PROCESS, not per node. Each engine has its own queue depth, so N
    // engines mean N times the depth of reads running at the device - and a thread and a ring each,
    // which is why to_concurrency caps it like the two concurrencies, and warns the same way.
    _max_async_engines(Config::to_concurrency(
        utils::getenv_positive<unsigned>("RUNAI_STREAMER_FS_MAX_ENGINES", 1U),
        "RUNAI_STREAMER_FS_MAX_ENGINES"))
{}

void BackendPools::push(Pool pool, Workload && workload)
{
    if (pool == Pool::ObjectStorage)
    {
        // Created by lock_object_plugin (once the plugin is known), which async_request always calls
        // before dispatching object-storage workloads - so no per-workload creation check on this path.
        ASSERT(_object_storage_pool != nullptr) << "object-storage workload dispatched before its plugin was locked";
        _object_storage_pool->push(std::move(workload));
        return;
    }

    ASSERT(pool != Pool::FileSystemAsync)
        << "async filesystem workloads are dispatched by mount - use push_async";

    // filesystem has no plugin to wait for, so create its pool lazily on first use
    std::call_once(_filesystem_once, [this]()
    {
        _filesystem_pool = std::make_unique<utils::ThreadPool<Workload>>(_filesystem_handler, _filesystem_size);
    });
    _filesystem_pool->push(std::move(workload));
}

void BackendPools::push_async(dev_t device, size_t block, unsigned depth, Workload && workload)
{
    utils::ThreadPool<Workload> * pool = nullptr;

    {
        const auto guard = std::unique_lock<std::mutex>(_async_mutex);

        const auto assigned = _async_by_device.find(device);
        if (assigned != _async_by_device.end())
        {
            // Once a mount has an engine, it keeps it. The engine holds the state that routes
            // completions and counts free slots. Moving a mount with reads running would lose both.
            pool = assigned->second;
        }
        else
        {
            // Routed by the depth the mount resolved, so a mount never lands on an engine built for
            // another depth. The cap applies within the depth, which is why discovery order cannot
            // cost a mount its configured value.
            auto & engines = _async_pools[depth];

            if (engines.size() < _max_async_engines)
            {
                // Lazy, like the synchronous pool: a streamer that never reads this mount never
                // builds a ring or a thread for it. Nothing has to be AGREED first, unlike object
                // storage's plugin lock - the strategy was settled before dispatch, and a mount needs
                // no agreement at all. (The mutex above guards the map, not a decision.)
                // The block is bound HERE and never changes for this engine.
                auto created = std::make_unique<utils::ThreadPool<Workload>>(
                    [factory = _filesystem_async_factory, device, block, depth]() { return factory(device, block, depth); }, 1);
                pool = created.get();
                engines.push_back(std::move(created));

                LOG(DEBUG) << "Engine " << engines.size() << " at queue depth " << depth << " serves mount "
                           << major(device) << ":" << minor(device);
            }
            else
            {
                // At the cap for this depth: share the engine with the least work waiting. We do not
                // wait for a free engine, because that would only move the delay to another place.
                pool = least_loaded_async(engines);
                ++_shared_mounts;

                // A warning, not a debug line, and it names the variable. The depth survives, because
                // the engine was built for it; isolation does not. A mount that stops responding now
                // also stops the mounts sharing its engine, and the sharer reads at the engine's
                // direct-I/O block. Nothing else would show this. Once per mount, because the choice
                // is permanent.
                LOG(WARNING) << "Mount " << major(device) << ":" << minor(device) << " shares an engine at"
                             << " queue depth " << depth << ": " << engines.size() << " already in use at"
                             << " that depth, so it reads at that engine's direct-I/O block rather than the "
                             << block << " it resolved. A stall on one of these mounts now stalls the"
                             << " others; raise RUNAI_STREAMER_FS_MAX_ENGINES to separate them";
            }

            _async_by_device.emplace(device, pool);
        }
    }

    pool->push(std::move(workload));
}

utils::ThreadPool<Workload> * BackendPools::least_loaded_async(
    const std::vector<std::unique_ptr<utils::ThreadPool<Workload>>> & engines)
{
    // "Least loaded" means the fewest workloads waiting, not the fewest mounts. A mount that reads
    // nothing gives its engine no work, so counting mounts would say nothing about real load.
    //
    // This is only an estimate. It counts work that no worker has started yet, so an engine that is
    // busy with one very large workload looks free. That is good enough here, because this choice is
    // made only above the cap, where mounts are no longer separated anyway.
    utils::ThreadPool<Workload> * best = nullptr;
    size_t fewest = 0;

    for (const auto & pool : engines)
    {
        const size_t queued = pool->pending();
        if (best == nullptr || queued < fewest)
        {
            best = pool.get();
            fewest = queued;
        }
    }

    ASSERT(best != nullptr) << "sharing an engine with none created";
    return best;
}

common::ResponseCode BackendPools::lock_object_plugin(Plugin plugin)
{
    // Fast path (no mutex): once the lock is fully established - plugin recorded AND pool created, published
    // together via _ready_plugin - a matching submission needs nothing more. The acquire pairs with the
    // release below, so observing _ready_plugin here guarantees the pool exists and is safe to dispatch to.
    // (Credentials are streamer-scoped, set once via runai_file_streamer_set_credentials - not per submission - so there is
    // nothing credential-related to lock here.)
    const int ready = _ready_plugin.load(std::memory_order_acquire);
    if (ready != -1)
    {
        if (ready != static_cast<int>(plugin))
        {
            LOG(ERROR) << "Streamer is locked to a single object storage plugin; rejecting a submission using a different plugin";
            return common::ResponseCode::UnsupportedBackendMix;
        }
        return common::ResponseCode::Success;
    }

    // Slow path: taken only until the lock is initialized (the first object-storage submission, plus any
    // concurrent submitters racing with it).
    const auto guard = std::unique_lock<std::mutex>(_plugin_mutex);

    if (!_plugin.has_value())
    {
        // First object-storage submission: create the pool, THEN commit the lock. Order matters for exception
        // safety - the ThreadPool ctor spawns worker threads and can throw (e.g. thread creation fails under a
        // process/fd/thread ulimit, not only OOM). Building it first means a throw leaves _plugin unset and
        // _ready_plugin at -1 with the pool still null, so the NEXT submission retries this branch cleanly -
        // instead of finding the plugin "locked" with a null pool, which would ASSERT on _pools.push.
        // per-worker pool: each thread owns an ObjectStorageWorker (from the factory) with its own in-flight window
        _object_storage_pool = std::make_unique<utils::ThreadPool<Workload>>(_object_storage_factory, _object_storage_size);
        // Commit the lock only now that the pool exists. _ready_plugin is stored last (release), so a fast-path
        // acquire that observes it is guaranteed the pool is built and safe to dispatch to. The s3_wrapper
        // backend handle is a process-wide static, hence the single-plugin lock.
        _plugin = plugin;
        _ready_plugin.store(static_cast<int>(plugin), std::memory_order_release);
        return common::ResponseCode::Success;
    }

    // Already locked by a concurrent submitter between the fast-path load and here: verify the plugin.
    if (_plugin.value() != plugin)
    {
        LOG(ERROR) << "Streamer is locked to a single object storage plugin; rejecting a submission using a different plugin";
        return common::ResponseCode::UnsupportedBackendMix;
    }

    return common::ResponseCode::Success;
}

bool BackendPools::async_pool_used() const
{
    const auto guard = std::unique_lock<std::mutex>(_async_mutex);
    return !_async_pools.empty();
}

unsigned BackendPools::shared_engine_mounts() const
{
    const auto guard = std::unique_lock<std::mutex>(_async_mutex);
    return _shared_mounts;
}

unsigned BackendPools::async_engines() const
{
    const auto guard = std::unique_lock<std::mutex>(_async_mutex);
    return count_async_engines();
}

unsigned BackendPools::pools_created() const
{
    const auto guard = std::unique_lock<std::mutex>(_async_mutex);

    return (_filesystem_pool != nullptr ? 1u : 0u)
         + count_async_engines()
         + (_object_storage_pool != nullptr ? 1u : 0u);
}

unsigned BackendPools::count_async_engines() const
{
    unsigned total = 0;
    for (const auto & [depth, engines] : _async_pools)
    {
        (void)depth;
        total += static_cast<unsigned>(engines.size());
    }
    return total;
}

}; // namespace runai::llm::streamer::impl
