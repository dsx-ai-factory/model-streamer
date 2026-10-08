
#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <ostream>

#include "streamer/impl/config/fs_queue_depth/fs_queue_depth.h"

namespace runai::llm::streamer::impl
{

// Environment variables, with their defaults when unset (config.cc):
//
//     RUNAI_STREAMER_CONCURRENCY        -> s3_concurrency when OBJ_CONCURRENCY is unset, and the file
//                                          system readers when FS_QUEUE_DEPTH is unset. Legacy, kept
//                                          for compatibility.
//     RUNAI_STREAMER_OBJ_CONCURRENCY    -> s3_concurrency (8): how much object storage work runs at
//                                          once. Object storage only.
//     RUNAI_STREAMER_CHUNK_BYTESIZE     -> fs_sync_read_block_bytesize (2 MiB, also the minimum) AND
//                                          s3_block_bytesize (8 MiB, minimum 5 MiB).
//     RUNAI_STREAMER_FS_CHUNK_BYTESIZE  -> fs_async_chunk_bytesize (8 MiB). File system only.
//     RUNAI_STREAMER_FS_QUEUE_DEPTH     -> fs_async_queue_depth per mount (64) AND concurrency as the
//                                          synchronous pool's threads (16). NODE-WIDE for the mounts,
//                                          divided per process by AsyncIoSettings.
//     RUNAI_STREAMER_FS_STRATEGY        -> fs_strategy_candidates. An ordered preference list; the
//                                          first the host can serve wins.

struct Config
{
    // Everything after `enforce_minimum` is defaulted and comes after the bool, only so existing
    // callers that pass enforce_minimum positionally keep working.
    Config(unsigned concurrency,
           unsigned s3_concurrency,
           size_t s3_block_bytesize,
           size_t fs_sync_read_block_bytesize,
           bool enforce_minimum = true,
           size_t fs_async_chunk_bytesize = default_fs_async_chunk_bytesize,
           FsQueueDepth fs_async_queue_depth = FsQueueDepth(default_fs_async_queue_depth),
           std::string fs_strategy_candidates = default_fs_strategy_candidates,
           unsigned long object_storage_retry_timeout_seconds = 0,
           std::string fs_no_register_buffers = default_fs_no_register_buffers,
           bool fs_register_buffers = default_fs_register_buffers);
    Config(bool enforce_minimum = true);

 private:
    // The deny list as written, into lowercased entries with empties dropped.
    static std::vector<std::string> split_types(const std::string & value);

    // Both file system settings come from the same variable, so they are resolved together: reading it
    // twice would parse it twice and state the precedence rule twice.
    struct FsSettings
    {
        FsQueueDepth depth;
        unsigned concurrency;
    };

    static FsSettings resolve_fs_settings();

    // Only so Config(bool) can resolve once and pass both values on.
    Config(FsSettings fs, bool enforce_minimum);

 public:
    static constexpr size_t min_fs_sync_read_block_bytesize = 2 * 1024 * 1024;

    // No shared floor with the synchronous block size: 2 MiB suits a reader that wants fewer, larger
    // reads, while an async reader with depth wants more, smaller ones.
    static constexpr size_t default_fs_async_chunk_bytesize = 8 * 1024 * 1024;

    // Node-wide, so it means the same thing at TP=1 and TP=8.
    static constexpr unsigned default_fs_async_queue_depth = 64;

    // RUNAI_STREAMER_FS_NO_REGISTER_BUFFERS: file system types whose reads must NOT use registered
    // buffers, comma separated, prefix matched so `nfs` covers `nfs4`. Empty registers everywhere.
    //
    // A DENY LIST rather than a value per type, because that is the shape of what we know: registration
    // helps unless the storage is one we have measured it to hurt. FsQueueDepth is not reused for it -
    // that type rejects zero on purpose ("zero divides by zero downstream"), which is exactly the value
    // a boolean needs.
    //
    // NFS is denied by default because registration COSTS there rather than paying: its data arrives
    // over the network through the client stack, so there is no device DMA into our pages to
    // streamline and registration only adds bookkeeping.
    static constexpr const char * default_fs_no_register_buffers = "nfs";

    // RUNAI_STREAMER_FS_REGISTER_BUFFERS: the master switch, on by default.
    //
    // Separate from the deny list because they answer different questions. The list says WHICH file
    // systems it is worth doing for; this says whether to do it at all - the switch a benchmark flips
    // to measure the feature, and the one an operator flips if it ever misbehaves in the field.
    //
    // Default ON because it cannot cost correctness: every failure path falls back to an ordinary read.
    // What it is WORTH is a CPU saving rather than a bandwidth one, so a throughput benchmark will not
    // show it.
    static constexpr bool default_fs_register_buffers = true;

    // 32 times smaller than the depth above, because here a concurrent read costs an OS thread rather
    // than a queue slot.
    static constexpr unsigned default_concurrency = 16;

    static constexpr unsigned default_s3_concurrency = 8;

    // A ceiling for every count of workers - the two concurrencies and the engines per queue depth.
    // A negative value is already rejected when the variable is parsed, but a large positive one is
    // not, and each unit costs an OS thread and, for object storage, a client with its own connection
    // pool and file descriptors. Clamped rather than rejected, with a warning: a number too big is a
    // misunderstanding, not a typo, and the load should still run.
    static constexpr unsigned max_concurrency = 1024;

    // Caps a worker count at max_concurrency and narrows it, naming the variable it came from. Takes
    // the wide type the variables are parsed as, because capping after the narrowing is too late.
    static unsigned to_concurrency(unsigned long value, const char * source);

    static constexpr const char * default_fs_strategy_candidates = "io_uring_direct,libaio_direct,sync_buffered";

    // Threads in the synchronous file system pool.
    unsigned concurrency;

    // How much object storage work runs at once. Named for the capacity, not for clients, so it stays
    // accurate if an implementation stops using one client per unit.
    unsigned s3_concurrency;
    size_t s3_block_bytesize;
    size_t fs_sync_read_block_bytesize;

    // Also where tasks are cut: a task never crosses a chunk boundary, so a completed chunk always
    // covers a whole number of tasks. Cut on both paths, to keep one rule rather than two.
    size_t fs_async_chunk_bytesize;

    // An ordered preference list, best first, parsed by parse_candidates. Only the DEFAULT - the
    // caller may override it until the first submission resolves it (StrategyResolver).
    std::string fs_strategy_candidates;

    // In-flight requests for the whole node, per file system type. What one process may hold is this
    // divided by the streamer processes on the node, which is not known this early - AsyncIoSettings
    // does it.
    FsQueueDepth fs_async_queue_depth;

    // The types from RUNAI_STREAMER_FS_NO_REGISTER_BUFFERS, lowercased, in the order written.
    std::vector<std::string> fs_no_register_buffers;

    // The master switch. False turns registered buffers off for every mount, whatever the list says.
    bool fs_register_buffers;

    // Whether reads on a mount of this type may use registered buffers. Prefix matched, so `nfs`
    // covers `nfs4`. An unknown or empty type registers - we deny what we have measured, not what we
    // have not seen.
    bool registers_buffers(const std::string & fs_type) const;

    // Application-level retry budget for each object chunk, starting when that chunk is first submitted to
    // the backend. Zero preserves fail-fast behavior after the storage plugin's native retry policy expires.
    std::chrono::seconds object_storage_retry_timeout;
};

std::ostream & operator<<(std::ostream &, const Config &);

}; // namespace runai::llm::streamer::impl
