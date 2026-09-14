#include "streamer/impl/config/config/config.h"

#include "posix_io/alignment/alignment.h"

#include <utility>

#include "common/s3_wrapper/s3_wrapper.h"

#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

namespace
{

// Object storage: the specific variable when set, then the legacy one, then the default. Resolved on
// PRESENCE - getenv with a default cannot tell "unset" from "set to the default", so the legacy
// variable could never take precedence over a default of its own.
unsigned resolve_obj_concurrency()
{
    unsigned long configured = 0;
    if (utils::try_getenv("RUNAI_STREAMER_OBJ_CONCURRENCY", configured))
    {
        return Config::to_concurrency(configured, "RUNAI_STREAMER_OBJ_CONCURRENCY");
    }

    return Config::to_concurrency(utils::getenv<unsigned long>("RUNAI_STREAMER_CONCURRENCY",
                                                               Config::default_s3_concurrency),
                                  "RUNAI_STREAMER_CONCURRENCY");
}

} // namespace

unsigned Config::to_concurrency(unsigned long value, const char * source)
{
    // Caps BEFORE narrowing. The variables are parsed as 64-bit, so a cast alone would wrap:
    // 4294967301 becomes 5, and 4294967296 becomes 0, which the assertions above then read as a
    // deliberate zero. Both are silently wrong rather than merely too large.
    if (value > max_concurrency)
    {
        LOG(WARNING) << "Value " << value << " from " << source << " is above the limit of "
                     << max_concurrency << " and is capped to it. Each unit costs an OS thread and"
                     << " its own I/O resources";
        return max_concurrency;
    }

    return static_cast<unsigned>(value);
}

Config::Config(unsigned concurrency, unsigned s3_concurrency, size_t s3_block_bytesize, size_t fs_sync_read_block_bytesize,
               bool enforce_minimum, size_t fs_async_chunk_bytesize, FsQueueDepth fs_async_queue_depth,
               std::string fs_strategy_candidates, unsigned long object_storage_retry_timeout_seconds) :
    concurrency(concurrency),
    s3_concurrency(s3_concurrency),
    s3_block_bytesize(s3_block_bytesize),
    fs_sync_read_block_bytesize(fs_sync_read_block_bytesize),
    fs_async_chunk_bytesize(fs_async_chunk_bytesize),
    fs_strategy_candidates(std::move(fs_strategy_candidates)),
    fs_async_queue_depth(std::move(fs_async_queue_depth)),
    object_storage_retry_timeout(object_storage_retry_timeout_seconds)
{
    // Resolved here, with the other configuration, so a malformed RUNAI_STREAMER_DIRECT_BLOCK fails
    // like every other malformed variable: runai_file_streamer_start builds a Config first and turns any failure
    // into InvalidParameterError, while a failure in the Streamer that follows becomes UnknownError.
    //
    // It has to be forced, because the value lives behind a cached static and is otherwise resolved
    // lazily on the first read - so a typo would surface mid-load, as a failed read, with nothing
    // naming the variable that caused it.
    (void)posix_io::direct_block_size();

    ASSERT(concurrency) << " threadpool size must be a positive number";

    // Zero divides the workload between no workers, and asks for a zero file descriptor budget.
    ASSERT(s3_concurrency) << "object storage concurrency must be a positive number";

    // A backstop for the positional constructor, whose caller passes a number directly. The values
    // that come from the environment were already capped where they were narrowed, in
    // resolve_obj_concurrency and resolve_fs_settings.
    this->concurrency = to_concurrency(this->concurrency, "the streamer configuration");
    this->s3_concurrency = to_concurrency(this->s3_concurrency, "the streamer configuration");

    ASSERT(s3_block_bytesize) << "s3 chunk bytesize must be positive";

    // Tasks are cut on multiples of this, so zero would divide by zero rather than merely misbehave.
    ASSERT(fs_async_chunk_bytesize) << "file system chunk bytesize must be positive";
    // parse() rejects a zero; a positional caller can still pass one
    ASSERT(this->fs_async_queue_depth.default_value()) << "file system queue depth must be positive";

    if (enforce_minimum)
    {
        if (s3_block_bytesize < common::s3::S3ClientWrapper::min_chunk_bytesize)
        {
            // For S3 requests the chunk size is the minimal value of 5 MB
            LOG(INFO) << "Setting s3 reading block size to 5 MiB";
            this->s3_block_bytesize = common::s3::S3ClientWrapper::min_chunk_bytesize;
        }

        if (fs_sync_read_block_bytesize < min_fs_sync_read_block_bytesize)
        {
            LOG(INFO) << "Setting file system reading block size to 2 MiB";
            this->fs_sync_read_block_bytesize = min_fs_sync_read_block_bytesize;
        }
    }
}

Config::FsSettings Config::resolve_fs_settings()
{
    std::string configured;
    if (utils::try_getenv("RUNAI_STREAMER_FS_QUEUE_DEPTH", configured))
    {
        auto depth = FsQueueDepth::parse(configured);
        const auto concurrency = depth.default_value();   // per-type entries are for the mounts

        LOG(DEBUG) << "File system settings from RUNAI_STREAMER_FS_QUEUE_DEPTH=" << configured;
        return FsSettings{ std::move(depth), concurrency };
    }

    unsigned long legacy = 0;
    if (utils::try_getenv("RUNAI_STREAMER_CONCURRENCY", legacy))
    {
        const auto concurrency = to_concurrency(legacy, "RUNAI_STREAMER_CONCURRENCY");

        LOG(DEBUG) << "File system settings from RUNAI_STREAMER_CONCURRENCY=" << legacy
                   << ", because RUNAI_STREAMER_FS_QUEUE_DEPTH is unset";
        return FsSettings{ FsQueueDepth(concurrency), concurrency };
    }

    // Apart, because a read costs a thread in one reader and a queue slot in the other.
    return FsSettings{ FsQueueDepth(Config::default_fs_async_queue_depth), Config::default_concurrency };
}

Config::Config(bool enforce_minimum /* = true */) :
    Config(resolve_fs_settings(), enforce_minimum)
{}

Config::Config(FsSettings fs, bool enforce_minimum) :
    Config(fs.concurrency,
           resolve_obj_concurrency(),
           utils::getenv<size_t>("RUNAI_STREAMER_CHUNK_BYTESIZE", common::s3::S3ClientWrapper::default_chunk_bytesize),
           utils::getenv<size_t>("RUNAI_STREAMER_CHUNK_BYTESIZE", min_fs_sync_read_block_bytesize),
           enforce_minimum,
           utils::getenv<size_t>("RUNAI_STREAMER_FS_CHUNK_BYTESIZE", default_fs_async_chunk_bytesize),
           std::move(fs.depth),
           utils::getenv<std::string>("RUNAI_STREAMER_FS_STRATEGY", default_fs_strategy_candidates),
           utils::getenv<unsigned long>("RUNAI_STREAMER_S3_TIMEOUT", 0UL))
{}

std::ostream & operator<<(std::ostream & os, const Config & config)
{
    return os << "Streamer concurrency " << config.concurrency << " ; s3 concurrency " << config.s3_concurrency << " ; s3 block size " << config.s3_block_bytesize << " bytes; " << " ; file system block size " << config.fs_sync_read_block_bytesize << " bytes; " << " ; file system chunk size " << config.fs_async_chunk_bytesize << " bytes; " << " ; file system queue depth " << config.fs_async_queue_depth << " (node-wide); " << " ; file system strategy " << config.fs_strategy_candidates << "; object storage retry timeout " << config.object_storage_retry_timeout.count() << " seconds; ";
}

}; // namespace runai::llm::streamer::impl
