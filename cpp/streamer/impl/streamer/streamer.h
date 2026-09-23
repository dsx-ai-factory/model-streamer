
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils/threadpool/threadpool.h"
#include "utils/fdlimit/fdlimit.h"

#include "common/device/device.h"
#include "common/responder/responder.h"
#include "common/s3_credentials/s3_credentials.h"
#include "streamer/impl/config/config/config.h"
#include "streamer/impl/workload/workload.h"
#include "streamer/impl/s3/s3.h"
#include "streamer/impl/batches/batches.h"
#include "streamer/impl/request/request.h"
#include "streamer/impl/submissions/submissions_mgr.h"
#include "streamer/impl/device_io/device_writer/device_writer.h"
#include "streamer/impl/pools/backend_pools.h"
#include "posix_io/mount_capabilities/mount_capabilities.h"
#include "streamer/impl/async_io/async_io_stats/async_io_stats.h"
#include "streamer/impl/async_io/async_io_worker/async_io_worker.h"
#include "streamer/impl/async_io/fs_async_router/fs_async_router.h"
#include "streamer/impl/strategy_resolver/strategy_resolver.h"

namespace runai::llm::streamer::impl
{

// Streamer for reading large files concurrently

// The user-facing responder is PERSISTENT and lives for the streamer's lifetime; many
// submissions share it, demuxed by submission_id. response(timeout, done) is the single consumer: it
// blocks / times out (no finish-on-drain) and reports completion per submission via submission_done.

// Synchronous read -  read a range of a file to a given buffer of host memory
// Asynchronous read - read a range of a file to a given buffer of host memory in two stages:
//                          1. request to read a range, specifying a list of sub ranges
//                          2. wait for a response for the next ready sub range
//                     Responses are returned without any promissed order - a response is returned when a sub range is completed

struct Streamer
{
    using MountProbe = FsAsyncRouter::MountProbe;
    using DirectProbe = FsAsyncRouter::DirectProbe;
    using DirectBlockProbe = FsAsyncRouter::DirectBlockProbe;
    using Environment = FsAsyncRouter::Environment;

    Streamer();
    explicit Streamer(Config config, Environment environment = {});
    ~Streamer();

    // Set the streamer's object-storage credentials 
    // Set-once and thread-safe
    common::ResponseCode set_credentials(const common::s3::Credentials & credentials);

    // Submit a read request: a list of files, each with the ranges to read from it. A range is an
    // arbitrary (offset, size) within its file with its own destination - ranges need not be contiguous
    // in the file, contiguous in memory, or ordered.
    // Exactly one response is issued per range, including for a zero-sized range (which is completed
    // immediately without reaching storage). A file with no ranges contributes no responses.
    // `device` is where every destination of this submission lives - one device per submission, so a
    // load spanning several GPUs submits once per GPU.
    common::ResponseCode async_request(
      std::vector<FileRanges> & request,
      common::Device device,
      SubmissionId * out_submission_id = nullptr);

    // Consume the next ready sub-range response over the persistent responder. Blocks up to timeout_ms
    // (0 = indefinitely) and returns TimedOut on expiry; FinishedError only on teardown (stop) - there is no
    // finish-on-drain. On a real response it consumes the owning submission's registry record and sets
    // submission_done to true iff it was that submission's last response (see consume_submission_response).
    // The response carries the submission_id.
    common::Response response(unsigned timeout_ms, bool & submission_done);

 
    // Set the filesystem strategy candidates. Set-once, and rejected once resolution has happened -
    common::ResponseCode set_fs_strategy(const std::string & candidates);

    // Valid only after a FILESYSTEM submission: an object-storage one never resolves a strategy.
    posix_io::Strategy fs_strategy() const;

    // Whether any workload was actually routed to the async pool.
    bool async_pool_used() const;

    // How many async engines exist
    unsigned async_engines() const;

    const AsyncIoStats & stats() const;

    // What the async workers have done, summed over all of them and over the streamer's whole life.
    AsyncIoCounters async_counters() const;

    // For testing only. Credentials are streamer-scoped: call set_credentials first (these use whatever
    // was set there).

    common::ResponseCode sync_read(const std::string & path, size_t offset, size_t bytesize, void * dst);

    common::ResponseCode async_read(const std::string & path, size_t offset, size_t bytesize, void * dst, unsigned num_sizes, size_t * internal_sizes);

    // List files under prefix, which may be an object storage URI or a local filesystem path.
    // Applies fnmatch allow/ignore filtering (empty vectors mean no filter) and returns
    // (full path, size) pairs. 
    std::vector<std::pair<std::string, size_t>> list_files(
      const std::string & prefix,
      bool is_recursive,
      const std::vector<std::string> & allow_patterns,
      const std::vector<std::string> & ignore_patterns);

    // The block a caller must lay destinations out at for THESE paths: the largest any of their
    // mounts requires.
    //
    // Success        out_block is measured. FileAccessError from a mount that refuses O_DIRECT
    //                contributes nothing, which is right: it imposes no padding requirement.
    // UnknownError   nothing could be measured. out_block is the host page size - a layout value, so
    //                the caller can still place its buffers - and the caller should ask again next
    //                submission rather than treat it as final.
    common::ResponseCode direct_block_for(const std::vector<std::string> & paths, size_t & out_block);

 private:
    // Returns nullptr for a filesystem path
    std::shared_ptr<common::s3::StorageUri> try_parse_uri(const std::string & path);

    // Reject a submission that mixes backends, and lock the streamer to a single object-storage plugin
    // (first object-storage submission wins). A submission must be either wholly filesystem or wholly one
    // object-storage plugin; the STREAMER may serve both kinds across different submissions (BackendPools
    // keeps one pool per kind). Returns UnsupportedBackendMix when a submission mixes filesystem with
    // object storage, mixes two object-storage plugins, or uses a plugin differing from the lock; else
    // Success.
    common::ResponseCode lock_object_plugin(const std::vector<FileRanges> & request);

 private:
    // Whether this submission reads object storage. The first file WITH RANGES decides, as everywhere
    // else: lock_object_plugin has already rejected a submission that mixes the two, so it is
    // homogeneous by the time anything asks.
    bool is_object_storage_submission(const std::vector<FileRanges> & request);
    // Build the object-storage params for a batch. Credentials are NOT included here (they are read only at
    // client creation, from credentials()); the batch params carry the URI, which is all the per-read path uses.
    common::s3::S3ClientWrapper::Params handle_s3(unsigned file_index, const std::string & path);

    // The streamer's credentials (empty Credentials if none set), read via _credentials_state (which locks
    // its own mutex). Called only at client-creation points (the worker's first client build, and list_files)
    // - never on the per-request path.
    common::s3::Credentials credentials() const;
    void verify_requests(std::vector<FileRanges> & request);

    // Account for one consumed response of submission_id (delegates to _submissions): on the
    // submission's last response, log per-submission throughput. Returns true iff it was the
    // submission's last response (i.e. submission_done).
    bool consume_submission_response(SubmissionId submission_id);

    // Fail workloads[from .. end] so the consumer does not hang when dispatch throws. `from` is the
    // index the loop threw on; the workloads before it were already moved into the pool.
    void drain_undispatched(SubmissionId submission_id, std::vector<Workload> & workloads, size_t from);

 private:
    std::shared_ptr<const Config> _config;

    // Streamer-scoped object-storage credentials with their own (encapsulated) mutex. Set-once: the first
    // set wins; setting the same value again succeeds; a different value is rejected (CredentialsAlreadySet).
    // Held via shared_ptr so the object-storage workers' credentials provider (which reads them at
    // client-creation time) keeps the state alive regardless of Streamer/worker destruction order. Declared
    // BEFORE _pools so it exists when the pool factory captures it.
    class CredentialsState
    {
     public:
        // Store the credentials (first call), or verify they match a previously-stored set. Returns Success
        // if stored or unchanged; CredentialsAlreadySet if a different value was already set.
        common::ResponseCode set(const common::s3::Credentials & credentials);
        // The stored credentials (empty Credentials if none set yet).
        common::s3::Credentials get() const;

     private:
        mutable std::mutex _mutex;
        std::optional<common::s3::Credentials> _credentials;
    };
    std::shared_ptr<CredentialsState> _credentials_state = std::make_shared<CredentialsState>();

    // Declared BEFORE _pools, and that order is load-bearing: the async pool's factory is built from
    // worker_factory() while _pools is constructed.
    FsAsyncRouter _router;

    // The copy path onto a device, shared by every worker of this streamer - one stream per device
    // however many workers copy to it. Costs nothing until a submission actually names a device, so
    // it is created unconditionally. Also BEFORE _pools, for the same reason as _router.
    std::shared_ptr<DeviceWriter> _device_writer;

    AsyncIoStats _stats;

    std::unique_ptr<S3Cleanup> _s3;
    // Lazily-created worker pools, one per backend kind. Occupies the slot the single ThreadPool used
    // to, so object-storage workers still join between _s3_stop (S3Stop) and _s3 (S3Cleanup) on teardown.
    BackendPools _pools;
    std::unique_ptr<S3Stop> _s3_stop;
    std::unique_ptr<utils::FdLimitSetter> _fd_limit;
    std::shared_ptr<common::Responder> _responder;

    // Lazy S3 init, each part exactly once in the streamer's lifetime and only for s3 paths.
    // Split because _s3 is shared by list_files and streaming, while fd limit / stop are
    // streaming-only. std::call_once is thread-safe for concurrent submitters and retries if the
    // callable throws (InsufficientFdLimit), so the error resurfaces on the next s3 submission.
    std::once_flag _s3_stream_init_flag;   // fd limit + S3Stop (streaming only)
    std::once_flag _s3_cleanup_init_flag;  // S3Cleanup (list_files and streaming)

    // per-submission bookkeeping (id allocation + completion + throughput); owns its own mutex
    SubmissionsMgr _submissions;
};

}; // namespace runai::llm::streamer::impl
