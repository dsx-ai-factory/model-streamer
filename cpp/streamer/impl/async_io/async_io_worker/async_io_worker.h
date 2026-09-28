#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "posix_io/engine_factory/engine_factory.h"
#include "posix_io/io_engine/io_engine.h"
#include "posix_io/scratch_pool/scratch_pool.h"
#include "posix_io/strategy/strategy.h"
#include "common/response_code/response_code.h"

#include "streamer/impl/async_io/async_io_settings/async_io_settings.h"
#include "streamer/impl/async_io/async_io_stats/async_io_stats.h"
#include "streamer/impl/async_io/inflight_chunks/inflight_chunks.h"
#include "streamer/impl/batch/batch.h"
#include "streamer/impl/workload/workload.h"
#include "utils/capacity_worker/capacity_worker.h"

namespace runai::llm::streamer::impl
{

// One chunk waiting in the window, by value.
//
// Deliberately NOT a pointer into Batch::chunks: entries sit in the queue while their workload can be
// erased underneath them (abort_all on stop, a drained responder, an OOM mid-registration), so a
// pointer would be a use-after-free on exactly the paths hardest to exercise. A stale value is merely
// useless and trips an assert. The copy is three fields per in-flight request - about 20 KB at depth
// 512.
//
// Same shape as ObjectStorageWorker's ObjectChunk, so the two can share a base later.
struct QueuedChunk
{
    posix_io::RequestId id = 0;
    size_t offset = 0;
    size_t bytesize = 0;
    char * buffer = nullptr;
};

// The single worker driving one IoEngine.
//
// A CapacityWorker, so the base owns the in-flight window and the submit/drain interleave and this
// only fills the backend hooks - the same arrangement ObjectStorageWorker uses, deliberately, so the
// completion routing can be lifted into a shared base once both exist.
//
// ONE worker owns ONE engine and makes every call on it (5.2), which is what keeps this free of locks.
// Under one engine per mount (5.2.3) there are several of these, one per mount, each in its own
// single-worker pool - so nothing here may assume it is the only instance.
class AsyncIoWorker : public utils::CapacityWorker<Workload, QueuedChunk>
{
 public:
    // Builds the engine for a strategy and its resolved configuration, or returns nullptr if this host
    // cannot provide it.
    //
    // Injectable so tests can drive the worker with MockIoEngine - the same shape as
    // ObjectStorageWorker's credentials provider, and for the same reason. Production passes nothing.
    using EngineFactory = std::function<std::unique_ptr<posix_io::IoEngine>(
                              posix_io::Strategy, const posix_io::AsyncIoConfig &)>;

    // `block` is the mount's measured direct-I/O block, or 0 when no file on it could be probed -
    // in which case the worker runs provisionally and adopts a later workload's measurement.
    // `on_engine_dead` is called once, from this worker's thread, when the engine fails for good. The
    // streamer uses it to route this mount to the synchronous reader from then on - the files are
    // still readable, it is only the ring that is gone. Empty in tests that do not care.
    // `node_wide_depth` is RUNAI_STREAMER_FS_QUEUE_DEPTH resolved for this mount's file system type.
    explicit AsyncIoWorker(posix_io::Strategy strategy,
                           size_t block = 0,
                           unsigned node_wide_depth = Config::default_fs_async_queue_depth,
                           EngineFactory factory = posix_io::make_io_engine,
                           std::function<void()> on_engine_dead = {});
    ~AsyncIoWorker() override;

    // Bytes copied out of a scratch buffer, over this worker's life.
    //
    // THIS is the cost of direct reads, not the number of passes: a pass that copies 24 bytes and one
    // that copies 4096 are not the same thing. The bound that matters is per region - at most one
    // partial block at each end, so at most 2 x block whatever the region's size. That is what makes
    // the cost about 0.1% of an 8 MiB region rather than a share of every chunk.
    //
    // Expected to be a small constant per region on the CPU pool path, and ZERO once destinations are
    // placed congruently with room to spare. A number that grows with the bytes read means the
    // placement has stopped working and every read is being copied.
    size_t bounced_bytes() const;

    // Every byte delivered to a caller by this worker. Only useful beside bounced_bytes(): the ratio
    // is what says whether direct reads are working.
    size_t bytes_read() const;

    // This worker's numbers, for the streamer to sum with the other workers'.
    //
    // Safe to call from ANY thread, which is the reason the counters below are atomic. Everything
    // else in this class is touched only by the worker's own thread; these are the exception, because
    // the streamer reads them while the worker is still running.
    //
    // A snapshot rather than a live view: the four values are read one at a time, so they can come
    // from slightly different moments. That is fine for what they are for - a ratio and a high-water
    // mark - and a lock would put contention on the read path for no gain.
    AsyncIoCounters counters() const;

 protected:
    // Builds the engine and returns the window size, on the FIRST workload.
    //
    // Here because CapacityWorker asks for the window size here, and the window size IS the engine's
    // depth - so one place decides both, and they cannot disagree.
    //
    // The timing constraint is weaker than it looks, and worth stating so nobody moves this on a
    // wrong premise: depth is divided by RUNAI_STREAMER_PROCESS_GROUP_SIZE, which Python writes in
    // stream_files() (distributed_streamer.py:154) just before the first runai_file_streamer_request. That rules
    // out building at STREAMER construction - runai_file_streamer_start() returns before stream_files() runs, so it
    // would read the unset default of 1, skip the division, and give the device the full depth per
    // process. It does NOT rule out building at this worker's construction: the async pool is created
    // lazily on first push, which is already inside that first request.
    //
    // The returned capacity IS the in-flight bound, and the only one: ids are monotonic rather than
    // indices into a fixed table (5.3), so nothing else can disagree about how many may be outstanding.
    //
    // THROWS if no engine can be built, so the base discards this workload rather than dropping it.
    // Unlike object storage - whose client can fail transiently on credentials or a missing plugin -
    // failure here is PERMANENT: a blocked io_uring stays blocked. So the retry the base offers buys
    // nothing, and reaching this at all means the dispatcher created an async pool for a host that
    // cannot serve one. Logged as an invariant violation, not as a condition to recover from.
    std::size_t capacity(const Workload & first) override;

    // The window could not come up. Finalize the workload anyway - its responses are already owed, and
    // dropping it hangs the consumer forever.
    void discard(Workload && workload) override;

    // Register the workload and queue its chunks. Zero-sized tasks are completed here: they appear in
    // no chunk (chunk_splitter.h), so nothing will ever complete for them, and they still owe a
    // response each.
    void enqueue(Workload && workload) override;

    // Stage one chunk. Staging is not issuing - flush() does that - so this must not block.
    void submit(const QueuedChunk & chunk) override;

    // Flush the backlog, harvest with a bounded wait, and route what arrived. On `stopped`, walk BOTH
    // the in-flight set and the staged-but-unissued set: the latter hold window credit and will never
    // produce a completion, so omitting them leaves tasks unfinalized.
    void drain_batch(std::atomic<bool> & stopped) override;

 private:
    // Per workload, kept until its last task is accounted.
    // A batch's descriptor, opened by the first chunk of it that actually issues.
    //
    // Lazy in, bulk out: nothing is opened at enqueue - a stopped submission does none at all, and a
    // zero-size batch has no chunks so is never opened - and everything is closed in one place when
    // the workload ends. A workload is the caller's statement that these files are read in parallel,
    // so holding a descriptor for as long as its workload lives is the honest lifetime.
    //
    // `error` remembers a failed open so the next chunk of the same batch fails immediately instead of
    // retrying the open, and logging, once per chunk.
    struct BatchFd
    {
        int fd = -1;

        // What the file WAS opened as, not what we wanted. A direct open can fail on a mount that
        // does not support it, and then this file is buffered while others stay direct. Every read
        // must carry this rather than the strategy's wish, or the engine would set IOSQE_ASYNC on a
        // direct read, or skip it on a buffered one.
        bool direct = false;

        common::ResponseCode error = common::ResponseCode::Success;
    };

    struct Inflight
    {
        Workload workload;
        std::vector<BatchFd> fds;   // one per batch, indexed by InflightChunk::batch_index
        // Chunks, not tasks. Every chunk completes exactly once, and every non-empty task is in
        // exactly one chunk - so all chunks done already means all tasks answered, and the zero-sized
        // ones (answered at enqueue, and able to sit inside a span) never enter the arithmetic.
        //
        // ObjectStorageWorker counts tasks because there a task spans several chunks. After the task
        // cut ours is the inverse, so the counter inverts with it.
        size_t remaining_chunks = 0;
        std::map<unsigned, common::ResponseCode> error_by_file_index;   // first error per file
    };

    using InflightMap = std::map<uint64_t, Inflight>;

    // Stage the outstanding part of a request - the whole chunk the first time, the remainder after a
    // short read. Resolves it here if it cannot be staged, since no completion would then arrive.
    void stage_pending(posix_io::RequestId id);

    // Push each batch's aggregate result: the whole-workload code if non-Success, else that file's
    // recorded error.
    void report_workload(Inflight & wl, common::ResponseCode code);

    // Account every task a landed chunk covered - one read carried them all, so they share its
    // outcome - and finalize the workload once its last task lands.
    void complete_chunk(posix_io::RequestId id, common::ResponseCode ret);

    // The batch's descriptor, opening it on first use. Returns -1 and sets out_error if it cannot be
    // opened - that is this file's failure, not the storage's.
    //
    // Takes the FIRST CHUNK's file offset and destination, because the open needs them. Whether a
    // direct read is possible at all depends on congruence - the address and the file offset having
    // the same remainder - and that cannot be known until there is a destination. Opening lazily on
    // the first staged chunk is what makes the answer available in time.
    //
    // A file is opened once and keeps that mode. Congruence is a property of the whole region, not of
    // one chunk: the ranges of a region are contiguous, so if the first chunk is congruent the rest
    // are too.
    // Can this file be opened at all? Success or FileAccessError.
    //
    // Used to answer zero-sized ranges, which produce no chunk and so would otherwise be answered
    // without the file ever being touched. The file is opened and closed again - fd_for() opens it
    // properly later, when it knows whether O_DIRECT can be used.
    static common::ResponseCode probe_open(const std::string & path);

    // Close the interval the current in-flight level lasted for. Called before every change to
    // _issued.
    void account_inflight();

    int fd_for(Inflight & wl, unsigned batch_index, size_t file_offset, const char * buffer,
               common::ResponseCode & out_error);

    // Whether this file should be opened with O_DIRECT.
    //
    // Two things must both hold: the strategy asked for it, and a direct read is possible at all. The
    // second is congruence. Without it no part of the region can be read directly, so O_DIRECT would
    // copy every byte through a scratch buffer - on our one worker - while buffered I/O copies too
    // and gets readahead as well. So a non-congruent file is opened buffered on purpose, not as a
    // failure.
    bool wants_direct(size_t file_offset, const char * buffer) const;

    // What one pass of a DIRECT read should ask the kernel for.
    //
    // A direct read needs offset, length and address to be block multiples, and a region rarely starts
    // or ends on a block boundary. So a pass is one of three, decided from the two numbers the chunk
    // already carries - no state, no phase:
    //
    //   cursor not on a block boundary   HEAD    one block into scratch, copy from (cursor % block)
    //   on a boundary, >= a block left   MIDDLE  whole blocks, straight into the destination
    //   on a boundary, < a block left    TAIL    one block into scratch, copy what is left
    //
    // The existing re-stage loop then walks head -> middle -> tail by itself, because each pass
    // advances the cursor and the next pass looks at it afresh.
    //
    // Returns false when a scratch buffer was needed and none was free. The caller then leaves the
    // read buffered rather than failing it.
    struct DirectPass
    {
        size_t offset = 0;      // what to ask the kernel for
        size_t bytesize = 0;
        char * buffer = nullptr;

        char * scratch = nullptr;   // null when the pass goes straight to the destination
        size_t skip = 0;            // where the wanted bytes start inside the scratch
        size_t wanted = 0;          // how many of them this pass yields
    };

    bool plan_direct_pass(const Chunk & pending, size_t block, DirectPass & out);

    // Copy a bounced pass out of its scratch buffer, give the buffer back, and return how many WANTED
    // bytes arrived.
    //
    // The conversion matters: the kernel reports a whole block, but only part of it was asked for.
    // Recording the raw count would advance the cursor past bytes that were never delivered, and the
    // read would silently skip data.
    // The file an in-flight request was staged against.
    //
    // Needed to map a failed completion: EINVAL means our alignment rule broke on a direct fd, and
    // something else on a buffered one. The engine cannot decide that, because a completion carries
    // only the id, so this worker does it.
    posix_io::FileRef file_of(const InflightChunk & entry) const;

    // Reopen this batch's file WITHOUT O_DIRECT after the kernel refused a direct read, so the same
    // bytes can be re-staged. False when there is nothing left to re-stage against, or the buffered
    // open itself failed - the caller then reports the original error.
    // Reopen this batch's file buffered after the kernel refused a direct read.
    //
    // Success when the descriptor was replaced and the caller should re-stage. Otherwise the code to
    // answer the chunk with: FileAccessError when the reopen failed, which belongs to the file, and
    // UnknownError when the workload has already gone, which belongs to us.
    // Record that this engine will never serve another read, and tell the streamer so the mount drops
    // to the synchronous reader. Idempotent: the first reason is kept.
    void engine_is_dead(common::ResponseCode code);

    common::ResponseCode demote_to_buffered(const InflightChunk & entry);

    size_t land_bounced_pass(posix_io::RequestId id, size_t bytes_transferred);

    void finalize(InflightMap::iterator it, common::ResponseCode code);

    // Wait until no read is still in flight, so no destination can be written after its range is
    // reported. Unbounded on purpose - see the definition.
    void quiesce();

    void abort_all(common::ResponseCode code);

    const posix_io::Strategy _strategy;

    // This mount's direct-I/O block. NOT const: it starts provisional when the first submission could
    // not measure the mount, and is adopted from a later workload that could.
    //
    // Provisional means MaxProbeBlock - the largest the LADDER can ever return. A measurement from
    // statx is bounded by nothing here, so adoption checks the scratch pool rather than assuming the
    // value can only shrink; see enqueue().
    //
    // The alternative was to freeze the provisional value for the life of the engine. That is a bug
    // for a long-lived streamer serving many submissions - checkpoint restore - where the first
    // submission's files may be unreadable and every later one is fine.
    size_t _block;

    // False until a real measurement has been adopted. Only then does _block stop being provisional.
    bool _block_measured = false;

    const unsigned _node_wide_depth;

    // Set once a measurement was refused for being larger than the scratch buffers. It only silences
    // the warning: _block_measured stays false, so a later submission reporting a block this engine
    // CAN serve is still adopted. Without it the same line would be logged on every submission for
    // the life of the process.
    bool _block_too_large_reported = false;

    // Set when the kernel refused a DIRECT read with EINVAL, so later files open buffered instead of
    // repeating an attempt we already know fails.
    //
    // A plain member is enough, and is deliberately NOT a MountCapabilities lookup. This worker owns
    // one engine, and BackendPools creates one engine per mount - so a per-worker flag IS a per-mount
    // memo, with no plumbing. The worker holds no MountCapabilities and needs none.
    //
    // One direction only, like IoUringProbe::mark_unavailable. A mount that refused an aligned direct
    // read will not start accepting one while we run, and re-trying would cost a failed read per file.
    bool _direct_refused = false;
    const EngineFactory _factory;

    // Why the engine could not be built, for discard() to report. Reset after use so the next workload
    // retries - though a failure here is permanent in practice.
    common::ResponseCode _engine_error = common::ResponseCode::UnknownError;

    // Set once the engine has failed for good, mid-run. Every later workload is answered with
    // _engine_error instead of being staged - see abort_all for why reuse is unsafe and not merely
    // pointless.
    //
    // NOT set on the shutdown path: there the engine is destroyed immediately afterwards.
    bool _engine_dead = false;

    // Told once, when _engine_dead is set. See the constructor.
    std::function<void()> _on_engine_dead;

    // Block-sized buffers for the partial blocks at the edges of a region. Empty on a buffered
    // engine, which never bounces.
    std::unique_ptr<posix_io::ScratchPool> _scratch;

    std::optional<AsyncIoSettings> _settings;          // resolved with the engine, on the first workload
    std::unique_ptr<posix_io::IoEngine> _engine;

    InflightChunks _chunks;   // id -> chunk, its progress, and where to route it
    InflightMap _inflight;

    uint64_t _next_workload_id = 1;

    // Issued and not yet completed. Tracked SEPARATELY from staged: staged is not issued, so waiting
    // with nothing issued waits on an empty ring for a completion that will never arrive. When this is
    // zero the wait must be NonBlocking.
    size_t _issued = 0;

    // The most reads this worker ever had outstanding at once.
    //
    // Tracked beside _issued rather than derived from it, because _issued is a live count that rises
    // and falls; by the time anyone asks, the peak is long gone. Written only by the worker thread,
    // read by anyone - hence atomic.
    std::atomic<unsigned> _achieved_depth{ 0 };

    // Time-weighted in-flight, as a numerator and a denominator so workers can be summed. See
    // AsyncIoCounters for why the peak alone is not enough.
    //
    // Written only by the worker thread and read by anyone, hence atomic - the same reason
    // _achieved_depth is.
    std::atomic<uint64_t> _inflight_nanos{ 0 };
    std::atomic<uint64_t> _observed_nanos{ 0 };

    // When the current in-flight level began. 0 until the first change, so the interval before any
    // read was issued is not counted as time spent at depth zero.
    uint64_t _inflight_since = 0;

    // Reads the kernel answered short, and which were re-staged for the remainder. Counted where the
    // re-stage happens, so it counts passes rather than chunks: one chunk answered in three pieces
    // adds two.
    std::atomic<uint64_t> _short_read_restages{ 0 };

    // Bytes copied out of scratch. See bounced_bytes().
    std::atomic<uint64_t> _bounced_bytes{ 0 };

    // Every byte delivered, bounced or not. Only meaningful next to _bounced_bytes: the RATIO is what
    // says whether direct reads are working, and a count of copied bytes alone cannot say that.
    std::atomic<uint64_t> _bytes_read{ 0 };

    // So the warning below is said once, not once per workload.
    bool _warned_about_bouncing = false;

    // Harvested into on every drain, sized once - nothing is allocated while completing.
    std::vector<posix_io::Completion> _completions;
};

// Every async worker the streamer has created, so their counters can be summed.
//
// Filled by the worker FACTORY rather than by walking the pools. The pools hold workers as
// Worker<Workload>, which knows nothing of counters, so reaching them from there would mean a
// dynamic_cast or a virtual method on a base that object storage shares. The factory already knows
// the concrete type, because it is the thing that constructs it.
//
// Held by shared_ptr and captured by value, like the strategy resolver and the credentials state, so
// the factory never captures `this` and the registry outlives the pools whatever the destruction
// order.
//
// RAW POINTERS, and they are not owning: each worker belongs to its pool. Nothing is ever removed,
// because pools are created on first use and live as long as the streamer - so a registered worker is
// alive for as long as anything can ask. If pools ever became destructible, this needs revisiting.
//
// Thread safe. Pools are created lazily, so workers can be registered while another thread reads.
class AsyncIoWorkers
{
 public:
    void add(const AsyncIoWorker * worker);

    // The sum over every worker. See AsyncIoCounters for why a sum is the right shape for three of
    // the four and a maximum for the last.
    AsyncIoCounters total() const;

    size_t size() const;

 private:
    mutable std::mutex _mutex;
    std::vector<const AsyncIoWorker *> _workers;
};

}; // namespace runai::llm::streamer::impl
