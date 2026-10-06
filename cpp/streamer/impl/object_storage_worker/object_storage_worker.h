/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <deque>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "common/backend_api/response/response.h"
#include "common/response_code/response_code.h"
#include "common/s3_credentials/s3_credentials.h"

#include "common/device/device.h"

#include "streamer/impl/config/config/config.h"
#include "streamer/impl/device_io/device_issuer/device_issuer.h"
#include "streamer/impl/device_io/device_writer/device_writer.h"
#include "streamer/impl/device_io/staging_pool/staging_pool.h"
#include "streamer/impl/object_storage_worker/object_storage_retry.h"
#include "streamer/impl/reader/reader.h"
#include "streamer/impl/workload/workload.h"

#include "utils/capacity_worker/capacity_worker.h"
#include "utils/deque/deque.h"

namespace runai::llm::streamer::impl
{

// One ranged read, covering a span of WHOLE tasks - the batch's own chunk (Batch::chunks), cut where
// the tasks were. The span is not stored here: it is recovered from the async handle via the worker's
// _inflight map, the same lookup the completion path uses.
struct ObjectChunk
{
    common::backend_api::ObjectRequestId_t handle;   // unique async handle, inside the owning workload's block
    size_t offset;                                   // absolute file offset
    size_t bytesize;                                 // chunk length
    char * buffer;                                   // destination
};

// ObjectStorageWorker - the per-thread worker driving the object-storage pool. It is a CapacityWorker,
// so the base owns the in-flight window (a CapacityQueue of ObjectChunk) and runs the submit/drain
// interleave; this class only fills the backend hooks. Because the base interleaves, ONE worker
// can hold chunks from SEVERAL workloads in flight at once, refilling the window from whichever workload
// has pending chunks - so a small/late submission is served without waiting for a large one to drain.
//
// It is plugin-agnostic (hence the name, not S3Worker): it drives a Reader over an S3ClientWrapper, which
// dispatches to the s3/gcs/azure plugin by URI. The reader/client is built once, on the first workload, from
// the streamer's credentials (read via the credentials provider - see the constructor), and reused for every
// later workload on this thread. Credentials are streamer-scoped and set once (a differing set is rejected at
// the streamer), so one client serves every workload - and every bucket - under that single identity.
class ObjectStorageWorker : public utils::CapacityWorker<Workload, ObjectChunk>
{
 public:
    // credentials_provider supplies the streamer's object-storage credentials. It is invoked ONCE, when this
    // worker builds its client (see capacity), so credentials are read only at client creation - never on the
    // per-request path.
    // `writer` and `issuer` are the streamer's copy path, shared by every worker. Null in tests that
    // read to host memory only: a device workload is then refused rather than read, because writing a
    // device pointer through the plugin is a segmentation fault and not an error.
    // `retainer` keeps this worker's staging pool alive past the worker, so its pinned memory is not
    // freed while a parked client may still be writing into it. Null in tests, which have no client
    // outliving them.
    explicit ObjectStorageWorker(std::function<common::s3::Credentials()> credentials_provider,
                                 std::shared_ptr<DeviceWriter> writer = nullptr,
                                 std::shared_ptr<DeviceIssuer> issuer = nullptr,
                                 std::shared_ptr<StagingPoolRetainer> retainer = nullptr);

    // Waits for every read and copy still in flight. MUST come before anything else is destroyed: a
    // read is writing into the staging pool's pinned memory, and a copy's completion holds `this`
    // while the waiter's thread is still running. The pool drains before joining, so this normally
    // finds nothing - it is the teardown that does not go through abort_all that needs it.
    ~ObjectStorageWorker();

    // Staging buffers this worker may hold BEYOND its in-flight window, so a chunk whose read has
    // landed can wait for its copy without costing another chunk its buffer.
    //
    // Two, measured: on a B200 over NFS the copy stage holds about 0.3 of a buffer at the rate the
    // storage sustains, and sweeping this from 0 to 64 put the peak at 1-2 and cost 16% at 64, where
    // the pinned memory starts to hurt. See AsyncIoWorker::DefaultCopyDepth, which is the same number
    // for the same reason.
    static constexpr unsigned CopyDepth = 2;

 protected:
    // First workload sizes the window: build the persistent reader/client from its params and return the
    // plugin's in-flight window as a chunk count (max_inflight_bytes / chunk_bytesize; unbounded for gcs/azure).
    // THROWS if the window cannot be sized - an empty workload (EmptyRequestError) or a client that cannot be
    // built (the build code, recorded in _reader_error). The base catches it and calls discard() (below), so
    // the workload is finalized and the window is retried on the next workload.
    std::size_t capacity(const Workload & first) override;

    // Called by the base when capacity() (or the queue allocation) threw: finalize this workload so the
    // consumer still gets responses, with a code that reflects why it could not be admitted (empty ->
    // EmptyRequestError, client build failure -> _reader_error, otherwise -> UnknownError).
    void discard(Workload && workload) override;

    // Enqueue the batches' chunks into the window and register per-task tracking so completions route
    // back to the owning tasks/batch. Zero-size tasks complete immediately: they belong to no chunk.
    // If an allocation here throws (OOM), the worker aborts its in-flight workloads as UnknownError rather
    // than leaving them unfinalized - see the catch in the definition; OOM requires the caller to abort anyway.
    void enqueue(Workload && workload) override;

    // Fire one chunk's async read.
    void submit(const ObjectChunk & chunk) override;

    // Promote due retries to the front of the capacity queue before the base selects chunks to submit.
    void pre_pump() override;

    // Process one batch of completions: route each to its task, free its window slot, and finalize a
    // workload once its last task lands. On stop / a drained responder, fail every in-flight workload.
    void drain_batch(std::atomic<bool> & stopped) override;

    bool has_deferred_work() const override;

 private:
    // Per-task runtime state, indexed by local task index (assigned 0..N-1 in enqueue order). Touched only
    // by this worker thread.
    struct TaskState
    {
        Batch * batch;                                              // owning batch: handle_response + file_index
        const Task * task;                                         // the task itself: handle_response arg
        common::ResponseCode error = common::ResponseCode::Success;
    };

    // One object read: the read itself, the tasks it covers as indices into Inflight::tasks, and its
    // retry budget.
    //
    // A chunk is read once and completes every task in its span at that moment, so there is no
    // per-task chunk counter: a task belongs to exactly one chunk, because Batch::chunks are built
    // from the same cut that produced the tasks.
    //
    // The ObjectChunk is kept rather than just its span because a retry re-enqueues this exact read
    // (see promote_due_retries).
    struct ChunkState
    {
        ObjectChunk chunk;
        size_t   first = 0;                 // first task of the span
        unsigned count = 0;                 // tasks in the span
        ObjectStorageRetry::State retry;

        // Where a DEVICE chunk's bytes land before they are copied. Here rather than on ObjectChunk,
        // which is a value the capacity queue copies in and out. Taken fresh in submit() and given
        // back by whoever ends the attempt - including a retry, which should not hold pinned memory
        // through its backoff when another chunk could be using it.
        StagingBuffer staging;
    };

    // Per-in-flight-workload state, owned here and kept alive until the workload's last task finalizes.
    struct Inflight
    {
        Workload workload;                                         // owns the batches/responder/config
        std::vector<ChunkState> chunks;                            // handle - handle_base -> chunk state
        std::vector<TaskState> tasks;                              // per task (only non-zero-size tasks)
        std::map<unsigned, common::ResponseCode> error_by_file_index;   // first error per file (finalize)
        // Chunks, not tasks. Every chunk completes exactly once, and every task belongs to exactly one
        // chunk, so all chunks done already means all tasks answered - and the zero-sized ones, which
        // are answered at enqueue and can sit inside a span, never enter the arithmetic.
        size_t remaining_chunks = 0;                               // 0 -> finalize
    };

    // In-flight workloads keyed by their handle block base. The blocks are contiguous and disjoint, so a
    // completion handle routes to its workload via upper_bound (see locate); a map (not a list) so
    // lookups/erase go by key - the number in flight on one worker can be large.
    using InflightMap = std::map<common::backend_api::ObjectRequestId_t, Inflight>;

    // Find the workload owning `handle` (the block whose [base, base+size) contains it) and that chunk's
    // index within the block. Returns {end(), 0} if no block contains it (stale / out-of-range).
    std::pair<InflightMap::iterator, size_t> locate(common::backend_api::ObjectRequestId_t handle);

    // Account one completed chunk: free the window slot and report EVERY task it covered - one read
    // carries them all, so they succeed or fail together. Finalizes the workload once its last task lands.
    // `free_slot` is false for a chunk whose COPY has just retired: its window slot went back when the
    // read landed, because the window counts reads in flight and a copy must not cost a read its
    // place.
    void complete_chunk(InflightMap::iterator wlit, size_t chunk_idx, common::ResponseCode ret,
                        bool free_slot = true);

    // Hand a landed device chunk to the issuer. The window slot goes back here - the read is done -
    // and the tasks are answered later, from copy_done(), once the bytes are on the device.
    //
    // Cannot fail: the issuer answers through the completion below whatever happens, including when
    // it could not open the device at all.
    void issue_copy(InflightMap::iterator wlit, size_t chunk_idx);

    // A copy that has retired, routed back to THIS thread. Answering a task and freeing a window slot
    // are this worker's business, and the completion arrives on the StreamWaiter's.
    struct CopyDone
    {
        common::backend_api::ObjectRequestId_t handle = 0;
        common::ResponseCode ret = common::ResponseCode::Success;
    };

    // Account every copy that has landed since the last turn.
    void drain_copies();

    // Waits until every copy handed to the issuer has reported. Called where this worker is about to
    // stop being a valid callback target, and before an abort reports its ranges.
    void quiesce_copies();

    // Waits until the backend holds none of our reads. Every submitted read reports exactly once, so
    // the in-flight credit reaching zero is the proof - the same condition AsyncIoWorker::quiesce()
    // waits on.
    //
    // FALSE when it could not get there: the responder has nothing left to give while reads are still
    // outstanding, so those reads will never report and their buffers are unsafe to hand back.
    bool quiesce_reads();

    // Give every parked chunk another chance at a staging buffer. One pass per queue: a chunk the pool
    // still cannot serve goes back to the end of its own queue.
    void resume_waiting();
    void resume_from(std::deque<common::backend_api::ObjectRequestId_t> & waiting);

    // Move retry entries whose jitter delay elapsed to the FRONT of the capacity queue, so a retry is
    // preferred over a new submission.
    void promote_due_retries();

    // Push each batch's aggregate result: the whole-workload `code` if non-Success, else the batch's own
    // per-file error (Success for files whose tasks all succeeded).
    void report_workload(Inflight & wl, common::ResponseCode code);

    // report_workload + erase from _inflight.
    void finalize(InflightMap::iterator wlit, common::ResponseCode code);

    // Fail every in-flight workload with `code` and zero the window, so the worker becomes idle and the
    // pool can join. Used on teardown (stopped) and when the responder drains early.
    // `worker_continues` is false only at teardown. It decides what happens to a staging buffer whose
    // read never reported: a worker that keeps running must not hand that memory to another chunk,
    // while at teardown the pool outlives the clients anyway (see StagingPoolRetainer).
    void abort_all(common::ResponseCode code, bool worker_continues = true);

    // This worker, not the backend, is what the submission is waiting for: a chunk parked for a staging
    // buffer was never submitted, and a copy still in flight will return the buffer that releases one.
    //
    // A drained backend responder then means "nothing in flight right now", not "nothing more is
    // coming" - the two are indistinguishable from the plugin and only this tells them apart.
    bool holding_work() const;

    // This worker's own pinned buffers, built on the first DEVICE chunk and sized to its window plus
    // CopyDepth. One consumer - this thread - so nothing contends for them.
    common::ResponseCode staging_pool_for(const common::Device & device);

    std::function<common::s3::Credentials()> _credentials_provider;   // streamer credentials, read once at client build

    // The copy path. The WRITER opens devices and owns a stream per device; the ISSUER is shared by
    // every reading thread in the streamer and enqueues the copies, so this worker never touches the
    // driver itself. Borrowed, not owned: the streamer outlives its pools.
    const std::shared_ptr<DeviceWriter> _writer;
    const std::shared_ptr<DeviceIssuer> _issuer;

    // Outlives this worker, and the pool is handed to it on creation. See StagingPoolRetainer.
    const std::shared_ptr<StagingPoolRetainer> _retainer;

    std::shared_ptr<StagingPool> _pool;

    // Filled by the StreamWaiter's thread, drained by this one.
    utils::Deque<CopyDone> _copies;

    // Chunks with no staging buffer, holding their window slots until one comes back.
    //
    // TWO QUEUES, each in arrival order. A chunk that has already attempted is running against a
    // wall-clock deadline that keeps ticking while it waits, so it goes before chunks that have
    // attempted nothing - which is what promote_due_retries() already gives it in the capacity queue,
    // through enqueue_front, and what parking would otherwise throw away.
    //
    // Not one queue with retries pushed to the front: that would order the retries among themselves
    // backwards, putting the newest - the one with the most deadline left - first.
    std::deque<common::backend_api::ObjectRequestId_t> _waiting_retries;
    std::deque<common::backend_api::ObjectRequestId_t> _waiting;

    // Copies handed over and not yet accounted. Keeps this worker busy while they are outstanding, so
    // the pool gives it the turns it needs to drain them.
    size_t _copies_issued = 0;

    // What capacity() answered: this worker's in-flight window, in chunks. Kept because the staging
    // pool is sized from it and is built later, on the first device chunk.
    size_t _window_chunks = 0;
    std::shared_ptr<const Config> _config;   // keeps the Config alive for the persistent reader's reference
    std::shared_ptr<Reader> _reader;         // persistent, built by capacity() on the first non-empty workload
    common::ResponseCode _reader_error = common::ResponseCode::Success;   // last client-build failure code (for discard)
    size_t _chunk_bytesize = common::s3::S3ClientWrapper::default_chunk_bytesize;   // sane object-storage default until capacity() sets it
    unsigned _max_responses = 1;

    InflightMap _inflight;
    ObjectStorageRetry _retry;

    // Next async chunk handle. The backend requires a unique handle per in-flight request on a client, and
    // each worker has its own client - so a completion only ever returns to the worker that submitted it and
    // is looked up in that worker's _inflight. Handles therefore only need to be unique per client, not
    // process-global: this is a per-worker member (single-owner worker state, like _inflight, so non-atomic).
    // Starts at 1 so a chunk handle is never 0, keeping 0 free as an unmistakable "not a real completion"
    // sentinel (some plugins emit it on a drained responder).
    // NOTE: if a client is ever shared across workers, this must become process-global-unique again.
    common::backend_api::ObjectRequestId_t _async_handle_counter = 1;
};

}; // namespace runai::llm::streamer::impl
