#include "streamer/impl/object_storage_worker/object_storage_worker.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

#include "streamer/impl/s3/s3.h"

#include "common/s3_wrapper/s3_wrapper.h"
#include "common/range/range.h"
#include "common/exception/exception.h"

#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

ObjectStorageWorker::ObjectStorageWorker(std::function<common::s3::Credentials()> credentials_provider,
                                         std::shared_ptr<DeviceWriter> writer,
                                         std::shared_ptr<DeviceIssuer> issuer) :
    _credentials_provider(std::move(credentials_provider)),
    _writer(std::move(writer)),
    _issuer(std::move(issuer))
{}

common::ResponseCode ObjectStorageWorker::staging_pool_for(const common::Device & target)
{
    if (_pool != nullptr)
    {
        return common::ResponseCode::Success;
    }

    if (_writer == nullptr || _issuer == nullptr)
    {
        LOG(ERROR) << "This reader has no copy path, so it cannot serve " << target;
        return common::ResponseCode::DeviceUnavailable;
    }

    // Through the device this chunk names. Pinned memory is reachable from every context, so the pool
    // built here serves any later device this worker reads for.
    DeviceWriter::Channel channel = nullptr;
    const auto code = _writer->open(target.id, channel);
    if (code != common::ResponseCode::Success)
    {
        return code;
    }

    // AND BIND IT. Pinning is a driver call and a new thread inherits no context: open_device only
    // retains the primary context, it does not make it current here. Without this cuMemHostAlloc
    // answers CUDA_ERROR_INVALID_CONTEXT - which no mock can show, because a mock has no contexts.
    const auto device = _writer->device(channel);
    const auto bound = device->bind_thread();
    if (bound != common::ResponseCode::Success)
    {
        LOG(ERROR) << "Could not bind a context for " << target << ": " << bound;
        return bound;
    }

    StagingPool::Params params;
    params.buffer_bytesize = _chunk_bytesize;
    params.slab_bytesize = _chunk_bytesize;   // one registration per buffer, and only as they are needed
    // A pool cannot be sized from the unset sentinel. Falling back to the copy depth alone reads
    // slowly rather than pinning without limit, and says so.
    if (_window_chunks == static_cast<size_t>(-1))
    {
        LOG(WARNING) << "This object storage plugin advertises no in-flight window; staging "
                     << CopyDepth << " buffers for the device copy and no more";
    }

    // The clamp is not a memory policy, it is a guard: the window is a size_t from a plugin, and
    // max_buffers is an unsigned that is multiplied by the chunk size. A window this large is already
    // 32 GiB of staging at the default chunk, so anything past it is a plugin reporting nonsense.
    static constexpr size_t SaneWindowChunks = 4096;

    const size_t window = _window_chunks == static_cast<size_t>(-1) ? 0 : _window_chunks;
    params.max_buffers = static_cast<unsigned>(std::min(window, SaneWindowChunks)) + CopyDepth;

    _pool = std::make_shared<StagingPool>(device, params);
    return common::ResponseCode::Success;
}

std::size_t ObjectStorageWorker::capacity(const Workload & first)
{
    // An empty workload (no batches) carries no params to build a client from and nothing to read. The
    // streamer never dispatches empty workloads (they are skipped at dispatch, empty submissions rejected in
    // verify_requests), so this is not a production path. Throw so the base discards it (discard() pushes no
    // responses, since there are no batches) and retries the window on the next workload.
    if (first.batches().empty())
    {
        LOG(WARNING) << "Object storage worker received an empty workload; the streamer is expected to skip these";
        throw common::Exception(common::ResponseCode::EmptyRequestError);
    }

    // Build the persistent reader/client once, from the first workload that can. Keep a shared_ptr to the
    // Config so the reference S3 holds stays valid after the building workload finalizes (batches dropped).
    if (_reader == nullptr)
    {
        const auto & batch = first.batches().front();
        _config = batch.config;
        _chunk_bytesize = std::max(static_cast<size_t>(1), _config->s3_block_bytesize);
        _retry = ObjectStorageRetry(_config->object_storage_retry_timeout);

        // Request one completion at a time by default for prompt, per-completion window refill;
        // RUNAI_STREAMER_INTERNAL_MAX_RESPONSES can raise it (internal tuning / test knob).
        _max_responses = utils::getenv_positive<unsigned>("RUNAI_STREAMER_INTERNAL_MAX_RESPONSES", 1U);

        try
        {
            // Credentials are streamer-scoped and read exactly once, here at client creation (never on the
            // per-request path). The batch params carry only the URI; combine it with the credentials for the
            // client config.
            const auto credentials = _credentials_provider();
            // The whole object-storage capacity, not this worker's share: S3 runs one worker holding
            // one client, so that client must be sized for all of it. Dropped for GCS and Azure, which
            // reach the same capacity with one client per worker.
            common::s3::S3ClientWrapper::Params client_params(batch.object_storage_params.uri, credentials,
                                                              _chunk_bytesize, _config->s3_concurrency);
            auto client = std::make_shared<common::s3::S3ClientWrapper>(client_params);
            _reader = std::make_shared<S3>(client, *_config);
        }
        catch (const common::Exception & e)
        {
            // e.g. the plugin library is missing (S3NotSupported). Record the code for discard() and rethrow:
            // the base discards this workload with it and retries the client on the next workload.
            LOG(ERROR) << "Failed to create object storage client: " << e.error();
            _reader_error = e.error();
            throw;
        }
    }

    // The window is a max in-flight chunk count: the plugin's byte window / chunk size. Each in-flight
    // chunk costs 1.
    //
    // SIZE_MAX is the UNSET sentinel, not "unbounded": every plugin advertises a real window, and gcs
    // and azure size theirs to their own threadpool (margin x readers x chunk). The value is filled in
    // when a client is created, and this runs after that - so it is not reached in practice.
    const size_t unbounded = static_cast<size_t>(-1);
    const size_t window_bytes = _reader->max_inflight_bytes();

    _window_chunks = (window_bytes == unbounded)
        ? unbounded
        : std::max(static_cast<size_t>(1), window_bytes / _chunk_bytesize);

    return _window_chunks;
}

void ObjectStorageWorker::discard(Workload && workload)
{
    // The base could not bring the window up for this workload. Finalize it with a code that reflects why:
    // an empty workload has no batches (so this pushes nothing); a client-build failure uses the recorded
    // code; anything else (e.g. the queue allocation threw) is UnknownError.
    common::ResponseCode code;
    if (workload.batches().empty())
    {
        code = common::ResponseCode::EmptyRequestError;
    }
    else if (_reader_error != common::ResponseCode::Success)
    {
        code = _reader_error;
    }
    else
    {
        code = common::ResponseCode::UnknownError;
    }

    Inflight wl;
    wl.workload = std::move(workload);
    report_workload(wl, code);

    _reader_error = common::ResponseCode::Success;   // reset for the next attempt
}

void ObjectStorageWorker::enqueue(Workload && workload)
{
    // Count chunks up front so we can reserve one contiguous block of handles and size chunk_tasks.
    //
    // The chunks are the batch's own (Batch::chunks), built where the tasks were cut and at the same
    // size, so this worker never re-derives the grouping. That is also what gives object storage
    // request PACKING: several small tensors falling inside one chunk become ONE ranged read, where
    // previously each task was chunked on its own and a small tensor meant a small request.
    size_t total_chunks = 0;
    for (const auto & batch : workload.batches())
    {
        total_chunks += batch.chunks.size();
    }

    // A workload with no chunks (only zero-size tasks) is reported inline and never entered into _inflight:
    // it reserves no handle block, so it would have no unique key, and there is nothing to route to it.
    if (total_chunks == 0)
    {
        Inflight wl;
        wl.workload = std::move(workload);
        // the workload was fully populated via add_batch before dispatch, so batches() is complete here
        for (auto & batch : wl.workload.batches())
        {
            for (const auto & task : batch.tasks)
            {
                common::backend_api::Response resp(common::ResponseCode::Success);
                batch.handle_response(resp, &task);
            }
        }
        report_workload(wl, common::ResponseCode::Success);
        return;
    }

    // enqueue only runs once the window is up (the base creates _queue only after capacity() succeeded), so
    // the reader is built and _chunk_bytesize is correct here - no retry needed.

    // Reserve this workload's contiguous handle block.
    const auto handle_base = _async_handle_counter;
    _async_handle_counter += total_chunks;

    // Registration and chunk-building allocate (the map node, chunk_tasks, the tasks vector, the queue
    // entries); under memory pressure any of these can throw. The workload's expected responses were already
    // counted (responder increment + submissions add) before dispatch, so bailing out here without pushing
    // them hangs the consumer forever. On a throw we finalize the workload as UnknownError instead - best
    // effort (report_workload could itself fail under severe OOM).
    try
    {
        auto [wlit, inserted] = _inflight.emplace(handle_base, Inflight{});
        ASSERT(inserted) << "duplicate handle base " << handle_base;

        Inflight & wl = wlit->second;
        wl.workload = std::move(workload);   // noexcept; the workload now lives in the _inflight entry
        wl.chunks.resize(total_chunks);

        size_t next_chunk = 0;
        // &batch below outlives this loop (it is stored in wl.tasks and used to route completions): the
        // workload has already been moved into its _inflight entry, _inflight is node-stable, and no batch is
        // added to a dispatched workload - so the batches vector is never grown or moved again.
        for (auto & batch : wl.workload.batches())
        {
            // Task indices are per batch, so shift them into this workload's flat task vector. Every
            // task gets an entry, including zero-size ones, so a chunk's span indexes straight in.
            const size_t task_base = wl.tasks.size();
            for (const auto & task : batch.tasks)
            {
                wl.tasks.push_back(TaskState{ &batch, &task, common::ResponseCode::Success });

                if (task.info.bytesize == 0)
                {
                    // Zero-size tasks appear in no chunk (chunk_splitter.h), so nothing will ever
                    // complete for them - but they still owe a response each. Finish them here.
                    common::backend_api::Response resp(common::ResponseCode::Success);
                    batch.handle_response(resp, &task);
                }
            }

            for (const auto & chunk : batch.chunks)
            {
                // One queue entry costs 1 and the window is sized in chunks, so an over-long chunk
                // would be worth more bytes than the window assumes. Batches cuts at the same size
                // this worker reports, but nothing enforces that the two agree.
                ASSERT(chunk.bytesize <= _chunk_bytesize)
                    << "chunk of " << chunk.bytesize << " bytes exceeds " << _chunk_bytesize
                    << " - the task cut and the chunk size have diverged";

                const ObjectChunk object_chunk{ handle_base + next_chunk, chunk.offset, chunk.bytesize, chunk.buffer };
                wl.chunks[next_chunk] = ChunkState{ object_chunk, task_base + chunk.first_task, chunk.task_count, {} };
                _queue->enqueue(object_chunk, 1);   // cost 1
                ++next_chunk;
            }
        }

        wl.remaining_chunks = total_chunks;
    }
    catch (...)
    {
        // OOM mid-registration. The caller aborts on any UnknownError, so rather than reconstruct exact
        // responses we fail this worker's in-flight workloads (this one included - the bulk allocations run
        // after the move, so it is already in _inflight) and zero the window, clearing the half-built entry
        // and any chunks enqueued before the throw (else a stale one later hits submit()'s unknown-handle ASSERT).
        abort_all(common::ResponseCode::UnknownError);
    }
}

std::pair<ObjectStorageWorker::InflightMap::iterator, size_t> ObjectStorageWorker::locate(common::backend_api::ObjectRequestId_t handle)
{
    // upper_bound gives the first block whose base is > handle; the previous block is the candidate owner.
    auto it = _inflight.upper_bound(handle);
    if (it == _inflight.begin())
    {
        return { _inflight.end(), 0 };   // handle precedes every block
    }
    --it;

    const auto rel = handle - it->first;
    if (rel >= it->second.chunks.size())
    {
        return { _inflight.end(), 0 };   // falls in a gap between blocks / past this block
    }
    return { it, static_cast<size_t>(rel) };
}

void ObjectStorageWorker::submit(const ObjectChunk & chunk)
{
    auto [wlit, chunk_idx] = locate(chunk.handle);
    ASSERT(wlit != _inflight.end()) << "submitting a chunk with unknown handle " << chunk.handle;

    ChunkState & cs = wlit->second.chunks[chunk_idx];
    ASSERT(cs.count > 0) << "chunk " << chunk.handle << " covers no tasks";

    // Every task in the span shares this one read, and a task belongs to exactly one chunk - so none
    // of them can already have failed when this runs. Master's "the owning task already failed, skip
    // the read" short-circuit existed because a task was split across several chunks and an early
    // failure could doom the rest; with one chunk per span there are no siblings to short-circuit.
    TaskState & first = wlit->second.tasks[cs.first];
    ASSERT(first.error == common::ResponseCode::Success)
        << "task already failed before its only chunk was submitted";

    // A DEVICE chunk cannot be read where it is going: the plugin writes with the CPU, and a device
    // pointer would be a segmentation fault. It lands in pinned host memory and is copied afterwards.
    //
    // A FRESH buffer every attempt, retries included. Holding one through a backoff pins memory that
    // another chunk could be reading into, and the buffer is given back by whoever ends the attempt.
    char * destination = chunk.buffer;

    if (!first.batch->device.is_host())
    {
        const auto ready = staging_pool_for(first.batch->device);
        if (ready != common::ResponseCode::Success)
        {
            complete_chunk(wlit, chunk_idx, ready);
            return;
        }

        StagingBuffer buffer;
        const auto code = _pool->try_acquire(buffer);
        if (code != common::ResponseCode::Success)
        {
            complete_chunk(wlit, chunk_idx, code);
            return;
        }

        if (!buffer.valid())
        {
            // Every buffer is out: the window's reads hold theirs, and the copy headroom is spent on
            // chunks waiting for the link. Not an error - the chunk waits for a buffer, keeping its
            // window slot so the base admits nothing in its place.
            //
            // BEFORE the retry accounting below, so waiting costs no part of this chunk's retry
            // budget: it has not attempted anything yet.
            //
            // A chunk that HAS attempted waits in the other queue, which is drained first: its
            // deadline is already running, and a chunk that has attempted nothing has none.
            if (_retry.retry_count(cs.retry) > 0)
            {
                _waiting_retries.push_back(chunk.handle);
            }
            else
            {
                _waiting.push_back(chunk.handle);
            }
            return;
        }

        cs.staging = buffer;
        destination = buffer.data;
    }

    // ObjectStorageRetry starts the deadline here, so queueing before the first backend attempt does not
    // consume the chunk's retry budget. A delayed retry promoted after its deadline is rejected here.
    if (_retry.enabled() && !_retry.begin_attempt(cs.retry))
    {
        LOG(WARNING) << "Object chunk " << chunk.handle << " exhausted RUNAI_STREAMER_S3_TIMEOUT after "
                     << _retry.retry_count(cs.retry) << " application retries";
        complete_chunk(wlit, chunk_idx, common::ResponseCode::FileAccessError);
        return;
    }

    try
    {
        const common::Range range(chunk.offset, chunk.bytesize);
        _reader->async_read(first.batch->object_storage_params, chunk.handle, range, destination);
    }
    catch (const common::Exception & e)
    {
        // the read could not be issued: fail every task this chunk carried
        complete_chunk(wlit, chunk_idx, e.error());
    }
    catch (...)
    {
        complete_chunk(wlit, chunk_idx, common::ResponseCode::UnknownError);
    }
}

void ObjectStorageWorker::issue_copy(InflightMap::iterator wlit, size_t chunk_idx)
{
    Inflight & wl = wlit->second;
    ChunkState & cs = wl.chunks[chunk_idx];
    const auto & batch = *wl.tasks[cs.first].batch;

    const auto handle = cs.chunk.handle;

    // The window slot goes back HERE, not when the copy retires: the window counts reads in flight,
    // and a copy holding a slot would cost a read its place for as long as the link takes.
    _queue->complete(1);

    // Ours no longer, on either path: the issuer returns the buffer to the pool whatever happens.
    const StagingBuffer buffer = cs.staging;
    cs.staging = StagingBuffer{};

    _issuer->submit(batch.device.id, _pool, buffer, cs.chunk.bytesize, cs.chunk.buffer,
                    [this, handle](common::ResponseCode ret) { _copies.push(CopyDone{ handle, ret }); });

    ++_copies_issued;
}

void ObjectStorageWorker::drain_copies()
{
    CopyDone done;

    while (_copies_issued > 0 && _copies.try_pop(done))
    {
        --_copies_issued;

        auto [wlit, chunk_idx] = locate(done.handle);
        if (wlit == _inflight.end())
        {
            continue;   // its workload was aborted while the copy was in flight
        }

        if (done.ret != common::ResponseCode::Success)
        {
            LOG(ERROR) << "[RunAI Streamer] copy to device failed for chunk " << done.handle
                       << ": " << done.ret;
        }

        // The slot went back when the read landed, so this must not free it again.
        complete_chunk(wlit, chunk_idx, done.ret, false /* slot already freed */);
    }
}

void ObjectStorageWorker::complete_chunk(InflightMap::iterator wlit, size_t chunk_idx,
                                        common::ResponseCode ret, bool free_slot)
{
    Inflight & wl = wlit->second;
    ChunkState & cs = wl.chunks[chunk_idx];

    // A DEVICE chunk is not done when its read lands: the bytes are in pinned host memory. Hand the
    // copy over and answer the tasks from drain_copies(), so a range is answered only once its bytes
    // are on the device. issue_copy() frees the window slot itself.
    if (ret == common::ResponseCode::Success && cs.staging.valid() && free_slot)
    {
        issue_copy(wlit, chunk_idx);
        return;
    }

    // Still ours only when no copy was handed over - a failed read, or an attempt that is about to be
    // retried. A buffer lost here would shrink the pool for the life of the worker.
    if (cs.staging.valid())
    {
        _pool->release(cs.staging);
        cs.staging = StagingBuffer{};
    }

    if (free_slot)
    {
        _queue->complete(1);   // free the window slot so the next chunk can be submitted
    }

    const auto & span = wl.chunks[chunk_idx];

    // One read carried all of these, so they share its outcome.
    for (unsigned i = 0; i < span.count; ++i)
    {
        TaskState & ts = wl.tasks[span.first + i];

        // Zero-sized tasks were answered at enqueue and read nothing. They can fall inside a span,
        // and answering one again would be harmless - finished_request is idempotent - but skipping
        // says so rather than relying on it.
        if (ts.task->info.bytesize == 0)
        {
            continue;
        }

        ts.error = ret;

        if (ret == common::ResponseCode::Success)
        {
            common::backend_api::Response resp(common::ResponseCode::Success);
            ts.batch->handle_response(resp, ts.task);
        }
        else
        {
            wl.error_by_file_index.emplace(ts.batch->file_index, ret);   // first error per file
        }
    }

    // Once per chunk, after its tasks are answered - so this is independent of which tasks the span
    // happened to contain. wl is gone after finalize, so nothing may touch it below.
    ASSERT(wl.remaining_chunks > 0) << "completing a chunk of a workload with none outstanding";
    if (--wl.remaining_chunks == 0)
    {
        finalize(wlit, common::ResponseCode::Success);
    }
}

void ObjectStorageWorker::promote_due_retries()
{
    const auto now = ObjectStorageRetry::Clock::now();
    while (const auto handle = _retry.pop_due(now))
    {
        auto [wlit, chunk_idx] = locate(handle.value());
        ASSERT(wlit != _inflight.end()) << "retrying a chunk with unknown handle " << handle.value();
        _queue->enqueue_front(wlit->second.chunks[chunk_idx].chunk, 1);
    }
}

void ObjectStorageWorker::resume_waiting()
{
    // Retries first, then the rest, each in arrival order.
    //
    // One pass over each queue. submit() parks a chunk again when the pool is still dry, pushing it to
    // the BACK of its queue, so a count taken up front is what stops this looping on the same chunk.
    resume_from(_waiting_retries);
    resume_from(_waiting);
}

void ObjectStorageWorker::resume_from(std::deque<common::backend_api::ObjectRequestId_t> & waiting)
{
    for (size_t remaining = waiting.size(); remaining > 0; --remaining)
    {
        const auto handle = waiting.front();
        waiting.pop_front();

        auto [wlit, chunk_idx] = locate(handle);
        if (wlit == _inflight.end())
        {
            continue;   // its workload was aborted while it waited
        }

        submit(wlit->second.chunks[chunk_idx].chunk);
    }
}

void ObjectStorageWorker::pre_pump()
{
    if (_retry.enabled())
    {
        promote_due_retries();
    }
}

bool ObjectStorageWorker::has_deferred_work() const
{
    return _retry.has_pending() || _copies_issued > 0 || !_waiting.empty() || !_waiting_retries.empty();
}

void ObjectStorageWorker::report_workload(Inflight & wl, common::ResponseCode code)
{
    for (auto & batch : wl.workload.batches())
    {
        // whole-workload abort fails every file; otherwise fail only files with a recorded error
        // (handle_error(Success) is a no-op for files whose tasks all completed)
        auto error_code = code;
        if (error_code == common::ResponseCode::Success)
        {
            // batch.file_index, not the batch's position: one file can contribute several batches (one per
            // contiguous transfer), and errors are recorded per file, so several batches can share an entry
            auto it = wl.error_by_file_index.find(batch.file_index);
            if (it != wl.error_by_file_index.end())
            {
                error_code = it->second;
            }
        }
        batch.handle_error(error_code);
    }
}

void ObjectStorageWorker::finalize(InflightMap::iterator wlit, common::ResponseCode code)
{
    report_workload(wlit->second, code);
    _inflight.erase(wlit);
}

bool ObjectStorageWorker::holding_work() const
{
    return !_waiting.empty() || !_waiting_retries.empty() || _copies_issued > 0;
}

void ObjectStorageWorker::abort_all(common::ResponseCode code)
{
    // Fail and drop every in-flight workload - including any whose reads are still outstanding at the
    // backend. Those reads are not cancelled: they can still finish and be delivered later as "late
    // completions" - completion events whose handle points into a handle block we erase here. drain_batch
    // recognises a late completion (locate() returns end()) and simply drops it, so abandoning the reads
    // now is safe. This holds even when the worker keeps running afterwards: most calls are on a terminal
    // client state (teardown, or the responder stopped / drained early), but enqueue also calls abort_all
    // mid-life on an allocation failure (OOM). Late completions stay safe either way because handle blocks
    // are allocated from a monotonic counter and never reused: every erased block sits strictly below any
    // future block, so a late completion from an erased block has a handle below every live block's base and
    // always locates to end(). The cost of the mid-life call is that sibling in-flight workloads on this
    // worker are failed too, and their still-outstanding reads may write to already-reported buffers - the
    // OOM caller is expected to abort on UnknownError and tear the streamer down.
    _retry.clear();

    // Every staging buffer still held by a chunk that will never complete. Before the workloads are
    // erased, because the chunks go with them - and a buffer lost here would shrink the pool for the
    // life of the worker. A buffer already handed to the issuer is NOT here: that one comes back on
    // its own, and its completion locates to end() and is dropped.
    if (_pool != nullptr)
    {
        for (auto & [base, wl] : _inflight)
        {
            (void)base;
            for (auto & cs : wl.chunks)
            {
                if (cs.staging.valid())
                {
                    _pool->release(cs.staging);
                    cs.staging = StagingBuffer{};
                }
            }
        }
    }

    for (auto it = _inflight.begin(); it != _inflight.end(); )
    {
        auto next = std::next(it);
        finalize(it, code);   // fails every batch and erases `it`
        it = next;
    }

    // Copies handed over are no longer anyone's business here: their workloads are gone, so their
    // completions locate to end() and are dropped. Forgetting them is what lets idle() become true.
    _copies_issued = 0;

    // With them, or a waiting chunk would name a workload that no longer exists - and it would keep
    // this worker busy for good, because a parked chunk is deferred work.
    _waiting.clear();
    _waiting_retries.clear();

    // Zero the window so idle() becomes true and the pool can join. clear() drops every pending chunk and
    // releases all in-flight credit in one step - the workloads those chunks belonged to were already failed
    // and erased above, so their tracking is gone. A try_take()/complete() drain could not do this: try_take()
    // stops at the full-window boundary, so a workload with chunks >> capacity would leave the remainder in
    // _pending (idle() never true from a single call, and the outer loop would re-pump spurious submit()s).
    if (_queue != nullptr)
    {
        _queue->clear();
    }
}

void ObjectStorageWorker::drain_batch(std::atomic<bool> & stopped)
{
    if (_reader == nullptr)
    {
        return;   // nothing was ever submitted
    }

    if (stopped)
    {
        abort_all(common::ResponseCode::FinishedError);   // teardown: fail all in flight, empty the window
        return;
    }

    // Copies that landed while this thread was elsewhere. Before the window is examined below, so a
    // chunk whose copy has retired is off the books by then.
    drain_copies();

    // EVERY TURN, not only after a copy retires: a buffer also comes back from a read that failed and
    // from an attempt about to be retried, and neither of those reports a copy.
    resume_waiting();

    // With no backend attempt in flight, sleep in short slices while a retry is deferred so the thread does
    // not busy-wait. CapacityWorker calls pump() immediately after this returns; pre_pump() then promotes a
    // due retry to the front of the queue before any newly queued chunks are selected.
    if (_queue->inflight() == 0)
    {
        if (_retry.enabled() && _retry.has_pending() && _queue->empty())
        {
            if (const auto retry_at = _retry.next_due())
            {
                const auto now = ObjectStorageRetry::Clock::now();
                const auto until = std::min(retry_at.value(), now + std::chrono::milliseconds(20));
                std::this_thread::sleep_until(until);
            }
        }
        return;
    }

    std::vector<common::backend_api::Response> responses;
    const auto r = _reader->async_response(responses, _max_responses);
    if (r != common::ResponseCode::Success)
    {
        // Any code but FinishedError is a backend failure: no more completions are coming for this
        // worker's in-flight chunks whatever else it is holding, so fail them all.
        if (r != common::ResponseCode::FinishedError)
        {
            LOG(ERROR) << "Object storage responder returned " << r;
            abort_all(r);
            return;
        }

        // FinishedError says the backend has nothing in flight - which is ALSO what it says while this
        // worker is the one holding things up, because a parked chunk was never submitted. Aborting
        // there fails a whole submission for being slower at copying than at reading.
        if (holding_work())
        {
            return;   // the next turn retires a copy, frees a buffer and submits a parked chunk
        }

        abort_all(r);
        return;
    }

    bool progressed = false;
    bool responder_drained = responses.empty();   // Success but no events -> responder ran dry this round
    for (const auto & response : responses)
    {
        // Some plugins (azure/gcs) append an "empty" FinishedError event once the responder runs dry, to
        // signal there is nothing more to hand out this round. It is not a real completion - stop here.
        if (response.ret == common::ResponseCode::FinishedError)
        {
            responder_drained = true;
            break;
        }

        auto [wlit, chunk_idx] = locate(response.handle);
        if (wlit == _inflight.end())
        {
            // A late completion (see abort_all): a chunk whose workload was already erased by an abort_all
            // while its read was still outstanding at the backend, now delivered - its handle falls in an
            // erased block. Nothing is left to complete for it, so drop it and keep scanning. Never
            // abort_all here: that would fail every OTHER in-flight submission over a stray event. (A
            // corrupt / never-issued handle from a buggy plugin lands here too and is likewise dropped.)
            LOG(DEBUG) << "Dropping late object storage completion with unknown handle " << response.handle;
            continue;
        }

        auto ret = response.ret;
        if (ret == common::ResponseCode::RetryableFileAccessError)
        {
            ChunkState & cs = wlit->second.chunks[chunk_idx];

            // No sibling guard here. Master also asked whether the owning task had already failed,
            // because a task was split across several chunks and one failure doomed the rest -
            // retrying such a chunk would be wasted work. Our cut gives a task exactly one chunk, so
            // no other chunk can have failed its tasks first, and there is nothing to guard against.
            //
            // Worth knowing when reading a failure: a chunk here covers a WHOLE SPAN of tasks, so an
            // exhausted budget fails all of them together. The retry matters more than it did, not
            // less.
            const auto retry = _retry.schedule(cs.retry, cs.chunk.handle);
            if (retry.has_value())
            {
                // GIVEN BACK BEFORE THE BACKOFF. This path does not go through complete_chunk, and the
                // next attempt takes a fresh buffer - so holding this one would lose it for the life of
                // the worker, and would pin memory another chunk could be reading into meanwhile.
                if (cs.staging.valid())
                {
                    _pool->release(cs.staging);
                    cs.staging = StagingBuffer{};
                }

                // The failed attempt is no longer in flight; the logical chunk remains pending in _retry.
                _queue->complete(1);
                LOG(DEBUG) << "Retrying object chunk " << cs.chunk.handle << " (offset " << cs.chunk.offset
                           << ", bytes " << cs.chunk.bytesize << ") after " << retry->delay.count()
                           << " ms; application retry " << retry->retry_count;
                progressed = true;
                continue;
            }
            // The retry budget is disabled/exhausted. The internal marker must never escape to callers.
            ret = common::ResponseCode::FileAccessError;
        }

        complete_chunk(wlit, chunk_idx, ret);
        progressed = true;
    }

    // The responder signalled it ran dry (empty round or end-of-round sentinel) while chunks are still in
    // flight: it was stopped or drained early. Abort rather than spin re-reading the same sentinel. Gated
    // on responder_drained, not merely !progressed, so a round that only dropped a late completion (the
    // client is alive) never aborts an unrelated in-flight submission.
    //
    // AND NOT WHILE THIS WORKER IS THE ONE HOLDING THINGS UP. A chunk parked for a staging buffer, or a
    // copy still in flight, keeps its window slot - so the queue is not idle and the plugin has nothing
    // left to report, which looks exactly like a client that went away. Aborting there fails a whole
    // submission because the device link was slower than the storage.
    if (!progressed && responder_drained && !holding_work() && _queue != nullptr && !_queue->idle())
    {
        abort_all(common::ResponseCode::FinishedError);
    }
}

}; // namespace runai::llm::streamer::impl
