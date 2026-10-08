// O_DIRECT is a GNU extension, so this must come before any libc header.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "streamer/impl/async_io/async_io_worker/async_io_worker.h"

#include "posix_io/completion_mapper/completion_mapper.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>

#include "posix_io/alignment/alignment.h"
#include "common/exception/exception.h"
#include "utils/fd/fd.h"
#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

namespace
{

// How long a wait may block before returning empty so the worker can notice `stopped`.
//
// 50 ms, a constant rather than a knob. It has no throughput role - with requests outstanding the wait
// returns on the first completion, never at the timeout - so it only bounds how long teardown takes to
// be noticed. An idle wakeup costs a syscall return, so anything from 1 ms to 200 ms is defensible;
// there is nothing here a measurement would resolve.
constexpr unsigned WaitTimeoutMs = 50;

// One pinning call per slab. The whole pool is depth x chunk - 512 MiB at depth 64 - and filling it
// takes about 135 ms at the ~3.8 GB/s the driver registers at, which is the entire first-round
// deficit measured on nscale (16 GiB round: 10.64 GB/s first against 11.82 steady).
//
// Small slabs mean more calls; large ones mean the first read waits longer for its first slab. Which
// of the two dominates is measured rather than assumed - RUNAI_STREAMER_DEVICE_SLAB_BYTESIZE exists
// so the sweep can be run without rebuilding.
size_t device_slab_bytesize()
{
    static const size_t configured =
        utils::getenv_positive<size_t>("RUNAI_STREAMER_DEVICE_SLAB_BYTESIZE", 16ul << 20);
    return configured;
}

} // namespace

AsyncIoWorker::AsyncIoWorker(posix_io::Strategy strategy, size_t block, unsigned node_wide_depth,
                             EngineFactory factory, std::function<void()> on_engine_dead,
                             std::shared_ptr<DeviceWriter> writer, bool register_buffers) :
    _writer(std::move(writer)),
    _strategy(strategy),
    _block(block != 0 ? block : posix_io::MaxProbeBlock),
    _block_measured(block != 0),
    _node_wide_depth(node_wide_depth),
    _register_buffers(register_buffers),
    _factory(std::move(factory)),
    _on_engine_dead(std::move(on_engine_dead))
{
    ASSERT(posix_io::is_async(strategy))
        << strategy << " is served by the synchronous pool, not by an engine";
}

AsyncIoWorker::~AsyncIoWorker()
{
    // Every issued copy reports exactly once, so this cannot hang. It MUST happen before anything
    // else here: the completion lambda holds `this`, and the StreamWaiter's thread is still running.
    drain_copies(true);

    // The pool drains before joining, so this should find nothing. Closing anyway: a descriptor leaked
    // per workload would exhaust the process over a long-lived streamer, and this is the only owner.
    for (auto & [id, wl] : _inflight)
    {
        (void)id;
        for (auto & batch_fd : wl.fds)
        {
            if (batch_fd.fd >= 0)
            {
                ::close(batch_fd.fd);
            }
        }
    }
}

std::size_t AsyncIoWorker::capacity(const Workload & first)
{
    // An empty workload carries no config to size anything from. The streamer skips these before
    // dispatch, so this is not a production path - throw and let the base discard it.
    if (first.batches().empty())
    {
        LOG(WARNING) << "Async io worker received an empty workload";
        throw common::Exception(common::ResponseCode::EmptyRequestError);
    }

    // Resolved HERE, not at construction: depth is divided by RUNAI_STREAMER_PROCESS_GROUP_SIZE, and
    // the window size the base wants IS the engine's depth. One moment, one place.
    _settings.emplace(*first.batches().front().config, _node_wide_depth);

    posix_io::AsyncIoConfig config;
    config.depth = _settings->depth();
    config.chunk_bytesize = _settings->chunk_bytesize();

    config.direct_block = _block;
    config.register_buffers = _register_buffers;

    _engine = _factory(_strategy, config);
    if (_engine == nullptr)
    {
        // Permanent, not transient: a blocked io_uring stays blocked. Reaching this means the
        // dispatcher created an async pool for a host that cannot serve one, which is a bug in
        // strategy resolution.
        //
        // FsAsyncEngineError, not UnknownError. It IS very likely our bug, but the outcome the caller
        // needs is the same as for an engine that dies later: this mount is not read asynchronously,
        // everything else is unaffected, and the files are still readable by the synchronous pool.
        // UnknownError would tell the caller to abort everything and restart the streamer instead.
        LOG(ERROR) << "No engine for " << _strategy << " - this host cannot serve the strategy it was"
                   << " dispatched for. This mount is read by the synchronous reader from here on";

        engine_is_dead(common::ResponseCode::FsAsyncEngineError);
        throw common::Exception(common::ResponseCode::FsAsyncEngineError);
    }

    // The number every congruence test and every direct pass on this mount will use. Reported because
    // it is measured rather than fixed, so it can differ between mounts in one process and between
    // hosts - and because when direct reads quietly stop happening, a block larger than the caller's
    // padding is the first thing to check.
    if (posix_io::is_direct(_strategy))
    {
        LOG(INFO) << "Direct-I/O block for this mount: " << _block << " bytes"
                  << (_block_measured ? " (measured)"
                                      : " (provisional - no file could be probed yet; a later"
                                        " submission that can measure will replace it)");
    }

    // Beside the engine, because both are sized by the same number. The client allocates nothing
    // until a device workload actually asks for a buffer, so a host-only mount pays 160 bytes for it.
    //
    // ONE BUFFER PER CHUNK, so buffer_bytesize is the chunk size: a read is one buffer and one copy,
    // with nothing to split or join. The slab stays small and independent of the window - see
    // DeviceWriterClient::Buffers for why a slab covering the window would stall the first read.
    DeviceWriterClient::Buffers buffers;
    buffers.buffer_bytesize = _settings->chunk_bytesize();
    buffers.slab_bytesize = std::max(_settings->chunk_bytesize(), device_slab_bytesize());

    // NOT the window. The window says how many reads may be in flight; the pool must also cover the
    // chunks whose read has landed and whose copy has not, or a copy would cost a read its slot. The
    // two were one number, which meant the only way to give the copy stage room was to read harder -
    // and reading harder is measurably worse past a shallow depth.
    //
    // Zero is a meaningful setting here - the pool is then exactly the window, which is what this
    // worker did before - so the floor is 0 rather than getenv_positive's usual 1.
    _copy_depth = utils::getenv_positive<unsigned>("RUNAI_STREAMER_INTERNAL_DEVICE_COPY_DEPTH",
                                                   DefaultCopyDepth, 0u);

    _device_out = std::make_unique<DeviceWriterClient>(_writer, buffers, _engine->depth() + _copy_depth);

    LOG(INFO) << "Async io worker ready: " << _strategy << ", " << *_settings;

    // Harvest into a buffer sized once - nothing is allocated while completing.
    _completions.resize(_engine->depth());

    // One block-sized buffer per in-flight read, so a bounced pass can always get one and there is no
    // limit to enforce. At depth 512 and a 4096-byte block that is 2 MB. Only a direct strategy ever
    // bounces, so a buffered engine builds none.
    if (posix_io::is_direct(_strategy))
    {
        try
        {
            _scratch = std::make_unique<posix_io::ScratchPool>(
                _engine->depth(), _block);
        }
        catch (const common::Exception &)
        {
            // The engine came up but its scratch did not - an allocation this size failing is a
            // resource condition, not a bug of ours.
            //
            // Answered like an engine that could not be built at all, because that is what it is to
            // everyone outside: this mount is not read asynchronously, and the synchronous pool serves
            // it instead. Without this the throw would reach discard(), which reports _engine_error -
            // still UnknownError here, since the engine itself was fine.
            LOG(ERROR) << "No scratch pool for " << _strategy << " at depth " << _engine->depth()
                       << " and a block of " << _block << " bytes. This mount is read by the"
                       << " synchronous reader from here on";

            engine_is_dead(common::ResponseCode::FsAsyncEngineError);
            throw common::Exception(common::ResponseCode::FsAsyncEngineError);
        }
    }

    return _engine->depth();
}

void AsyncIoWorker::discard(Workload && workload)
{
    // The window never came up, so no chunk of this will ever be read. Its responses are already owed,
    // so finalize it here - dropping it hangs the consumer forever.
    //
    // An empty workload has no batches to report to, and it is why capacity() threw in the first
    // place, so it is reported as such rather than as an engine failure.
    const auto code = workload.batches().empty() ? common::ResponseCode::EmptyRequestError : _engine_error;

    Inflight wl;
    wl.workload = std::move(workload);
    report_workload(wl, code);
}

void AsyncIoWorker::enqueue(Workload && workload)
{
    // Nothing new is read through a dead engine - see abort_all.
    //
    // Reported rather than dropped: these responses are owed whatever state the engine is in, and
    // dropping them hangs the consumer. Reported HERE rather than through discard(), because the base
    // only calls discard() when the window failed to come up, and this window came up long ago.
    if (_engine_dead)
    {
        Inflight wl;
        wl.workload = std::move(workload);
        report_workload(wl, _engine_error);
        return;
    }

    // Adopt a measurement this worker did not have when its engine was built.
    //
    // The first submission may have been unable to probe the mount - every path missing or unreadable
    // - leaving _block provisional. A later submission that could measure carries the answer here.
    // Without this a long-lived streamer would keep the provisional block for the life of the
    // process: wrong for checkpoint restore, and merely invisible for a single model load.
    //
    // A block the scratch pool cannot serve is REFUSED, not adopted.
    //
    // The pool is sized and aligned once, in capacity(), to the block in force when the engine was
    // built. A bounced pass reads a whole block into one of its buffers, so adopting a larger block
    // would have the kernel write past the end of one - and the buffers would no longer be aligned to
    // it either, since the base was aligned to the old block.
    //
    // This used to be argued away: the provisional value is the ladder's top rung, so a measurement
    // could only shrink. That covers the ladder and not statx, which reports whatever the filesystem
    // says and is bounded by nothing here. Today no filesystem reports more than 64 KiB, which is
    // equal to the provisional value rather than below it - no margin at all.
    //
    // Keeping the smaller block is safe. Direct reads on a mount that needs more fail with EINVAL, and
    // demote_to_buffered re-stages them buffered.
    if (!_block_measured && workload.direct_block != 0)
    {
        if (_scratch != nullptr && workload.direct_block > _scratch->block())
        {
            // Once. _block_measured stays false, so this branch is re-entered on every submission and
            // a later one reporting a block this engine CAN serve is still adopted.
            if (!_block_too_large_reported)
            {
                _block_too_large_reported = true;

                LOG(WARNING) << "This mount reports a direct-I/O block of " << workload.direct_block
                             << ", larger than the " << _scratch->block() << " this engine was built"
                             << " for. Keeping the smaller one. The first direct read here will be"
                             << " refused with EINVAL and this engine then reads buffered - still"
                             << " asynchronous, but without O_DIRECT";
            }
        }
        else
        {
            LOG(INFO) << "Adopting a measured direct-I/O block of " << workload.direct_block
                      << " for this mount, replacing the provisional " << _block;

            _block = workload.direct_block;
            _block_measured = true;
        }
    }

    const auto workload_id = _next_workload_id++;

    auto [it, inserted] = _inflight.emplace(workload_id, Inflight{});
    ASSERT(inserted) << "duplicate workload id " << workload_id;

    Inflight & wl = it->second;
    wl.workload = std::move(workload);
    wl.fds.resize(wl.workload.batches().size());   // nothing is opened yet

    size_t queued_chunks = 0;

    // Zero-sized ranges are answered here, before anything is queued: they appear in no chunk, so
    // nothing would ever complete for them.
    for (unsigned batch_index = 0; batch_index < wl.workload.batches().size(); ++batch_index)
    {
        auto & batch = wl.workload.batches()[batch_index];

        // Zero-size tasks appear in no chunk, so nothing will ever complete for them - but a
        // zero-sized range owes a response like any other.
        //
        // The answer depends on whether the FILE can be opened, so the file is checked first. The
        // synchronous reader opens the file for every batch before it looks at any range size, so a
        // zero-sized range in a missing file fails with the file. Both readers must agree: which one
        // served a request is meant to be invisible to the caller.
        //
        // Answering Success without looking was wrong in two ways. A batch whose ranges are ALL
        // zero-sized never opened the file at all, so a missing file reported success. A mixed batch
        // answered its zero-sized ranges Success and its other ranges FileAccessError, for one file.
        const bool has_zero_sized = std::any_of(
            batch.tasks.begin(), batch.tasks.end(),
            [](const auto & task) { return task.info.bytesize == 0; });

        if (has_zero_sized)
        {
            // Only when there IS a zero-sized range, so the usual path keeps its lazy open.
            //
            // Opened and closed rather than kept: fd_for() opens this file later and decides then
            // whether it can use O_DIRECT, which needs a chunk's offset and destination. Keeping this
            // buffered fd would take that decision away from every batch that has a zero-sized range.
            const auto code = probe_open(batch.path);

            if (code == common::ResponseCode::Success)
            {
                for (const auto & task : batch.tasks)
                {
                    if (task.info.bytesize == 0)
                    {
                        common::backend_api::Response resp(common::ResponseCode::Success);
                        batch.handle_response(resp, &task);
                    }
                }
            }
            else
            {
                // Recorded against the FILE, not answered here - the same way every other failure on
                // this path is handled (see complete_chunk). report_workload applies it to every task
                // of this file that is still unanswered when the workload finalizes.
                //
                // Not through handle_response: that THROWS on a non-Success code, because it is the
                // object-storage abort path. Answering an error through it would throw out of
                // enqueue, the workload would never finalize, and the caller would wait for a
                // response that can no longer come.
                wl.error_by_file_index.emplace(batch.file_index, code);
            }
        }

        for (const auto & chunk : batch.chunks)
        {
            const auto id = _chunks.add(chunk, workload_id, batch_index);
            _queue->enqueue(QueuedChunk{ id, chunk.offset, chunk.bytesize, chunk.buffer }, 1);   // cost 1
            ++queued_chunks;
        }
    }

    wl.remaining_chunks = queued_chunks;

    // Nothing to read: every task was zero-sized and has already been answered.
    if (wl.remaining_chunks == 0)
    {
        finalize(it, common::ResponseCode::Success);
    }
}

bool AsyncIoWorker::wants_direct(size_t file_offset, const char * buffer) const
{
    if (!posix_io::is_direct(_strategy))
    {
        return false;
    }

    // This mount already refused an aligned direct read. Opening direct again would cost one failed
    // read per file to learn the same thing - see the EINVAL branch in the completion loop.
    if (_direct_refused)
    {
        return false;
    }

    // Congruence, not alignment. A caller can align its buffer and still be out of step with the file,
    // because the file offset is fixed by the file's own layout. When that happens no part of the
    // region can be read directly, so O_DIRECT would copy every byte on this one thread. Buffered I/O
    // copies too, and the kernel does it, and adds readahead - so buffered wins.
    const auto block = _block;

    if (!posix_io::is_congruent(file_offset, buffer, block))
    {
        LOG(DEBUG) << "Reading buffered: offset " << file_offset << " and destination are not"
                   << " congruent for block " << block << ", so no part of this file could be read"
                   << " directly";
        return false;
    }

    return true;
}

bool AsyncIoWorker::plan_direct_pass(const Chunk & pending, size_t block, DirectPass & out)
{
    const size_t head = pending.offset % block;

    if (head != 0)
    {
        // The cursor is inside a block. Read that whole block into scratch and copy out the part that
        // was asked for - which may be the whole rest of the block, or less if the chunk ends inside
        // it.
        char * scratch = _scratch == nullptr ? nullptr : _scratch->take();
        if (scratch == nullptr)
        {
            return false;
        }

        out.offset = pending.offset - head;
        out.bytesize = block;
        out.buffer = scratch;
        out.scratch = scratch;
        out.skip = head;
        out.wanted = std::min(block - head, pending.bytesize);
        return true;
    }

    if (pending.bytesize >= block)
    {
        // On a boundary with at least a block to read: whole blocks, straight into the destination.
        // This is the case that costs nothing, and it is every chunk in the middle of a region.
        out.offset = pending.offset;
        out.bytesize = pending.bytesize - (pending.bytesize % block);
        out.buffer = pending.buffer;
        return true;
    }

    // On a boundary with less than a block left: the tail. One block into scratch, keep what is owed.
    char * scratch = _scratch == nullptr ? nullptr : _scratch->take();
    if (scratch == nullptr)
    {
        return false;
    }

    out.offset = pending.offset;
    out.bytesize = block;
    out.buffer = scratch;
    out.scratch = scratch;
    out.skip = 0;
    out.wanted = pending.bytesize;
    return true;
}

size_t AsyncIoWorker::bounced_bytes() const
{
    return _bounced_bytes.load(std::memory_order_relaxed);
}

AsyncIoCounters AsyncIoWorker::counters() const
{
    AsyncIoCounters counters;
    counters.bytes_read = _bytes_read.load(std::memory_order_relaxed);
    counters.bounced_bytes = _bounced_bytes.load(std::memory_order_relaxed);
    counters.short_read_restages = _short_read_restages.load(std::memory_order_relaxed);
    counters.achieved_depth = _achieved_depth.load(std::memory_order_relaxed);
    counters.inflight_nanos = _inflight_nanos.load(std::memory_order_relaxed);
    counters.observed_nanos = _observed_nanos.load(std::memory_order_relaxed);
    return counters;
}

size_t AsyncIoWorker::bytes_read() const
{
    return _bytes_read.load(std::memory_order_relaxed);
}

posix_io::FileRef AsyncIoWorker::file_of(const InflightChunk & entry) const
{
    // The file this request was staged against, rebuilt from what the worker already recorded. The
    // engine cannot supply it: a completion carries only the id.
    //
    // An unknown workload gives an empty FileRef. That happens when the workload was finalized while
    // this read was still out, and the result is only a slightly worse log line - the mapping falls
    // back to what a buffered fd would give.
    const auto wlit = _inflight.find(entry.workload_id);
    if (wlit == _inflight.end() || entry.batch_index >= wlit->second.fds.size())
    {
        return posix_io::FileRef{};
    }

    const auto & batch_fd = wlit->second.fds[entry.batch_index];
    return posix_io::FileRef{ batch_fd.fd, batch_fd.direct };
}

size_t AsyncIoWorker::land_bounced_pass(posix_io::RequestId id, size_t bytes_transferred)
{
    auto * entry = _chunks.find_mutable(id);
    ASSERT(entry != nullptr) << "landing a bounced pass for an unknown request " << id;

    if (entry->scratch == nullptr)
    {
        return bytes_transferred;   // not bounced; the bytes are already where they belong
    }

    // What the kernel gave us minus the part we did not ask for. A short read can land inside the
    // skipped head, in which case nothing wanted arrived at all.
    const size_t useful = bytes_transferred <= entry->scratch_skip
                        ? 0
                        : std::min(entry->scratch_wanted, bytes_transferred - entry->scratch_skip);

    if (useful > 0)
    {
        // pending() gives the destination for the cursor, which is where these bytes belong.
        const auto pending = _chunks.pending(id);
        std::memcpy(pending.buffer, entry->scratch + entry->scratch_skip, useful);

        // Counted where the copy happens, so the number is the bytes actually moved rather than an
        // estimate from the request sizes. A pass reads a whole block but copies only what was asked
        // for.
        _bounced_bytes.fetch_add(useful, std::memory_order_relaxed);
    }

    _scratch->give(_chunks.clear_bounce(id));
    return useful;
}

common::ResponseCode AsyncIoWorker::probe_open(const std::string & path)
{
    // Buffered, always. The question here is only whether the file can be opened, and O_DIRECT would
    // add a second reason to fail that has nothing to do with the file existing - some mounts refuse
    // it outright.
    const int fd = utils::Fd::open_for_read(path);
    if (fd < 0)
    {
        LOG(ERROR) << "Failed to open " << path << " : " << std::strerror(errno);
        return common::ResponseCode::FileAccessError;   // this file's failure, not the storage's
    }

    // The type check is inside open_for_read, so it has already run - which is what makes a path answer
    // the same way whether or not its ranges have bytes in them. Without it a directory would be
    // Success for a zero-sized range and FileAccessError for every other range of the same path.
    ::close(fd);
    return common::ResponseCode::Success;
}

void AsyncIoWorker::engine_is_dead(common::ResponseCode code)
{
    if (_engine_dead)
    {
        return;   // the first reason is the real one; later ones are consequences
    }

    _engine_dead = true;
    _engine_error = code;

    // Told AFTER the flag is set, so nothing can route a new workload here in between. The listener
    // only records the mount; it does not touch this worker.
    //
    // Two callers, deliberately sharing this: an engine that could never be built and one that failed
    // later are the same thing to everyone outside - this mount is no longer read asynchronously.
    if (_on_engine_dead)
    {
        _on_engine_dead();
    }
}

common::ResponseCode AsyncIoWorker::demote_to_buffered(const InflightChunk & entry)
{
    const auto wlit = _inflight.find(entry.workload_id);
    if (wlit == _inflight.end() || entry.batch_index >= wlit->second.fds.size())
    {
        // The workload was finalized while this read was out, so there is nothing left to re-stage
        // against, and nothing left to report to either. UnknownError, as before: this says our own
        // bookkeeping and the engine disagree about what is outstanding.
        return common::ResponseCode::UnknownError;
    }

    BatchFd & batch_fd = wlit->second.fds[entry.batch_index];
    const auto & path = wlit->second.workload.batches()[entry.batch_index].path;

    const int fd = utils::Fd::open_for_read(path);
    if (fd < 0)
    {
        // THIS FILE's failure, not ours.
        //
        // It is a surprise - the direct fd worked well enough to be refused for alignment - but every
        // way it happens belongs to the environment: the file was unlinked or renamed since, the
        // process or the node ran out of descriptors, NFS went stale, or the permissions changed.
        //
        // Reporting the original EINVAL instead would map it as a direct-fd alignment failure, which
        // is_internal_error calls ours, and that becomes UnknownError - a code that aborts the whole
        // submission and tells the operator to file a bug against the streamer. For a file that was
        // deleted while we read it.
        LOG(ERROR) << "Cannot reopen " << path << " buffered after a direct read was refused: "
                   << std::strerror(errno);
        return common::ResponseCode::FileAccessError;
    }

    // The old descriptor number is reused, not closed.
    //
    // Some requests of this batch may already be staged. stage() writes the fd number into the
    // submission entry. The kernel reads that number later, when the entry is submitted. So a request
    // staged a moment ago still points at this descriptor.
    //
    // That happens in this very loop. An earlier completion can re-stage its chunk, after a short read
    // or after its own demotion. Those entries wait for the next flush().
    //
    // Closing here would break them. The next flush would fail with EBADF.
    //
    // Leaving the old descriptor open is not enough either. Those requests would run direct again, on
    // a mount that just refused a direct read. Their EINVAL would arrive after batch_fd.direct is
    // false. The code below only re-stages a read that failed on a DIRECT file, so it would skip them.
    // The error would map to FileAccessError and the file would fail.
    //
    // dup2 solves both. It points this number at the buffered open, in one step. Already staged
    // requests then run buffered and land.
    //
    // Their shape is still the direct one - block aligned, and possibly bounced through scratch. A
    // buffered read of that shape is harmless, and land_bounced_pass copies the wanted part out.
    if (batch_fd.fd >= 0)
    {
        if (::dup2(fd, batch_fd.fd) < 0)
        {
            // Nothing has changed yet, so the batch still has its working direct descriptor. Report
            // the original error rather than leaving the batch in a state neither branch describes.
            LOG(ERROR) << "Cannot replace the direct descriptor for " << path << " with a buffered"
                       << " one: " << std::strerror(errno);
            ::close(fd);
            return common::ResponseCode::FileAccessError;
        }

        // batch_fd.fd now refers to the buffered open, so this one is a duplicate.
        ::close(fd);
    }
    else
    {
        batch_fd.fd = fd;
    }

    batch_fd.direct = false;

    // Remembered for the whole mount, not just this file - see _direct_refused. One engine per mount
    // means this worker serves exactly one, so the next file skips the direct open entirely.
    if (!_direct_refused)
    {
        _direct_refused = true;

        LOG(WARNING) << "Direct reads refused with EINVAL on " << path << " at block "
                     << _block << ". This mount needs a larger"
                     << " alignment than we assume, so this engine now reads buffered - still"
                     << " asynchronous, but without O_DIRECT";
    }

    return common::ResponseCode::Success;
}

int AsyncIoWorker::fd_for(Inflight & wl, unsigned batch_index, size_t file_offset, const char * buffer,
                          common::ResponseCode & out_error)
{
    BatchFd & entry = wl.fds[batch_index];

    if (entry.error != common::ResponseCode::Success)
    {
        out_error = entry.error;   // already failed to open; do not try again per chunk
        return -1;
    }

    if (entry.fd >= 0)
    {
        return entry.fd;
    }

    const auto & path = wl.workload.batches()[batch_index].path;

    if (wants_direct(file_offset, buffer))
    {
        entry.fd = utils::Fd::open_for_read(path, O_DIRECT);
        if (entry.fd >= 0)
        {
            entry.direct = true;
            return entry.fd;
        }

        // The mount refused it. Some filesystems have no O_DIRECT at all, and tmpfs accepts the open
        // and fails later. Falling back for THIS FILE only: another file on another mount is
        // unaffected, which is the whole reason the mode is per file rather than per engine.
        LOG(WARNING) << "Cannot open " << path << " with O_DIRECT (" << std::strerror(errno)
                     << "); reading it buffered";
    }

    entry.fd = utils::Fd::open_for_read(path);
    if (entry.fd < 0)
    {
        LOG(ERROR) << "Failed to open " << path << " : " << std::strerror(errno);
        entry.error = common::ResponseCode::FileAccessError;   // this file's failure, not the storage's
        out_error = entry.error;
        return -1;
    }

    entry.direct = false;

    return entry.fd;
}

void AsyncIoWorker::stage_pending(posix_io::RequestId id)
{
    const auto * entry = _chunks.find(id);
    ASSERT(entry != nullptr) << "staging request " << id << " with no in-flight record";

    const auto wlit = _inflight.find(entry->workload_id);
    if (wlit == _inflight.end())
    {
        // Its workload was aborted while this sat in the queue. Nothing left to answer for it; give
        // back the window credit and forget it.
        _queue->complete(1);
        _chunks.release(id);
        return;
    }

    // A device chunk reads into pinned host memory and is copied afterwards. Done before pending() is
    // read, because pending() answers with the staging buffer once there is one.
    if (!entry->staging.valid())
    {
        const auto & batch = wlit->second.workload.batches()[entry->batch_index];
        if (!batch.device.is_host())
        {
            auto error = common::ResponseCode::Success;
            switch (stage_into_device_buffer(id, batch, error))
            {
            case Staged::Yes:
                break;

            case Staged::NoBuffer:
                // Keeps its window slot, so the base admits nothing in its place. resume_waiting()
                // gives it another chance on every turn the worker takes.
                _waiting.push_back(id);
                return;

            case Staged::Failed:
                complete_chunk(id, error);
                return;
            }
        }
    }

    // pending(), not the chunk's own extent: after a short read this is the remainder, resumed where
    // the last pass stopped. On the first pass they are the same.
    //
    // Read BEFORE the open, because the open needs it: whether a direct read is possible depends on
    // this chunk's offset and destination, and the file is opened lazily on the first chunk exactly so
    // that the answer is available in time.
    const auto pending = _chunks.pending(id);

    auto error = common::ResponseCode::Success;
    const int fd = fd_for(wlit->second, entry->batch_index, pending.offset, pending.buffer, error);
    if (fd < 0)
    {
        complete_chunk(id, error);
        return;
    }

    // What the file WAS opened as, not what the strategy wanted. A direct open can fall back per file.
    const bool direct = wlit->second.fds[entry->batch_index].direct;
    const posix_io::FileRef file{ fd, direct };

    // A buffered read is issued exactly as asked. A direct one is cut to what the kernel will accept,
    // which may mean reading one block into scratch and copying the wanted part out.
    auto pass = DirectPass{ pending.offset, pending.bytesize, pending.buffer, nullptr, 0, 0 };

    if (direct && !plan_direct_pass(pending, _block, pass))
    {
        // No scratch was free. Not expected - there is one per in-flight read - but failing the whole
        // read over it would be worse than reading a little less this pass.
        LOG(WARNING) << "No scratch buffer for a direct read at offset " << pending.offset
                     << "; the pool is sized to the window, so this should not happen";
        complete_chunk(id, common::ResponseCode::UnknownError);
        return;
    }

    if (pass.scratch != nullptr)
    {
        _chunks.set_bounce(id, pass.scratch, pass.skip, pass.wanted);
    }

    const auto ret = _engine->stage(id, file, pass.offset, pass.bytesize, pass.buffer,
                                    registration_for(entry->staging, pass));
    if (ret != common::ResponseCode::Success)
    {
        // Not staged, so no completion will arrive for it - this worker has to resolve it. Give the
        // scratch back first: nothing else will, and a leaked buffer drains the pool.
        if (char * scratch = _chunks.clear_bounce(id))
        {
            _scratch->give(scratch);
        }

        complete_chunk(id, ret);
    }
}

void AsyncIoWorker::account_inflight()
{
    // Time-weighted, so the answer is "how many reads were outstanding on average" rather than "how
    // many were outstanding at the moments we happened to look".
    //
    // Called on EVERY change to _issued, before the change is applied - each call closes the interval
    // the previous level lasted for. Missing one does not lose a sample, it mis-attributes a duration
    // to the wrong level, which is worse.
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const uint64_t now = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);

    if (_inflight_since != 0)
    {
        const uint64_t elapsed = now - _inflight_since;
        _inflight_nanos.fetch_add(_issued * elapsed, std::memory_order_relaxed);
        _observed_nanos.fetch_add(elapsed, std::memory_order_relaxed);
    }

    _inflight_since = now;
}

void AsyncIoWorker::submit(const QueuedChunk & chunk)
{
    stage_pending(chunk.id);
}

void AsyncIoWorker::drain_batch(std::atomic<bool> & stopped)
{
    if (stopped)
    {
        abort_all(common::ResponseCode::FinishedError);
        return;
    }

    // Copies that landed while this thread was elsewhere. Blocking only when no read is outstanding:
    // there is then nothing else to wait for, and without it the pool would spin on has_deferred_work.
    drain_copies(_issued == 0);

    // EVERY TURN, not only after a copy reports. A buffer also comes back when a copy could not be
    // issued at all and when a read failed - neither queues a completion - so tying this to
    // drain_copies would leave a read parked with a free buffer in the pool and nothing to wake it.
    resume_waiting();

    // Flush first: the previous pump may have left a backlog, and issuing it before waiting is what
    // stops it sitting a whole loop iteration longer than it must.
    unsigned issued = 0;
    const auto flushed = _engine->flush(issued);
    if (flushed != common::ResponseCode::Success)
    {
        LOG(ERROR) << "Failed to issue staged reads: " << flushed;
        abort_all(flushed);
        return;
    }
    account_inflight();
    _issued += issued;

    // The high-water mark, recorded where the count RISES - the only place it can grow. _issued
    // itself is a live count that falls again, so by the time anyone asks the peak has gone.
    if (_issued > _achieved_depth.load(std::memory_order_relaxed))
    {
        _achieved_depth.store(static_cast<unsigned>(_issued), std::memory_order_relaxed);
    }

    // Block only when the window is FULL, not merely when something is in flight.
    //
    // Two conditions, and they answer different questions. Nothing issued means there is nothing to
    // wait for at all: staged is not issued, so waiting then parks on an empty ring for a completion
    // that cannot arrive. A window that is not yet full means there IS something to wait for, but
    // waiting is still the wrong move - the queue has work and capacity for it, so the thread should
    // go back and submit rather than stop here.
    //
    // Blocking whenever anything was in flight cost exactly that: with a depth of 64 and five reads
    // outstanding, the thread stopped instead of filling the other fifty-nine. It shows up on every
    // workload's ramp-up, and on the drain between requests.
    //
    // This is what fio does on the same engine and one thread: it waits only once in-flight reaches
    // the target depth, and otherwise loops back to staging. It reaches 17 GB/s on a mount where we
    // reached 11.7.
    //
    // Non-blocking still HARVESTS whatever is ready - it just does not sleep - so nothing is delayed
    // by taking this path.
    const bool window_full = _engine != nullptr && _issued >= _engine->depth();
    const auto mode = (_issued > 0 && window_full) ? posix_io::WaitMode::Block
                                                   : posix_io::WaitMode::NonBlocking;

    unsigned count = 0;
    const auto ret = _engine->wait_for_completions(_completions.data(), _completions.size(), count, mode, WaitTimeoutMs);
    if (ret != common::ResponseCode::Success)
    {
        LOG(ERROR) << "Failed to harvest completions: " << ret;
        abort_all(ret);
        return;
    }

    for (unsigned i = 0; i < count; ++i)
    {
        const auto & completion = _completions[i];

        // Decremented HERE, before any path that might drop this completion. quiesce() waits on
        // _issued reaching zero, so a completion that is dropped without decrementing would leave
        // the count permanently high and hang teardown - the fix producing the very hang it exists
        // to prevent. Every completion counts, whether or not it can be routed.
        ASSERT(_issued > 0) << "more completions than were issued";
        account_inflight();
        --_issued;

        const auto * entry = _chunks.find(completion.id);
        if (entry == nullptr)
        {
            // Ids are never reused, so this cannot be a live request - it is a completion for
            // something already resolved or abandoned. Drop it rather than aborting everything else
            // over a stray event.
            LOG(DEBUG) << "Dropping completion for unknown request " << completion.id;
            continue;
        }

        if (completion.failed())
        {
            // Mapped HERE, not in the engine, because mapping needs the file. EINVAL means our
            // alignment rule broke on a direct fd, and means something else on a buffered one
            // (completion_mapper.h). The engine holds only the id when a completion arrives, so it
            // cannot tell the two apart. This worker can: it opened the file and recorded how.
            const auto file = file_of(*entry);

            // Give the scratch back first, whichever way this goes, or the pool drains one buffer per
            // failed read. A buffered re-stage never bounces, so clearing it is right there too.
            if (char * scratch = _chunks.clear_bounce(completion.id))
            {
                _scratch->give(scratch);
            }

            // EINVAL on a DIRECT fd is the kernel refusing our alignment - the mount, or this file,
            // needs a larger block than DirectBlockSize assumes. Nothing was read, so the chunk is
            // untouched and the same bytes can be re-issued buffered.
            //
            // Re-staged rather than failed. This used to become UnknownError, which is the one code
            // that tells a caller to abort the WHOLE submission, so one misaligned range ended a
            // model load. The caller asked for bytes, not for a particular syscall.
            //
            // It is also not necessarily our bug: statx(2) defines both direct-I/O alignments per
            // FILE, so a mount that answered correctly can still be wrong for one file.
            if (file.direct && completion.res == -EINVAL)
            {
                const auto demoted = demote_to_buffered(*entry);
                if (demoted == common::ResponseCode::Success)
                {
                    stage_pending(completion.id);
                    continue;
                }

                // Answered with what the DEMOTION hit, not with the original EINVAL.
                //
                // Mapping the EINVAL here would run it through is_internal_error on a still-direct
                // file, which calls it ours and returns UnknownError - ending the whole submission
                // because one file could not be reopened. demote_to_buffered already knows whether the
                // cause was the file (FileAccessError) or our own bookkeeping (UnknownError).
                complete_chunk(completion.id, demoted);
                continue;
            }

            // Mapped HERE, not in the engine, because mapping needs the file. The engine holds only
            // the id when a completion arrives, so it cannot tell a direct EINVAL from a buffered one
            // (completion_mapper.h). This worker can: it opened the file and recorded how.
            const auto code = posix_io::map_completion(completion.res, file);

            complete_chunk(completion.id, code);
            continue;
        }

        // A bounced pass read a whole block but only part of it was asked for. Converting here, before
        // record(), is what stops the cursor advancing past bytes that never arrived.
        const size_t useful = land_bounced_pass(completion.id, completion.bytes_transferred());
        _bytes_read.fetch_add(useful, std::memory_order_relaxed);

        switch (_chunks.record(completion.id, useful))
        {
        case Progress::Complete:
            complete_chunk(completion.id, common::ResponseCode::Success);
            break;

        case Progress::Eof:
            // Zero further bytes while bytes were still owed: the file is shorter than the caller
            // asked for. Not success, however healthy the read looked.
            complete_chunk(completion.id, common::ResponseCode::EofError);
            break;

        case Progress::Partial:
            // Re-stage the remainder on the same request. The window credit is still held, so this
            // does not go back through the queue - it is the same path submit() takes, resumed.
            //
            // Counted here rather than at the chunk, so it counts PASSES: a chunk answered in three
            // pieces adds two. That is the number that says how much a short read really cost, since
            // each extra pass is another submit and another completion.
            _short_read_restages.fetch_add(1, std::memory_order_relaxed);
            stage_pending(completion.id);
            break;
        }
    }
}

posix_io::Registration AsyncIoWorker::registration_for(const StagingBuffer & staging,
                                                      const DirectPass & pass) const
{
    // Asked here and not left to the engine, so a mount that does not register pays nothing: slab_at()
    // takes the pool's lock, and producing a description only to have it discarded is a lock per read.
    //
    // A bounced pass reads into scratch, not into this buffer, so it has no region to name either.
    if (!_register_buffers || !staging.valid() || pass.scratch != nullptr || _device_out == nullptr)
    {
        return posix_io::Registration();
    }

    const auto slab = _device_out->slab_at(staging.slab);
    if (!slab.valid())
    {
        return posix_io::Registration();
    }

    posix_io::Registration registration;
    registration.base = slab.base;
    registration.bytesize = slab.bytesize;
    registration.id = staging.slab;
    return registration;
}

AsyncIoWorker::Staged AsyncIoWorker::stage_into_device_buffer(posix_io::RequestId id, const Batch & batch,
                                                              common::ResponseCode & error)
{
    StagingBuffer buffer;
    error = _device_out->take(batch.device, buffer);
    if (error != common::ResponseCode::Success)
    {
        LOG(ERROR) << "No staging buffer for " << batch.device << ": " << error;
        return Staged::Failed;
    }

    // Every buffer is out: the window's reads hold theirs, and the copy headroom is spent on chunks
    // waiting for the link. Not an error - the read waits for a copy to retire.
    if (!buffer.valid())
    {
        return Staged::NoBuffer;
    }

    _chunks.set_staging(id, buffer);
    return Staged::Yes;
}

bool AsyncIoWorker::issue_copy(posix_io::RequestId id, const InflightChunk & entry, const Batch & batch)
{
    // chunk.bytesize, not what the last pass moved: every pass landed in this buffer, so the whole
    // extent is here by the time a chunk completes.
    const auto code = _device_out->write(
        batch.device, entry.staging, entry.chunk.bytesize, entry.chunk.buffer,
        [this, id](common::ResponseCode ret) { _copies.push(CopyDone{ id, ret }); });

    if (code != common::ResponseCode::Success)
    {
        LOG(ERROR) << "Failed to enqueue a copy to " << batch.device << ": " << code;
        return false;
    }

    ++_copies_issued;
    return true;
}

void AsyncIoWorker::resume_waiting()
{
    // At most one pass over what is waiting now. stage_pending() parks a read again when the pool is
    // still dry, pushing it to the BACK, so a count taken up front is what stops this looping on the
    // same read forever.
    for (size_t remaining = _waiting.size(); remaining > 0; --remaining)
    {
        const auto id = _waiting.front();
        _waiting.pop_front();
        stage_pending(id);
    }
}

bool AsyncIoWorker::drain_copies(bool wait)
{
    bool any = false;
    CopyDone done;

    while (_copies_issued > 0)
    {
        if (!_copies.try_pop(done))
        {
            // Every issued copy reports exactly once, so a blocking wait here cannot hang. Only taken
            // when there is nothing else to wait for - otherwise the engine's own wait does the
            // sleeping and this returns with whatever had already landed.
            if (!wait || !_copies.pop(done))
            {
                break;
            }
        }

        --_copies_issued;
        complete_copy(done.id, done.ret);
        any = true;
    }

    return any;
}

void AsyncIoWorker::complete_copy(posix_io::RequestId id, common::ResponseCode ret)
{
    if (ret != common::ResponseCode::Success)
    {
        LOG(ERROR) << "Copy to device failed: " << ret;
    }

    // Straight to the accounting: the copy is the last step, and the buffer is already back in the
    // pool - the waiter returns it before this is even queued. The window slot went back when the
    // read landed.
    account_chunk(id, ret, false /* slot already freed */);
}

unsigned AsyncIoWorker::staging_buffers() const
{
    return _device_out != nullptr ? _device_out->buffers() : 0;
}

unsigned AsyncIoWorker::waiting_for_a_buffer() const
{
    return static_cast<unsigned>(_waiting.size());
}

bool AsyncIoWorker::has_deferred_work() const
{
    return _copies_issued > 0 || !_waiting.empty();
}

void AsyncIoWorker::complete_chunk(posix_io::RequestId id, common::ResponseCode ret)
{
    const auto * entry = _chunks.find(id);
    ASSERT(entry != nullptr) << "completing request " << id << " twice";

    // A DEVICE chunk is not done when its read lands: the bytes are in pinned host memory. Issue the
    // copy and let complete_copy() account it, so the caller is told only once the bytes are there.
    if (ret == common::ResponseCode::Success && entry->staging.valid())
    {
        const auto wlit = _inflight.find(entry->workload_id);
        if (wlit != _inflight.end())
        {
            const auto & batch = wlit->second.workload.batches()[entry->batch_index];
            const bool issued = issue_copy(id, *entry, batch);

            // Ours no longer on either path: a copy that was enqueued comes back through the waiter,
            // and a write() that failed returned the buffer itself.
            _chunks.set_staging(id, StagingBuffer{});

            if (issued)
            {
                // The read is done with its slot: the bytes are in pinned host memory and the window
                // means reads in flight. Holding it until the copy retired cost a read its place,
                // which is the whole reason the pool is no longer sized by the window.
                _queue->complete(1);
                return;
            }

            ret = common::ResponseCode::DeviceTransferError;
        }
    }

    account_chunk(id, ret);
}

void AsyncIoWorker::account_chunk(posix_io::RequestId id, common::ResponseCode ret, bool free_slot)
{
    const auto * entry = _chunks.find(id);
    ASSERT(entry != nullptr) << "accounting request " << id << " twice";

    // Still ours only when no copy was issued for it - a failed read, or a workload that went away.
    // A buffer lost here would shrink the pool for the life of the worker.
    if (entry->staging.valid())
    {
        _device_out->release(entry->staging);
    }

    const auto workload_id = entry->workload_id;
    const auto batch_index = entry->batch_index;
    const auto chunk = _chunks.release(id);

    // Free the window slot - once per chunk, never on a re-stage, and never twice: a chunk whose copy
    // was issued gave its slot back when the read landed.
    if (free_slot)
    {
        _queue->complete(1);
    }

    const auto wlit = _inflight.find(workload_id);
    if (wlit == _inflight.end())
    {
        return;   // aborted while this was in flight
    }

    Inflight & wl = wlit->second;
    auto & batch = wl.workload.batches()[batch_index];

    // One read carried every task in the span, so they share its outcome.
    for (unsigned i = 0; i < chunk.task_count; ++i)
    {
        const auto & task = batch.tasks[chunk.first_task + i];

        // Zero-sized tasks were answered at enqueue. They can fall inside a span, and answering one
        // twice is harmless (finished_request is idempotent), but skipping is clearer than relying on
        // that.
        if (task.info.bytesize == 0)
        {
            continue;
        }

        if (ret == common::ResponseCode::Success)
        {
            common::backend_api::Response resp(common::ResponseCode::Success);
            batch.handle_response(resp, &task);
        }
        else
        {
            wl.error_by_file_index.emplace(batch.file_index, ret);   // first error per file
        }
    }

    // Once per chunk, after its tasks are answered. Counting chunks rather than tasks keeps this
    // independent of which tasks the span happened to contain.
    ASSERT(wl.remaining_chunks > 0) << "completing a chunk of a workload with none outstanding";
    if (--wl.remaining_chunks == 0)
    {
        finalize(wlit, common::ResponseCode::Success);   // wl is gone after this
    }
}

void AsyncIoWorker::report_workload(Inflight & wl, common::ResponseCode code)
{
    for (auto & batch : wl.workload.batches())
    {
        auto error = code;
        if (error == common::ResponseCode::Success)
        {
            const auto it = wl.error_by_file_index.find(batch.file_index);
            error = (it == wl.error_by_file_index.end()) ? common::ResponseCode::Success : it->second;
        }
        batch.handle_error(error);
    }
}

void AsyncIoWorker::finalize(InflightMap::iterator it, common::ResponseCode code)
{
    report_workload(it->second, code);

    // Cumulative for this worker, not for this workload: workloads overlap in one window, so bytes
    // cannot be attributed to one of them. Streamer::async_counters() sums this across workers, and is
    // engine-scoped for the same reason.
    LOG(DEBUG) << "Async io worker: " << counters();

    // The number that decides whether direct reads are worth having.
    //
    // Expected near zero: a region copies at most one partial block at each end, so about 0.1% of an
    // 8 MiB region. A large share means destinations are no longer congruent with their file offsets,
    // and then EVERY byte is copied - by this one thread - which is slower than reading buffered.
    //
    // There is no error for this. The reads all succeed. Without this line the only symptom is that
    // the model loads slowly, which is not a symptom anyone can act on.
    // Not judged before enough has been read. A region copies at most one partial block at each end,
    // so a SMALL region is legitimately a large share - a 9 KB region with a 4096 block can be half
    // copied while everything is working exactly as designed. Judging that would warn about nothing on
    // any model with many small tensors.
    //
    // 64 MiB is a few regions at the default chunk size, which is enough for the ratio to mean
    // something.
    constexpr size_t EnoughToJudge = 64ul << 20;

    if (!_warned_about_bouncing && _bytes_read >= EnoughToJudge && _bounced_bytes * 100 > _bytes_read)
    {
        _warned_about_bouncing = true;
        LOG(WARNING) << "More than 1% of bytes read are being copied through a scratch buffer ("
                     << _bounced_bytes << " of " << _bytes_read << "). Destinations are probably not"
                     << " congruent with their file offsets, which makes direct reads copy everything."
                     << " Buffered reads would be faster.";
    }

    for (auto & batch_fd : it->second.fds)
    {
        if (batch_fd.fd >= 0)
        {
            ::close(batch_fd.fd);
        }
    }

    _inflight.erase(it);
}

void AsyncIoWorker::quiesce()
{
    // Wait until the kernel holds none of our destinations.
    //
    // A response promises that nothing will write to that range again. An ISSUED read has its
    // destination inside the kernel, so reporting its range before the completion arrives hands a
    // live write target to whoever gets the buffer next - under the Python ring, the next
    // submission. Reads that were only staged, or still queued, never reached the kernel and need
    // no wait.
    //
    // UNBOUNDED, terminating only on _issued reaching zero. A timeout here would re-open the exact
    // invariant this exists to protect. A wedged mount therefore hangs its own teardown - which is
    // what one engine per mount (5.2.3) is for: it hangs that engine, not every engine.
    // A copy in flight is the same promise: cuMemcpyHtoDAsync is reading out of a staging buffer and
    // writing into the caller's device memory. Reporting its range first hands a live write target to
    // whoever gets that memory next.
    drain_copies(true);

    if (_engine == nullptr || _issued == 0)
    {
        return;
    }

    LOG(INFO) << "Waiting for " << _issued << " reads in flight before reporting";

    while (_issued > 0)
    {
        unsigned count = 0;
        const auto ret = _engine->wait_for_completions(_completions.data(), _completions.size(),
                                                       count, posix_io::WaitMode::Block,
                                                       WaitTimeoutMs);
        if (ret != common::ResponseCode::Success)
        {
            // The engine cannot tell us any more. Waiting longer cannot make it, and looping here
            // would spin - so stop, and accept that the guarantee is only as good as the engine.
            LOG(ERROR) << "Failed to drain " << _issued << " reads in flight: " << ret
                       << ". Reporting anyway; their destinations may still be written";
            break;
        }

        // Reaped, so the kernel is done with these destinations - which is the whole point. NOT
        // routed: the workload is about to be reported with the abort code, so routing would answer
        // its ranges twice, and a short read must not be re-staged or the drain never ends.
        ASSERT(_issued >= count) << "drained " << count << " completions with " << _issued << " issued";
        account_inflight();
        _issued -= count;
    }
}

void AsyncIoWorker::abort_all(common::ResponseCode code)
{
    // A mid-life failure ENDS this engine. It is never used again.
    //
    // Only two things reach here with a code other than FinishedError, and neither recovers:
    // io_uring_submit or io_getevents failing with something that is not backpressure. The ring or the
    // context is gone, and no later call can bring it back.
    //
    // Reusing it would also be unsafe rather than merely futile. Requests already staged in the engine
    // are not issued and not dropped, and they hold descriptor numbers that finalize() below is about
    // to close - numbers the next open on this worker will be handed straight back. A later flush would
    // then read a DIFFERENT file into memory the caller has since reused. Never flushing again is what
    // closes that.
    //
    // FinishedError is the shutdown path and marks nothing: the engine is being destroyed anyway, and
    // io_uring_queue_exit discards whatever is staged in it.
    if (code != common::ResponseCode::FinishedError)
    {
        LOG(ERROR) << "The asynchronous reader for this mount failed permanently (" << code << ")."
                   << " Every submission still in flight here is failed now";

        engine_is_dead(code);
    }

    // Drops pending entries and releases their credit in one step - draining through
    // try_take()/complete() would stop at the full-window boundary. First, so nothing new is staged
    // while we wait.
    if (_queue != nullptr)
    {
        _queue->clear();
    }

    // Then wait, and only then report. The order is the whole fix: report-then-abandon declares
    // buffers free while the kernel may still write to them.
    quiesce();

    while (!_inflight.empty())
    {
        finalize(_inflight.begin(), code);
    }

    // Every buffer still held by an abandoned pass. clear() forgets the chunks, so this must come
    // first or the pool would be empty for the rest of the streamer's life.
    if (_scratch != nullptr)
    {
        _chunks.release_all_scratch([this](char * scratch) { _scratch->give(scratch); });
    }

    if (_device_out != nullptr)
    {
        _chunks.release_all_staging([this](const StagingBuffer & buffer) { _device_out->release(buffer); });
    }

    _chunks.clear();

    // With them, or a parked read would name a record that no longer exists and resume_waiting()
    // would assert on it. They are already answered: their chunks were finalized above.
    _waiting.clear();

    account_inflight();
    _issued = 0;
}

void AsyncIoWorkers::add(const AsyncIoWorker * worker)
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    _workers.push_back(worker);
}

AsyncIoCounters AsyncIoWorkers::total() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);

    AsyncIoCounters total;
    for (const auto * worker : _workers)
    {
        total += worker->counters();
    }
    return total;
}

size_t AsyncIoWorkers::size() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _workers.size();
}

}; // namespace runai::llm::streamer::impl
