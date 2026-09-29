/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/io_uring_engine/io_uring_engine.h"

#include <time.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "common/exception/exception.h"
#include "posix_io/alignment/alignment.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

uint64_t now_nanos()
{
    struct timespec ts;

    // CLOCK_MONOTONIC goes through the vDSO, so this is tens of nanoseconds and no syscall.
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

} // namespace

namespace
{

// liburing returns -errno rather than setting errno. Named because `-ret` reads as arithmetic
// everywhere it appears otherwise.
int error_of(int ret)
{
    return -ret;
}

} // namespace

IoUringEngine::IoUringEngine(const AsyncIoConfig & config, size_t max_read_bytesize)
{
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));

    // No flags at all - see the header for what is deliberately absent, IORING_SETUP_CLAMP above all.
    const int ret = io_uring_queue_init_params(config.depth, &_ring, &params);
    if (ret < 0)
    {
        LOG(ERROR) << "io_uring_queue_init(" << config.depth << ") failed: "
                   << std::strerror(error_of(ret));
        throw common::Exception(common::ResponseCode::UnknownError);
    }

    // What the ring REALLY is. io_uring rounds entries up to a power of two, so asking for 700 gives
    // 1024 - and the caller's window is sized from this, so it gets the larger number rather than
    // leaving slots unused.
    _depth = _ring.sq.ring_entries;

    _limits.max_read_bytesize = max_read_bytesize;

    // What a DIRECT read on this host requires. Buffered reads need none of it, but Limits describes
    // the engine, not one request, and the caller only consults these when it is considering a direct
    // read.
    //
    // One shared constant, so routing and this engine cannot disagree - see DirectBlockSize for why
    // the value is assumed rather than measured, and what happens if an engine ever measures it.
    //
    // Reporting 1 here would be much worse than wasteful. The caller tests congruence against this
    // number, and everything is congruent modulo 1 - so every file would be opened with O_DIRECT and
    // every unaligned read would then fail with EINVAL.
    // The MOUNT's measured block when the caller supplied one, else the process-wide default.
    const size_t block = config.direct_block != 0 ? config.direct_block : direct_block_size();

    _limits.offset_alignment = block;
    _limits.buffer_alignment = block;

    LOG(INFO) << "io_uring ready: " << _depth << " submission entries"
              << (_depth == config.depth ? "" : " (rounded up from the configured depth)")
              << ", " << params.cq_entries << " completion entries";
}

IoUringEngine::IoUringEngine(const AsyncIoConfig & config) :
    IoUringEngine(config, max_read_bytesize())
{}

SubmitStats IoUringEngine::submit_stats() const
{
    return _submit_stats;
}

IoUringEngine::~IoUringEngine()
{
    // What submitting cost, logged once, in the same shape libaio uses so the two can be read side
    // by side.
    if (_submit_stats.calls != 0)
    {
        LOG(INFO) << "io_uring submit: " << _submit_stats.calls << " calls carrying "
                  << _submit_stats.requests << " reads, " << _submit_stats.nanos / 1000
                  << " us in total, worst call " << _submit_stats.max_nanos / 1000 << " us";
    }

    // Unmaps the rings and closes the ring fd. Anything still in flight is the caller's failure to
    // quiesce (io_engine.h) - the kernel drops it here, having possibly already written to a
    // destination the caller believes is free.
    io_uring_queue_exit(&_ring);
}

Limits IoUringEngine::limits() const
{
    return _limits;
}

unsigned IoUringEngine::depth() const
{
    return _depth;
}

common::ResponseCode IoUringEngine::stage(RequestId id, FileRef file, size_t offset, size_t bytesize, char * buffer)
{
    struct io_uring_sqe * sqe = io_uring_get_sqe(&_ring);
    if (sqe == nullptr)
    {
        // Unreachable by construction: the caller's window is sized from depth() - the ring's real
        // size - and every staged request holds a credit, so prepared-but-unsubmitted can never
        // exceed the queue. Reaching here means that invariant broke somewhere else.
        //
        // Reported rather than asserted: ASSERT is fatal in every build here, and killing the host
        // process over a condition we have argued cannot happen is worse than a failed range. And
        // NOT flushed-and-retried: an internal submit would put reads in flight that the caller's
        // own accounting cannot see, so its teardown could report while the kernel still holds
        // those destinations (5.7).
        LOG(ERROR) << "io_uring submission queue full at " << _staged << " staged of " << _depth
                   << " - the in-flight window and the ring have disagreed";
        return common::ResponseCode::UnknownError;
    }

    io_uring_prep_read(sqe, file.fd, buffer, bytesize, offset);
    io_uring_sqe_set_data64(sqe, id);

    if (!file.direct)
    {
        // IOSQE_ASYNC on buffered reads, always.
        //
        // Without it io_uring tries the read INLINE in the submitting task, and a page-cache hit is
        // then copied on our one worker - which stages, flushes and reaps for everything. It never
        // shows on a cold read, only on a second pass over the same model, and the benchmark harness
        // drops caches. So the cost would ship unmeasured. Direct reads never punt, so the flag is
        // pointless there.
        io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
    }

    ++_staged;
    return common::ResponseCode::Success;
}

common::ResponseCode IoUringEngine::flush(unsigned & out_issued)
{
    out_issued = 0;

    if (_staged == 0)
    {
        return common::ResponseCode::Success;
    }

    const uint64_t started = now_nanos();
    const int ret = io_uring_submit(&_ring);
    const uint64_t elapsed = now_nanos() - started;

    ++_submit_stats.calls;
    _submit_stats.nanos += elapsed;
    _submit_stats.max_nanos = std::max(_submit_stats.max_nanos, elapsed);

    if (ret < 0)
    {
        const int error = error_of(ret);

        // Backpressure, not failure. The staged entries keep their place at the head of the ring and
        // go out on the next flush, in order - both APIs issue a prefix, so the unissued set is
        // always the tail and nothing has to be tracked.
        //
        // Zero progress is the hazard, so it is REPORTED rather than retried here: only reaping frees
        // capacity, and reaping runs on this same thread, so a loop would spin against itself (5.9).
        if (error == EAGAIN || error == EBUSY)
        {
            LOG(DEBUG) << "io_uring_submit deferred " << _staged << " staged reads: "
                       << std::strerror(error);
            return common::ResponseCode::Success;
        }

        // This RING is gone, and nothing of ours is in doubt - so not UnknownError, which would tell
        // the caller to abort everything and treat it as a bug in the streamer. One engine serves one
        // mount, and that mount is still readable synchronously.
        LOG(ERROR) << "io_uring_submit failed: " << std::strerror(error);
        return common::ResponseCode::FsAsyncEngineError;
    }

    out_issued = static_cast<unsigned>(ret);
    _submit_stats.requests += out_issued;

    ASSERT(out_issued <= _staged) << "io_uring_submit issued " << out_issued << " of " << _staged
                                  << " staged";
    _staged -= out_issued;

    return common::ResponseCode::Success;
}

common::ResponseCode IoUringEngine::wait_for_completions(Completion * out, unsigned max, unsigned & out_count,
                                                 WaitMode mode, unsigned timeout_ms)
{
    out_count = 0;

    if (mode == WaitMode::Block)
    {
        struct io_uring_cqe * cqe = nullptr;
        int ret = 0;

        if (timeout_ms == 0)
        {
            ret = io_uring_wait_cqe(&_ring, &cqe);
        }
        else
        {
            struct __kernel_timespec ts;
            ts.tv_sec = timeout_ms / 1000;
            ts.tv_nsec = static_cast<long long>(timeout_ms % 1000) * 1000000;
            ret = io_uring_wait_cqe_timeout(&_ring, &cqe, &ts);
        }

        if (ret < 0)
        {
            const int error = error_of(ret);

            // An expired wait and an interrupted one are both "nothing arrived", not errors. The
            // first is also the teardown wake-up: no other thread may touch this engine, so this
            // returning is the only way a waiting worker learns it should stop.
            if (error != ETIME && error != EINTR && error != EAGAIN)
            {
                // The ring cannot be waited on any more. Reported as this engine failing rather than
                // as an internal error, for the same reason as the submit path above.
                LOG(ERROR) << "io_uring_wait_cqe failed: " << std::strerror(error);
                return common::ResponseCode::FsAsyncEngineError;
            }
        }
    }

    // Harvest whatever is ready, however the wait ended - a timed-out wait can still have completions
    // that landed while it was returning.
    unsigned head = 0;
    struct io_uring_cqe * cqe = nullptr;

    io_uring_for_each_cqe(&_ring, head, cqe)
    {
        if (out_count == max)
        {
            break;
        }

        Completion & completion = out[out_count];
        completion.id = io_uring_cqe_get_data64(cqe);

        // Passed through as the kernel gave it: bytes when >= 0, minus an errno when < 0. A short
        // read is a small positive number, so it does not look like an error here.
        //
        // NOT mapped to a common::ResponseCode here. Mapping EINVAL correctly needs to know whether the fd
        // was direct, and at this point only the id is available - the CQE carries nothing else, and
        // this engine keeps no per-file state. The caller knows the file, so the caller maps.
        completion.res = cqe->res;

        ++out_count;
    }

    io_uring_cq_advance(&_ring, out_count);

    // A NON-ZERO value here means the kernel DROPPED completions. It is not a "reaping too slowly"
    // signal, which is what this check used to claim.
    //
    // io_cqring_add_overflow() (io_uring/io_uring.c) increments cq_overflow on ONE branch only - the
    // one whose own comment is "we need to drop it on the floor" - and sets IO_CHECK_CQ_DROPPED_BIT
    // with it. The ordinary NODROP spill takes the other branch: it queues the CQE on
    // cq_overflow_list and sets IORING_SQ_CQ_OVERFLOW, leaving this counter untouched. So the benign
    // case never appears here. io_uring_cq_has_overflow() is what reports that one.
    //
    // It should be unreachable. The kernel sizes the CQ at twice the SQ (cq_entries = 2 *
    // sq_entries, and we pass no setup flags), while the worker's window credit caps reads in flight
    // at the engine's depth, which IS sq.ring_entries. At most `depth` completions can be pending
    // against room for 2 * depth. A short-read re-stage keeps the credit it already holds, so it
    // cannot push past the window either.
    //
    // Which is why this is an ERROR rather than a warning: reaching it means that invariant broke,
    // or the kernel could not allocate under memory pressure. Both lose a read that nothing will
    // ever account for - _issued never decrements for it, and quiesce() waits on _issued reaching
    // zero.
    //
    // Cumulative, and the kernel never resets it. Report only what is new: the worker reaps in a
    // loop, so testing against zero would log this on every pass for the rest of the run.
    if (_ring.cq.koverflow != nullptr && *_ring.cq.koverflow > _overflow_reported)
    {
        LOG(ERROR) << "io_uring DROPPED " << (*_ring.cq.koverflow - _overflow_reported)
                   << " completions (" << *_ring.cq.koverflow << " since this ring was created)"
                   << " - those reads will never be answered";
        _overflow_reported = *_ring.cq.koverflow;
    }

    return common::ResponseCode::Success;
}

}; // namespace runai::llm::streamer::posix_io
