#pragma once

#include <libaio.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "posix_io/io_engine/io_engine.h"

namespace runai::llm::streamer::posix_io
{


// IoEngine over libaio.
//
// The fallback engine. io_uring is faster, but container runtimes block io_uring_setup in their
// default seccomp profile, and hardened platforms turn it off. libaio is not in that blocked set -
// measured, io_setup returns 0 both with and without seccomp. So this is the engine that runs on
// hosts where io_uring cannot (design 5.7).
//
// Direct reads only. libaio is asynchronous only with O_DIRECT. A buffered read inside io_submit runs
// inline, so one thread would serve one file at a time, and the path it would replace is a 16-thread
// pool of preads with kernel readahead. The Strategy enum has no LibaioBuffered value, so there is
// nothing for this class to reject at runtime.
//
// NOT THREAD SAFE, like the interface. One worker owns it and makes every call.
//
// Deliberately absent:
//
//   RWF_NOWAIT       it is unreliable on NFS and virtiofs, where a filesystem that does not
//                    implement it falls back to blocking without saying so - and those are the
//                    mounts where the blocking actually lasts. It also does not make io_submit
//                    non-blocking; faulting pages, ring-lock contention and allocation under memory
//                    pressure all survive it. Its EAGAIN would also break the no-deadlock argument in
//                    5.9, because it means "this read would have blocked", which reaping does not
//                    relieve. See SubmitStats for what is measured instead.
//
//   io_cancel        measured: it returns EINVAL for a direct read, and the read then completes
//                    normally anyway. So there is nothing to cancel, and the drain is required
//                    whatever we do. Teardown is quiesce-then-report in the worker, which needs
//                    nothing from the engine - see io_engine.h.
//
//   registered memory   libaio has no equivalent. The base class no-ops cover it.
class LibaioEngine : public IoEngine
{
 public:
    // Builds the aio context, or throws common::Exception if this host cannot. Callers should go
    // through make_io_engine(), which turns the failure into nullptr.
    //
    // The context can come back smaller than config.depth. /proc/sys/fs/aio-max-nr is node wide and
    // other pods draw on it, so io_setup may refuse the size we ask for. The constructor then halves
    // and retries, and depth() reports what was really granted.
    //
    // Handled here rather than later, because the symptom is misleading. A context smaller than the
    // configured depth means that depth never happens, and all anyone sees is a slow read - which
    // looks like a slow disk, not like a limit being hit (design 5.9.1).
    LibaioEngine(const AsyncIoConfig & config, size_t max_read_bytesize);
    explicit LibaioEngine(const AsyncIoConfig & config);

    ~LibaioEngine() override;

    Limits limits() const override;

    // The granted in-flight capacity, after any clamp. The caller sizes its window from this, never
    // from AsyncIoConfig::depth, so the window can never exceed what the context holds.
    unsigned depth() const override;

    // `registration` is ignored: libaio has no registered-buffer concept, so every read is ordinary.
    using IoEngine::stage;   // keeps the no-registration overload visible through this type

    common::ResponseCode stage(RequestId id, FileRef file, size_t offset, size_t bytesize,
                               char * buffer, Registration registration) override;
    common::ResponseCode flush(unsigned & out_issued) override;
    common::ResponseCode wait_for_completions(Completion * out, unsigned max, unsigned & out_count,
                                      WaitMode mode, unsigned timeout_ms = 0) override;

    // For tests, and for the line logged when this engine is destroyed. Not carried up to
    // AsyncIoStats yet: that path is built once for every worker counter (plan S8b).
    SubmitStats submit_stats() const override;

 private:
    io_context_t _ctx = nullptr;

    // One iocb per in-flight read, and a free list over them.
    //
    // libaio does not copy the iocb. The kernel keeps our pointer until the read completes, so the
    // struct has to outlive the submit. io_uring copies the SQE into the ring, which is why
    // IoUringEngine keeps no per-request storage at all.
    //
    // Sized once at construction and never resized, so the pointers the kernel holds stay valid. A
    // completion returns its own iocb through io_event.obj, so reclaiming is exact.
    std::vector<struct iocb>   _iocbs;
    std::vector<struct iocb *> _free;

    // Prepared and not yet accepted by the kernel.
    //
    // io_submit takes an array and accepts a prefix of it, so whatever did not go out is always the
    // tail - measured, a bad fd at index 1 of 3 makes it return 1. flush() drops the accepted prefix,
    // so the unissued head stays at _pending[0] and nothing has to record which reads failed to go out
    // (design 5.9).
    std::vector<struct iocb *> _pending;

    // Reused across calls so a reap allocates nothing. Sized to the granted depth, which is the most
    // that can ever be outstanding.
    std::vector<struct io_event> _events;

    // Reads the kernel refused to accept, waiting to be handed back as completions.
    //
    // libaio checks the file descriptor when a read is submitted, and reports everything else later
    // as a completion. Measured: a bad fd makes io_submit return -EBADF and a directory fd makes it
    // return -EINVAL, while a misaligned O_DIRECT read completes with -EINVAL, the same as it does
    // under io_uring. So only fd problems arrive here.
    //
    // Such a read cannot stay in _pending. It is the head of the queue, and it would fail the same way
    // on every flush, so nothing behind it would ever go out and the caller would wait for a
    // completion that cannot come. Failing the whole flush is no better: the worker answers a failed
    // flush by aborting every read on this engine, so one bad file would kill the workload.
    //
    // So the engine takes that read out of the queue and reports it in the form the caller already
    // understands - one completion carrying minus the errno. stage() promises a completion for
    // everything it accepts, and this is how the promise is kept. These are the only completions this
    // engine writes itself; every other one comes from the kernel.
    //
    // Handed out before anything is reaped. Otherwise a blocking wait with nothing in flight would sit
    // for its whole timeout while a completion was already waiting here.
    std::vector<Completion> _submit_failures;

    SubmitStats _submit_stats;

    Limits _limits;
    unsigned _depth = 0;
};

}; // namespace runai::llm::streamer::posix_io
