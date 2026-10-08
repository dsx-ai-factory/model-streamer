#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

#include "posix_io/strategy/strategy.h"
#include "common/response_code/response_code.h"

namespace runai::llm::streamer::posix_io
{

// Caller-assigned id, echoed back on the completion. Opaque to the engine: it is carried in
// sqe->user_data / iocb->aio_data and never interpreted, so the engine imposes no range on it.
//
// 64 bits because the caller's ids only ever increase - they are never reused, so that a completion
// arriving after its request was abandoned finds nothing rather than landing on whatever now holds
// that id (InflightChunks). A narrower counter would eventually wrap and re-create exactly that
// aliasing. Object storage types its request id the same way (ObjectRequestId_t).
using RequestId = uint64_t;

// The file, as the engine sees it. A plain value: the fd belongs to whoever opened it, never to the
// engine.
//
// `direct` travels with each request because the engine keeps no per-file state, and two things need
// it: IOSQE_ASYNC must be set on buffered reads and not on direct ones, and -EINVAL means an
// alignment bug only on a direct fd.
//
// A struct, not a bare fd, so a later field is not a signature change.
struct FileRef
{
    int  fd = -1;
    bool direct = false;   // as opened, after any fallback
};

// Host memory a caller offers for REGISTRATION with the kernel, so reads into it need no per-read page
// mapping. What it is worth differs by file system, and is a CPU saving rather than a bandwidth one.
//
// `id` names this region on every read into it. THE SAME id MUST ALWAYS MEAN THE SAME MEMORY: an engine
// registers the region once under that number and trusts the number afterwards. A staging pool's slab
// index is one such number.
//
// Advisory in both directions. An engine that does not register ignores it, and an engine that does may
// still decline - the kernel can refuse, RLIMIT_MEMLOCK above all. Either way the read is served the
// ordinary way, so a caller never has to handle registration failing.
//
// Default-constructed means "ordinary read", which is what every caller without such memory passes.
struct Registration
{
    void *   base = nullptr;
    size_t   bytesize = 0;
    unsigned id = 0;

    bool valid() const { return base != nullptr; }
};

// What the engine imposes; AsyncIoConfig is what the caller chooses. Read once, at construction.
struct Limits
{
    // Largest single read. Shared by pread, io_prep_pread and IORING_OP_READ. Going over is not an
    // error - the kernel short-reads, and the caller re-stages the rest.
    size_t max_read_bytesize = 0;

    // Advisory, and mirrors stx_dio_offset_align - which governs transfer LENGTH as well as offset.
    // O_DIRECT constrains three things but the kernel reports only two, so there is no third field.
    size_t offset_alignment = 1;   // 1 = no constraint, block size for O_DIRECT
    size_t buffer_alignment = 1;   // 1 = no constraint
};

struct Completion
{
    RequestId id = 0;

    // The kernel's own result for this read, passed through without being interpreted:
    //
    //   res >= 0   the number of bytes that were read
    //   res <  0   minus an errno, so -22 is EINVAL
    //
    // io_uring and libaio both report a result this way, so the engines have nothing to convert.
    //
    // RAW, and not a common::ResponseCode, because the engine cannot map it correctly on its own. Mapping
    // needs to know whether the fd was opened with O_DIRECT: EINVAL means an alignment bug on a
    // direct fd and something else on a buffered one (completion_mapper.h). At completion time the
    // engine holds only the id, and it keeps no per-file state - see FileRef above. The caller has
    // the FileRef, so the caller calls map_completion().
    //
    // The caller MUST also compare a positive res against what it asked for. Short reads are routine
    // under io_uring, so treating "no error" as "all bytes arrived" truncates tensors silently.
    long res = 0;

    // Bytes read, or 0 for an error. A short helper because res has two meanings and every caller
    // needs this one of them.
    size_t bytes_transferred() const { return res > 0 ? static_cast<size_t>(res) : 0; }

    bool failed() const { return res < 0; }
};

enum class WaitMode { NonBlocking, Block };

// How long an engine spent inside its submission call - io_submit for libaio, io_uring_enter for
// io_uring.
//
// The design accepts a blocking submit instead of using RWF_NOWAIT (5.8.1). These numbers are how we
// check that choice. If they stay small, nothing more is needed. If one call takes milliseconds, the
// answer is a submit thread, and these are the evidence for it.
//
// How to read them:
//
//   nanos / requests   the normal cost of submitting one read
//   max_nanos          the worst single call
//   requests / calls   the average batch size, which says whether stage-then-flush is buying anything
//
// The total alone would mix two different things. Submission always costs something, and that cost
// grows with the number of reads in the call. A stall looks different: one call far longer than the
// per-read cost. That is why the worst call is kept separately.
//
// REPORTED BY BOTH ENGINES, on purpose. libaio's io_submit is known to block - it waits on filesystem
// metadata, on extent lookup, on faulting the destination, or on a busy block layer. io_uring's
// submit is described as a ring append, which should make it far cheaper. That is a claim about the
// kernel, and measuring only the engine we already distrust cannot check it: without the same
// counters on both, a batch size tuned on libaio evidence would be applied to io_uring on faith.
struct SubmitStats
{
    uint64_t calls = 0;        // submission calls made
    uint64_t requests = 0;     // reads those calls carried
    uint64_t nanos = 0;        // total time inside the submission call
    uint64_t max_nanos = 0;    // the slowest single call
};

// What the operator chooses. Both fields are clamped to their real ceiling when the engine is built,
// and logged if clamped.
struct AsyncIoConfig
{
    // In-flight requests, PER PROCESS. Configured node-wide and divided by the process group size:
    // with one worker per process, n processes on a node make the device see n x depth.
    unsigned depth = 0;

    // Bytes per request. Any large power of two satisfies offset_alignment. Not the synchronous
    // reader's block size: that 2 MiB floor suits a reader wanting fewer, larger reads.
    size_t chunk_bytesize = 0;

    // The direct-I/O block this engine's MOUNT requires, measured by MountCapabilities. 0 means no
    // file on it could be probed, and the engine keeps the process-wide default.
    //
    // Carried here rather than looked up, because one engine serves one mount and the number is
    // fixed for its life - and because the engine has no way to ask: it never sees a path.
    //
    // Measured because a block larger than the mount needs is not free. It pads more between ranges
    // and bounces more of every chunk's head and tail; on NFS under the chunks policy a 64 KiB block
    // against a 4 KiB mount cost 2.4x the load time.
    size_t direct_block = 0;

    // Whether reads on this MOUNT should use registered buffers when the caller offers them.
    //
    // Per mount, because it is worth different amounts on different file systems, and on some it is a
    // LOSS: where the data arrives over the network there is no device DMA into our pages for
    // registration to streamline, so it only adds bookkeeping.
    //
    // Defaults to false so a caller that says nothing gets the behaviour it had before.
    bool register_buffers = false;
};

// The asynchronous I/O engine: io_uring or libaio, direct or buffered.
//
// Named IoEngine, not Engine - vLLM and SGLang are "engines" too, and this is fio's term
// (--ioengine=io_uring / libaio).
//
// The synchronous reader is NOT one of these: stage/flush/wait model a submission and a completion
// it does not have. It stays as it is, on its own pool.
//
// NOT THREAD SAFE. One worker owns an engine and makes every call on it - which is what keeps this
// free of locks, and what io_uring's ring state needs anyway.
class IoEngine
{
 public:
    virtual ~IoEngine() = default;

    virtual Limits limits() const = 0;
    virtual unsigned depth() const = 0;   // in-flight capacity, after clamping

    // Stage one read of [offset, offset + bytesize) from `file` into `buffer`. May not issue a
    // syscall - see flush().
    //
    // `file.fd` must stay open and `buffer` valid until this id's completion is seen. `id` must not
    // be one already in flight; nothing else is required of it - the in-flight bound is the caller's,
    // enforced by its window rather than by any table sized here.
    //
    // `bytesize` is both "bytes wanted" and "bytes you may write": never write past
    // buffer + bytesize. An unaligned length is bounced, not rounded up, or the kernel writes a whole
    // block and overruns a destination sized exactly to the range.
    //
    // A non-Success return means nothing was staged and no completion will arrive; the caller has to
    // resolve the request itself.
    //
    // `registration` describes `buffer` for an engine that can read into registered memory.
    virtual common::ResponseCode stage(RequestId id, FileRef file, size_t offset, size_t bytesize,
                                       char * buffer, Registration registration) = 0;

    // For the callers with no registered memory to offer, which is most of them.
    //
    // An OVERLOAD rather than a default argument: a default is resolved from the static type, so one
    // declared here would be invisible through a derived pointer - which is how every test holds its
    // engine. Overriding classes need `using IoEngine::stage;` to keep this one visible.
    common::ResponseCode stage(RequestId id, FileRef file, size_t offset, size_t bytesize, char * buffer)
    {
        return stage(id, file, offset, bytesize, buffer, Registration());
    }

    // Issue what is staged, in as few syscalls as possible, and report how many went out.
    //
    // out_issued counts the reads a COMPLETION WILL ARRIVE FOR, which is not always the same as the
    // reads the kernel accepted. A read the kernel refuses outright still owes the caller an answer,
    // and an engine that reports it as a completion must count it here too.
    //
    // That is the invariant the caller depends on: it adds out_issued to its in-flight count and
    // subtracts one per completion reaped, so every completion must have been counted exactly once.
    // An undercount is silent - quiesce() waits for that count to reach zero before reporting, so it
    // can declare a destination free while the kernel still holds it.
    //
    // Whatever could not be issued stays staged and is retried, in order, on the next call. Never
    // blocks or spins waiting for capacity: only reaping completions frees capacity, and reaping runs
    // on this same thread, so spinning here deadlocks.
    //
    // So out_issued matters, and zero progress is backpressure, not an error.
    virtual common::ResponseCode flush(unsigned & out_issued) = 0;

    // Collect completions into a caller-owned array - nothing is allocated here.
    //
    // Waits for at least one only when mode == Block, and then at most timeout_ms (0 = forever,
    // matching SharedQueue::pop).
    //
    // An expired timeout is Success with out_count == 0, not an error - the same shape as finding
    // nothing ready. It is also the teardown wake-up: no other thread may touch the engine, so this
    // returning is the only way a waiting worker learns it should stop.
    virtual common::ResponseCode wait_for_completions(Completion * out, unsigned max, unsigned & out_count,
                                              WaitMode mode, unsigned timeout_ms = 0) = 0;

    // NO CANCELLATION, deliberately - there is no cancel_all() here.
    //
    // A response promises that nothing will write to that range's destination again. An issued read
    // has that destination inside the kernel, so abandoning it and reporting the range hands a live
    // write target to whoever gets the buffer next - and under the Python ring that is the next
    // submission.
    //
    // So teardown is quiesce-then-report: abort what was never issued, WAIT for what was, report
    // last. The caller does that; it needs nothing from the engine, which is why cancellation buys
    // nothing. IORING_OP_ASYNC_CANCEL would only shorten the wait, and libaio cannot cancel at all
    // once a request is with the driver - so having it would mean two engines with different
    // teardown semantics, for no change in what the caller must do.

    // Optional: pin a long-lived region once instead of per I/O (io_uring registered buffers).
    // libaio has no equivalent, hence the no-op default.
    //
    // For engine- or streamer-owned memory ONLY - never a caller's destination, and never the Python
    // ring buffer pool. Python allocates and frees that pool and C++ cannot veto the free, while the
    // kernel keeps pinned pages alive after the mapping goes; freeing a still-registered region
    // corrupts the next allocation. Draining I/O does not help - it is a different hazard.
    // Time spent submitting; see SubmitStats. Read at teardown, so returning a copy costs nothing.
    //
    // Defaulted rather than pure so a test double does not have to measure anything - an engine that
    // reports zeros is saying "not measured", which is exactly true of a mock.
    virtual SubmitStats submit_stats() const { return {}; }

    virtual void register_memory(char * /* base */, size_t /* size */) {}
    virtual void unregister_memory(char * /* base */) {}
};

// Largest read the kernel will do in one go: INT_MAX rounded down to a page.
//
// Computed, never hard-coded as 0x7FFFF000 - that is the 4 KiB value, and a 64 KiB-page kernel
// (RHEL/SUSE aarch64) caps 60 KiB lower, so the constant would be a clamp that does not clamp.
//
// Takes the page size so a test can check the 64 KiB case. Every machine we test on has 4 KiB pages,
// where the formula and the constant look identical - measured: `return 0x7FFFF000;` passed the whole
// suite before this was split in two.
size_t max_read_bytesize(size_t page_size);

// This host's ceiling: max_read_bytesize(sysconf(_SC_PAGESIZE)).
size_t max_read_bytesize();

}; // namespace runai::llm::streamer::posix_io
