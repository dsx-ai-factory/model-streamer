#pragma once

#include <liburing.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "posix_io/io_engine/io_engine.h"

namespace runai::llm::streamer::posix_io
{

// IoEngine over io_uring.
//
// NOT THREAD SAFE, like the interface. One worker owns it and makes every call - which is what keeps
// this free of locks and what the ring's user-space head/tail state needs anyway. The SQ and CQ sides
// *can* be split across two threads (io_uring supports it), but our state cannot: see design 5.2.4
// for why that split is rejected.
//
// Deliberately absent:
//
//   IORING_SETUP_SQPOLL   a busy-polling kernel thread per ring, for little gain once submissions are
//                         batched by stage/flush (5.7)
//   IORING_SETUP_CLAMP    the one flag that could hand back a ring SMALLER than asked for, which is
//                         the only way the window could exceed the queue (5.7). Without it the kernel
//                         rounds up or refuses.
//   fixed files           ~64 us/s saved at our request rate, against real machinery (5.4)
//   cancellation          teardown is quiesce-then-report in the worker; see io_engine.h
//
// REGISTERED BUFFERS ARE PRESENT, but only for memory a caller offers with a Registration - which in
// practice means a staging pool's slab. Caller destinations are still never registered: they belong to
// the caller, and the Python ring frees them out from under us (5.10).
class IoUringEngine : public IoEngine
{
 public:
    // Builds the ring, or throws common::Exception if this host cannot. Callers should go through
    // make_io_engine(), which consults IoUringProbe first and turns a failure into nullptr.
    IoUringEngine(const AsyncIoConfig & config, size_t max_read_bytesize);
    explicit IoUringEngine(const AsyncIoConfig & config);

    ~IoUringEngine() override;

    Limits limits() const override;

    // The ring's REAL size, which is not always what was asked for: io_uring rounds entries up to a
    // power of two. The caller's window is sized from this, never from AsyncIoConfig::depth, so the
    // window can never exceed the queue.
    unsigned depth() const override;

    using IoEngine::stage;   // keeps the no-registration overload visible through this type

    common::ResponseCode stage(RequestId id, FileRef file, size_t offset, size_t bytesize,
                               char * buffer, Registration registration) override;
    common::ResponseCode flush(unsigned & out_issued) override;
    common::ResponseCode wait_for_completions(Completion * out, unsigned max, unsigned & out_count,
                                      WaitMode mode, unsigned timeout_ms = 0) override;

    // Time spent inside io_uring_submit. Reported for the same reason libaio reports it, and so the
    // two can be compared: io_uring's submit is described as a ring append rather than work done
    // inline, and that is a claim worth checking rather than assuming.
    SubmitStats submit_stats() const override;

    // Regions this ring has registered. Diagnostics, and the only way a test can tell a fixed read
    // from an ordinary one - both return the same bytes, so a test without this would pass whether or
    // not the path under test ran.
    unsigned registered_regions() const;

    // Regions the kernel REFUSED. Counted because a refusal costs speed and nothing else, so nothing
    // else would ever show it: the reads still succeed, and a run that quietly registered none would
    // look exactly like a run that registered all of them.
    unsigned refused_regions() const;

 private:
    // The buffer index to read through, or -1 for an ordinary read. Registers the region on first
    // sight, and remembers a refusal so the kernel is asked once and not once per read.
    int registered_index(const Registration & registration);

    // How many regions the table holds. A SPARSE table is created once at this size and slots are
    // filled in as slabs appear, because re-registering the whole set on every growth costs time
    // proportional to the WHOLE pool, where a single-slot update is a fixed and much smaller cost.
    // See design_io_uring_registration.md.
    //
    // A staging pool cuts slabs of tens of MiB, so this is far more than any pool reaches. An id past
    // it reads the ordinary way rather than failing.
    static constexpr unsigned MaxRegisteredRegions = 1024;

    // What the kernel has been told about each region, indexed by Registration::id. Grown on demand,
    // so a run that registers nothing carries nothing.
    enum class RegionState : unsigned char { Unknown, Registered, Refused };
    std::vector<RegionState> _regions;

    // The sparse table exists. Created on the first region offered, not at construction, so a load
    // that never reads into registered memory makes no registration syscall at all.
    bool _table = false;

    // This ring may register. Read from the probe once: false means never ask, because the answer
    // cannot change while we run and a refused registration costs a syscall per attempt.
    bool _fixed_buffers = false;

    // Prepared with io_uring_get_sqe() and not yet handed to io_uring_submit(). The kernel has not
    // seen these; only flush() makes them real.
    unsigned _staged = 0;

    // Last value seen in cq.koverflow, which counts completions the KERNEL DROPPED (see the reap
    // loop for why that is not the same as the benign NODROP spill). The counter only grows and is
    // never reset, so remembering it keeps the report to one line per new drop instead of one per
    // reap.
    //
    // unsigned, not our usual uint64_t for a growing counter: it mirrors the kernel's __u32
    // cq_overflow, and a wider type here would only invite a comparison against a different width.
    unsigned _overflow_reported = 0;

    SubmitStats _submit_stats;

    struct io_uring _ring;
    Limits _limits;
    unsigned _depth = 0;
};

}; // namespace runai::llm::streamer::posix_io
