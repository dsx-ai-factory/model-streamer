#pragma once

#include <cstddef>
#include <deque>
#include <map>
#include <vector>

#include "posix_io/io_engine/io_engine.h"

namespace runai::llm::streamer::posix_io
{

// A test double for IoEngine: no kernel, no threads, and the test decides what completes and when.
//
// Its own target, marked testonly, so Bazel refuses to link it into anything shipped - a mock in the
// product is a defect, not a convenience.
//
// Three things are otherwise reachable only against a real kernel under load, where they are
// non-deterministic: out-of-order completion, a partial flush(), and a wait that times out.
//
// NOT THREAD SAFE, like the interface it implements. Tests drive it from their own thread.
//
// DRIVEN, not self-completing: a read completes only when the test calls complete(), complete_short()
// or fail(). Behind a live worker thread nothing ever completes and the submission hangs until the
// test times out - there, delegate to make_io_engine and record what it was asked for.
class MockIoEngine : public IoEngine
{
 public:
    explicit MockIoEngine(unsigned depth, Limits limits = Limits{});

    // What the engine was asked to do. Kept so a test can assert on the request itself: a chunk
    // staged at the wrong offset, or against the wrong destination, still completes cleanly.
    struct Request
    {
        RequestId id = 0;
        FileRef   file;
        size_t    offset = 0;
        size_t    bytesize = 0;
        char *    buffer = nullptr;
    };

    // ---- IoEngine ----

    Limits limits() const override;
    unsigned depth() const override;

    // `registration` is recorded rather than used, so a test can assert what a caller offered.
    using IoEngine::stage;   // keeps the no-registration overload visible through this type

    common::ResponseCode stage(RequestId id, FileRef file, size_t offset, size_t bytesize,
                               char * buffer, Registration registration) override;

    // Issues up to the flush limit, oldest first. The rest stays staged - both real APIs issue a
    // prefix, so the unissued set is always the tail.
    common::ResponseCode flush(unsigned & out_issued) override;

    // Returns what the test has completed, in the order it completed it. Never blocks: with nothing
    // ready this is Success and zero for both wait modes, which is also the expired-timeout case - so
    // a test reaches the teardown path by simply completing nothing.
    common::ResponseCode wait_for_completions(Completion * out, unsigned max, unsigned & out_count,
                                      WaitMode mode, unsigned timeout_ms = 0) override;

    void register_memory(char * base, size_t size) override;
    void unregister_memory(char * base) override;

    // ---- what the engine holds ----

    // Asserts if `id` is not live. Live means staged or in flight; completed requests are gone.
    const Request & request(RequestId id) const;

    std::vector<RequestId> staged() const;      // stage order

    // What the last stage() was offered. A mock engine registers nothing, so this is the only way to
    // check that a caller described its buffer at all.
    Registration last_registration() const;
    std::vector<RequestId> in_flight() const;   // issue order

    size_t staged_count() const;
    size_t in_flight_count() const;

    // Every id ever staged, including completed ones - the live views empty out as completions
    // arrive, so this is what a test asserts the full set of requests against.
    const std::vector<Request> & history() const;

    // ---- injection ----

    // Issue at most `n` per flush; 0 means no limit. "Issue nothing" is set_flush_stalled(), so that
    // zero cannot mean two things.
    void set_flush_limit(unsigned n);

    // flush() issues nothing while set. Models io_submit's EAGAIN - including the RWF_NOWAIT kind,
    // which can fire with nothing in flight, so reaping frees nothing and a retry loop livelocks.
    void set_flush_stalled(bool stalled);

    // Make stage() / flush() return this code instead of acting.
    void set_stage_result(common::ResponseCode ret);
    void set_flush_result(common::ResponseCode ret);

    // ---- completion, in the order the test picks ----
    //
    // Each of these moves a request to the ready queue, and wait_for_completions() hands them out in
    // that order. Asserts if the request is not in flight - a kernel cannot complete something it was
    // never given, and allowing it would let a test assert an impossible state.

    void complete(RequestId id);                       // the full requested length
    void complete_short(RequestId id, size_t bytes);   // bytes < requested; 0 models EOF

    // `error` is a POSITIVE errno, such as EIO. The engine reports it as -errno, which is how both
    // io_uring and libaio report a failed read (io_engine.h). Taking the errno rather than a
    // common::ResponseCode keeps the mock at the same level as a real engine: mapping to a common::ResponseCode is
    // the caller's job now, and a mock that mapped for the caller could not test that mapping.
    void fail(RequestId id, long error);
    void complete_all();                               // everything in flight, oldest first

    // ---- destination fill ----

    // When on (the default), a completion reporting N bytes writes N bytes of a pattern derived from
    // the file offset.
    //
    // On by default because a mock that transfers nothing cannot catch the mistakes that matter here:
    // a chunk staged against the wrong destination, or at the wrong offset, completes perfectly well.
    // Deriving the bytes from the offset makes both visible.
    void set_fill(bool fill);

    // Complete up to `per_wait` in-flight requests, oldest first, on each wait_for_completions().
    // 0 (the default) is off.
    //
    // Models a kernel delivering completions to a waiter, which is what a teardown drain waits on.
    // Without it a caller that loops until nothing is in flight spins forever against this mock, and
    // completing from a second thread is not an option - this class is not thread safe.
    void set_auto_complete_on_wait(unsigned per_wait);

    // ---- what the caller asked of the wait ----
    //
    // Recorded rather than acted on, so the rule that matters is assertable without a clock: pass
    // NonBlocking when nothing is issued, or a worker waits on an empty ring for a completion that
    // will never arrive. Making the mock actually sleep would buy no coverage and make the suite
    // timing-dependent.
    WaitMode last_wait_mode() const;
    unsigned last_wait_timeout_ms() const;
    unsigned waits() const;

    // ---- the O_DIRECT rules ----
    //
    // How many direct reads this engine refused because they were not aligned.
    //
    // A direct read must have its file offset, its length AND its buffer address all be multiples of
    // the block size. The kernel returns EINVAL when any of the three is wrong. This engine applies
    // the same rule, so the tests that drive it are checking the alignment maths and not only the
    // bookkeeping around it.
    //
    // This is a CHECK, not a copy of what a block device does. The engine reads no file and moves no
    // page. It tests one documented rule its caller must follow, which keeps it small and keeps it
    // from slowly turning into a second, different kernel.
    //
    // The rule is only applied when the request says `direct`. Buffered reads have no alignment
    // requirement, and a test that leaves Limits at its default has an alignment of 1, where every
    // value is a multiple and the check does nothing.
    size_t misaligned_direct_stages() const;

    // The byte written for `file_offset`, so a test can build what it expects.
    static char pattern(size_t file_offset);

    // Fill `buffer` with the pattern for [offset, offset + bytesize).
    static void fill_expected(char * buffer, size_t offset, size_t bytesize);

 private:
    void ready(RequestId id, long res);

    const unsigned _depth;
    const Limits _limits;

    size_t _misaligned_direct_stages = 0;

    std::map<RequestId, Request> _live;      // staged or in flight, by id
    std::deque<RequestId> _staged;           // stage order
    Registration _last_registration;
    std::deque<RequestId> _in_flight;        // issue order
    std::deque<Completion> _ready;           // the order the test completed them

    std::vector<Request> _history;

    WaitMode _last_wait_mode = WaitMode::NonBlocking;
    unsigned _last_wait_timeout_ms = 0;
    unsigned _waits = 0;

    unsigned _auto_complete_on_wait = 0;     // 0 = off
    unsigned _flush_limit = 0;               // 0 = no limit
    bool _flush_stalled = false;
    bool _fill = true;

    common::ResponseCode _stage_result = common::ResponseCode::Success;
    common::ResponseCode _flush_result = common::ResponseCode::Success;
};

}; // namespace runai::llm::streamer::posix_io
