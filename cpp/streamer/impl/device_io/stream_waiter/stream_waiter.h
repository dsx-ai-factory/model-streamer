#pragma once

#include <functional>
#include <memory>

#include "device/device.h"
#include "streamer/impl/device_io/staging_pool/staging_pool.h"
#include "utils/draining_worker/draining_worker.h"

namespace runai::llm::streamer::impl
{

// Waits for the copies issued on ONE stream, and gives their buffers back.
//
// It BLOCKS on each event rather than polling, which is affordable only because the events carry
// CU_EVENT_BLOCKING_SYNC: a 29 ms wait costs 1% of a core with that flag and 100% without, at the
// same wall time (H200 and B200). Blocking also removes the poll interval.
//
// ONE PER STREAM, so a thread blocked on one stream cannot delay another's completions.
//
// The thread starts on the FIRST enqueue, so a host-only load creates nothing.
class StreamWaiter
{
 public:
    // Called with the copy's result. Success means those bytes are on the device.
    using Completion = std::function<void(common::ResponseCode)>;

    explicit StreamWaiter(std::shared_ptr<device::Device> device);

    // Waits for what is queued, so every buffer is back in its pool before this returns - which a
    // pool's own teardown assumes.
    ~StreamWaiter();

    StreamWaiter(const StreamWaiter &) = delete;
    StreamWaiter & operator=(const StreamWaiter &) = delete;

    // Hand over a buffer whose copy has been issued and whose event recorded, with the pool to
    // return it to. Starts the thread the first time it is called.
    //
    // The pool comes per buffer because one stream serves every worker of its device, and each has
    // its own pool.
    //
    // Call it immediately after event_record: the gap is how long another thread can interleave on
    // the shared stream, and so how long the buffer is held.
    void enqueue(std::shared_ptr<StagingPool> pool, const StagingBuffer & buffer, Completion on_done);

    // Waits for what is already queued, then ends the thread.
    //
    // PRECONDITION: no enqueue may be in flight. A worker stops issuing copies before its waiter is
    // stopped, so nothing pays for a guarantee against a case that cannot arise.
    void stop();

    bool running() const;
    unsigned completed() const;

 private:
    struct Entry
    {
        // Shared, not borrowed: the pool must outlive the copy, whatever its worker does meanwhile.
        std::shared_ptr<StagingPool> pool;
        StagingBuffer buffer;
        Completion on_done;
    };

    void wait_for(Entry && entry);

    const std::shared_ptr<device::Device> _device;

    // Touched only by the worker's own thread, so it needs no synchronisation.
    bool _thread_bound = false;

    // The thread, the queue and the drain-before-stop are all in here. Concurrency belongs in a
    // tested primitive rather than in this class - see general_directions.md.
    utils::DrainingWorker<Entry> _worker;
};

} // namespace runai::llm::streamer::impl
