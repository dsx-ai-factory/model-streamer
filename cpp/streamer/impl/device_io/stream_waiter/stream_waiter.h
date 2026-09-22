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
// It BLOCKS on each event rather than polling. That is only affordable because the events carry
// CU_EVENT_BLOCKING_SYNC: waiting on a 29 ms copy costs 1% of a core with that flag and 100%
// without, at the same wall time (measured on an H200 and a B200). Blocking also removes the poll
// interval, so a completion is reported the moment it lands rather than on the next tick.
//
// ONE PER STREAM, not one per reader. A thread blocked on one stream cannot delay another stream's
// completions, which is what makes blocking safe here.
//
// The thread starts on the FIRST enqueue. Nothing is created for a load whose destinations are all
// host memory, because such a load never issues a copy and so never enqueues.
class StreamWaiter
{
 public:
    // Called with the copy's result. Success means those bytes are on the device.
    using Completion = std::function<void(common::ResponseCode)>;

    StreamWaiter(std::shared_ptr<device::Device> device, std::shared_ptr<StagingPool> pool);

    // Waits for what is queued, then ends the thread, so every buffer is back in the pool before
    // this returns - the pool's own teardown assumes exactly that.
    ~StreamWaiter();

    StreamWaiter(const StreamWaiter &) = delete;
    StreamWaiter & operator=(const StreamWaiter &) = delete;

    // Hand over a buffer whose copy has been issued and whose event has been recorded. Starts the
    // thread the first time it is called.
    //
    // Call it immediately after event_record and nothing in between: the window where another
    // thread can interleave on a shared stream is what decides how long a buffer is held.
    void enqueue(const StagingBuffer & buffer, Completion on_done);

    // Waits for what is already queued, then ends the thread.
    //
    // PRECONDITION: no enqueue may be in flight. The engine stops issuing copies before it stops
    // its waiter, which every caller does anyway, so nothing pays for a guarantee against a case
    // that cannot arise.
    void stop();

    bool running() const;
    unsigned completed() const;

 private:
    struct Entry
    {
        StagingBuffer buffer;
        Completion on_done;
    };

    void wait_for(Entry && entry);

    const std::shared_ptr<device::Device> _device;

    // Shared, not borrowed: this returns buffers to the pool, so the pool must outlive it. Holding
    // a reference would leave that to whoever declared the two, which is the kind of rule a
    // reordered member breaks in silence.
    const std::shared_ptr<StagingPool> _pool;

    // Touched only by the worker's own thread, so it needs no synchronisation.
    bool _thread_bound = false;

    // The thread, the queue and the drain-before-stop are all in here. Concurrency belongs in a
    // tested primitive rather than in this class - see general_directions.md.
    utils::DrainingWorker<Entry> _worker;
};

} // namespace runai::llm::streamer::impl
