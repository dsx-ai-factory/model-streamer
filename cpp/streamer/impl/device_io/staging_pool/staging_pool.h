#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "device/device.h"
#include "device/owned/owned.h"

namespace runai::llm::streamer::impl
{

// A pinned host buffer on loan from a pool.
//
// `event` marks the end of the copy OUT of this buffer, and nothing else. Whoever issues the copy
// records it; whoever reaps queries it. It is never re-recorded to mark later work - doing that
// holds the buffer until the later work finishes, which halves the depth the same memory supports.
//
// A COPY, never a reference into the pool. The pool's buffer vector grows, and growth reallocates.
struct StagingBuffer
{
    char *      data = nullptr;
    size_t      bytesize = 0;
    device::EventHandle event = nullptr;
    unsigned    index = 0;      // position in the pool, so release() needs no search

    bool valid() const { return data != nullptr; }
};

// Pinned host buffers for one worker, created on demand and reused for the streamer's lifetime.
//
// Buffers are not created one at a time. The pool registers a SLAB and cuts buffers out of it,
// because creation carries a fixed per-call cost: on a B200, 16 MiB registers at 3.43 GB/s against
// 3.94 for 64 MiB.
//
// Buffers are ANONYMOUS. Nothing ties buffer i to chunk i, so reads completing out of order need no
// handling here - any free buffer serves any read. Keying a buffer to a chunk id, as
// InstantTensor's `chunk_id % io_depth` does, would make the oldest chunk the one whose buffer is
// reused and block buffers that are already free.
//
// ONE CONSUMER, ONE PRODUCER: try_acquire() from one thread, release() from another. That is why
// the free list is two atomics rather than a lock.
//
// The streamer uses SharedStagingPool instead, because it cannot know in advance how many devices a
// run will touch: one waiting thread per stream means one producer per device, and a promise of
// "only ever one device" that the caller could break would be silent corruption rather than a loud
// failure. This class is the cheaper shape for a case where that is known.
class StagingPool
{
 public:
    struct Params
    {
        size_t   buffer_bytesize = 0;   // one read, and one copy
        size_t   slab_bytesize = 0;     // registration unit, rounded down to whole buffers
        unsigned max_buffers = 0;       // ceiling; the pool never grows past it
    };

    StagingPool(std::shared_ptr<device::Device> device, Params params);

    // Waits for every event, then lets the owners below free the events and the slabs. Pinned
    // memory must not be freed while the driver may still be writing into it, which is what the
    // wait is for.
    //
    // Every buffer must already be back: the streamer drains its in-flight requests before teardown
    // anyway, and the waiter must be stopped. It synchronises every event rather than tracking which
    // are outstanding, because that bookkeeping is exactly what would be racy.
    virtual ~StagingPool();

    StagingPool(const StagingPool &) = delete;
    StagingPool & operator=(const StagingPool &) = delete;

    // A free buffer. Registers another slab when none is free and the ceiling allows. NEVER waits.
    //
    // Success with an INVALID buffer means every buffer is in flight. It is not an error: an async
    // engine treats it as one more reason not to submit and falls through to waiting for
    // completions, which is the backpressure that bounds pinned memory.
    virtual common::ResponseCode try_acquire(StagingBuffer & out);

    // Hand a buffer back once its copy has landed. The StreamWaiter calls this.
    virtual void release(const StagingBuffer & buffer);

    // Diagnostics. Not synchronised - call from the acquiring thread.
    unsigned created() const;
    unsigned slabs() const;

 protected:
    // Holding the device keeps its primary context alive for longer than every buffer allocated
    // through it. Releasing that context underneath pinned memory is undefined, and nothing else
    // here would prevent it.
    const std::shared_ptr<device::Device> _device;
    const Params _params;

    // Takes a free index, or returns false. Consumer side.
    bool take(unsigned & index);
    void give(unsigned index);

    std::vector<StagingBuffer> _buffers;   // never shrinks, so indices stay valid

 private:
    common::ResponseCode grow();

    // The handles this pool owns. Held as owners rather than freed by hand in the destructor, so a
    // throw part-way through grow() cannot leak an event or a slab, and so the rule that they are
    // released is a property of the type rather than of one function being written correctly.
    //
    // A StagingBuffer carries a BORROWED copy of its event handle: it is a value handed to callers,
    // so it cannot own anything.
    std::vector<device::OwnedPinned> _slabs;
    std::vector<device::OwnedEvent> _events;

    // Buffers that exist but have never been handed out. Consumer only, so growth - which happens
    // inside try_acquire() - never writes to the ring below and never becomes a second producer.
    unsigned _next_new = 0;

    // Returned buffers, as a ring. `_head` is written only by the consumer and `_tail` only by the
    // producer, which is what makes this correct without a lock. Sized at max_buffers and never
    // resized: a buffer is either free or in flight, so it cannot overflow.
    std::vector<unsigned> _free;
    std::atomic<size_t> _head{0};
    std::atomic<size_t> _tail{0};
};

// The pool the streamer uses: one consumer, several PRODUCERS. There is a waiting thread per stream,
// so a worker copying to two devices has two threads returning buffers to its pool.
//
// The lock is affordable because the free list is touched once per BUFFER, not once per copy: at
// 16 MiB buffers that is a few thousand times a second against a mutex of about 20 ns.
//
// Locking and waiting are separate: try_acquire() takes the lock and returns, acquire() takes the
// lock and waits. An async engine wants the first, because it has I/O to wait for instead.
class SharedStagingPool : public StagingPool
{
 public:
    using StagingPool::StagingPool;

    // Locks, never waits. For a caller with something else to do.
    common::ResponseCode try_acquire(StagingBuffer & out) override;

    // Waits until a buffer is free or the pool can grow. An invalid buffer means stopped.
    common::ResponseCode acquire(StagingBuffer & out);

    void release(const StagingBuffer & buffer) override;

    // Wakes every acquire(), which then gets an invalid buffer. Without it one sleeps for a
    // StreamWaiter that has already stopped.
    void stop();

 private:
    std::mutex _mutex;
    std::condition_variable _ready;
    bool _stopped = false;
};

} // namespace runai::llm::streamer::impl
