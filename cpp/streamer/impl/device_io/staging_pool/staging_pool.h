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
// It carries NO event. An event belongs to the context that created it, but this memory is reachable
// from every context - so a buffer can serve any device while an event cannot. The event that marks
// a copy comes from an EventPool for the device being copied to.
//
// A COPY, never a reference into the pool. The pool's buffer vector grows, and growth reallocates.
struct StagingBuffer
{
    char *   data = nullptr;
    size_t   bytesize = 0;
    unsigned index = 0;      // position in the pool, so release() needs no search

    bool valid() const { return data != nullptr; }
};

// Pinned host buffers for one worker, created on demand and reused for the streamer's lifetime. ONE
// pool serves every device the worker names: this memory is reachable from every context.
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

    // PRECONDITION: every buffer is already back. A buffer returns only after its copy's event was
    // synchronised, so this is also what guarantees no DMA is still reading out of this memory when
    // it is freed. The streamer drains its in-flight requests before teardown anyway, and every
    // StreamWaiter is stopped first.
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

    // Hand out a buffer that already exists - returned, or created and never used. False when there
    // is none, which is when the pool has to grow.
    bool hand_out(StagingBuffer & out);

    // The next slab to register: how many buffers it yields, and how many bytes that is. Zero when
    // the pool is at its ceiling.
    //
    // Split from the registration itself so that SharedStagingPool can PIN OUTSIDE ITS LOCK. Pinning
    // a 16 MiB slab takes about 4 ms, and holding the lock across it blocks the StreamWaiter's
    // release() for that whole time - stalling the buffer returns of every copy that lands while the
    // pool is still growing.
    unsigned plan_slab(size_t & bytesize) const;

    // Adopt a slab that has been pinned, and cut buffers out of it.
    void publish_slab(void * base, unsigned per_slab);

    std::vector<StagingBuffer> _buffers;   // never shrinks, so indices stay valid

 private:
    // Plan, pin and publish in one step, with no lock to release: this class has a single consumer.
    common::ResponseCode grow();

    // Held as owners rather than freed by hand in the destructor, so a throw part-way through grow()
    // cannot leak a slab, and so the rule that they are released is a property of the type.
    std::vector<device::OwnedPinned> _slabs;

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
    //
    // An INVALID buffer has a second meaning here that the base class does not have: another consumer
    // is registering a slab, so this one is not allowed to register a second. With ONE consumer - a
    // per-worker client - that cannot happen, and an invalid buffer means only "everything is in
    // flight". A caller that treats invalid as impossible is therefore relying on being alone; the
    // synchronous reader, where several threads share a pool, must use acquire() instead, which waits
    // for the slab rather than reporting nothing.
    common::ResponseCode try_acquire(StagingBuffer & out) override;

    // Waits until a buffer is free or the pool can grow. An invalid buffer means stopped.
    common::ResponseCode acquire(StagingBuffer & out);

    void release(const StagingBuffer & buffer) override;

    // Wakes every acquire(), which then gets an invalid buffer. Without it one sleeps for a
    // StreamWaiter that has already stopped.
    void stop();

 private:
    // Pin a slab with the lock RELEASED, then take it again to publish. Returns the driver's code.
    //
    // `grew` says whether there was any room to grow into. Without it a caller cannot tell "a slab
    // arrived, look again" from "the pool is at its ceiling, wait" - and a waiter that cannot tell
    // spins instead of sleeping.
    common::ResponseCode grow_unlocked(std::unique_lock<std::mutex> & lock, bool & grew);

    mutable std::mutex _mutex;
    std::condition_variable _ready;
    bool _stopped = false;

    // Set while a thread is pinning outside the lock, so a second consumer does not register a slab
    // the first one is already registering - which would take the pool past its ceiling.
    bool _growing = false;
};

} // namespace runai::llm::streamer::impl
