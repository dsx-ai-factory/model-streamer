#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "device/device.h"

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

// Pinned host buffers for one reader, created on demand and reused for the streamer's lifetime.
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
// ONE CONSUMER: acquire() from one thread only. release() may be called from another - the reaper -
// which is why the free list is two atomics rather than a lock. For a pool shared by several
// readers, use SharedStagingPool.
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

    // Destroys every event, then frees the slabs - pinned memory must not be freed while the driver
    // may still be writing into it.
    //
    // Every buffer must already be back: the streamer drains its in-flight requests before teardown
    // anyway, and the reaper must be stopped. The destructor synchronises every event rather than
    // tracking which are outstanding, because that bookkeeping is exactly what would be racy.
    virtual ~StagingPool();

    StagingPool(const StagingPool &) = delete;
    StagingPool & operator=(const StagingPool &) = delete;

    // A free buffer. Registers another slab when none is free and the ceiling allows.
    //
    // Success with an INVALID buffer means every buffer is in flight. It is not an error: an async
    // engine treats it as one more reason not to submit and falls through to waiting for
    // completions, which is the backpressure that bounds pinned memory.
    virtual common::ResponseCode acquire(StagingBuffer & out);

    // Hand a buffer back once its copy has landed. The reaper calls this.
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

    struct Slab
    {
        void * base = nullptr;
        size_t bytesize = 0;
    };

    std::vector<Slab> _slabs;

    // Buffers that exist but have never been handed out. Consumer only, so growth - which happens
    // inside acquire() - never writes to the ring below and never becomes a second producer.
    unsigned _next_new = 0;

    // Returned buffers, as a ring. `_head` is written only by the consumer and `_tail` only by the
    // producer, which is what makes this correct without a lock. Sized at max_buffers and never
    // resized: a buffer is either free or in flight, so it cannot overflow.
    std::vector<unsigned> _free;
    std::atomic<size_t> _head{0};
    std::atomic<size_t> _tail{0};
};

// The same pool for several readers, as the synchronous threadpool needs: its threads have nothing
// else to do while they wait, so acquire() blocks here rather than returning nothing.
//
// The lock is affordable because the free list is touched once per BUFFER - hundreds of times a
// second - not once per copy.
class SharedStagingPool : public StagingPool
{
 public:
    using StagingPool::StagingPool;

    // Waits until a buffer is free or the pool can grow. Returns an invalid buffer only when
    // stopped.
    common::ResponseCode acquire(StagingBuffer & out) override;
    void release(const StagingBuffer & buffer) override;

    // Wakes every waiter, which then gets an invalid buffer. Needed for shutdown, or acquire()
    // waits for a reaper that has already stopped.
    void stop();

 private:
    std::mutex _mutex;
    std::condition_variable _ready;
    bool _stopped = false;
};

} // namespace runai::llm::streamer::impl
