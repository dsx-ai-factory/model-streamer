#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "device/device.h"
#include "device/owned/owned.h"
#include "utils/deque/deque.h"

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

    // Which slab this was cut from. Carried because a reader that registers its buffers with the
    // kernel identifies them by INDEX rather than by address, and one slab is one registration - so
    // this is that index. Meaningless to anyone who does not register.
    unsigned slab = 0;

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
// ONE CONSUMER, SEVERAL PRODUCERS is the shape every user has: a worker or a reading thread takes
// buffers, and there is a waiting thread per stream giving them back, so a worker copying to two
// devices has two threads returning buffers. The lock is affordable because it is taken once per
// BUFFER, not once per copy: at 16 MiB buffers that is a few thousand times a second against a mutex
// of about 20 ns.
//
// Several consumers are SAFE but wasteful: two arriving at an empty pool at the same moment both
// register a slab, and the one that comes second frees what it pinned. Nothing in the streamer does
// this - a pool belongs to one thread - so the cost is never paid.
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
    //
    // A slab holding a RETIRED buffer is the one case that precondition cannot be met for, so that
    // slab is leaked rather than freed - see retire().
    ~StagingPool();

    StagingPool(const StagingPool &) = delete;
    StagingPool & operator=(const StagingPool &) = delete;

    // A free buffer. Registers another slab when none is free and the ceiling allows. NEVER waits.
    //
    // Success with an INVALID buffer means every buffer is in flight. It is not an error: an async
    // engine treats it as one more reason not to submit and falls through to waiting for
    // completions, which is the backpressure that bounds pinned memory.
    common::ResponseCode try_acquire(StagingBuffer & out);

    // The same, but waits for a buffer to come back rather than reporting none. For a caller with
    // nothing else to do, which is every thread of the synchronous reader.
    //
    // An invalid buffer means the pool was stopped.
    common::ResponseCode acquire(StagingBuffer & out);

    // Hand a buffer back once its copy has landed. The StreamWaiter calls this.
    void release(const StagingBuffer & buffer);

    // Take a buffer OUT of circulation for good, because nothing can say when the device stops
    // reading from it. `code` is why, and is what acquire() reports once every buffer has gone this
    // way - a caller cannot be left waiting for a buffer that can never come back.
    //
    // ITS SLAB IS THEN LEAKED, not freed at teardown. Keeping the buffer out of a reader's hands is
    // only half of it: cuMemFreeHost on memory an enqueued copy is still reading from is undefined,
    // and the pool cannot wait for that copy - waiting is what already failed. One slab of pinned
    // memory is lost for the life of the process, which is bounded and reported, where freeing it is
    // neither.
    void retire(const StagingBuffer & buffer, common::ResponseCode code);

    // Wakes every acquire(), which then gets an invalid buffer. Without it one sleeps for a
    // StreamWaiter that has already stopped.
    void stop();

    // What one buffer holds. A caller must never read more than this into one, so the size belongs
    // here rather than being carried separately and kept in step by hand.
    size_t buffer_bytesize() const;

    // One slab's memory, for a caller that wants to register it with the kernel.
    //
    // The whole slab rather than one buffer: registration costs time proportional to the BYTES it
    // covers whatever the shape, so registering each buffer would pay a per-call cost per buffer
    // instead of once per slab.
    struct Slab
    {
        void * base = nullptr;
        size_t bytesize = 0;

        bool valid() const { return base != nullptr; }
    };

    // The slab a StagingBuffer::slab index names. Invalid for an index this pool has not grown to.
    //
    // A slab is never freed or moved while the pool lives, so what this returns stays good - which is
    // what lets a caller register it once and keep using the index.
    Slab slab_at(unsigned index) const;

    // Diagnostics.
    unsigned retired() const;
    unsigned created() const;
    unsigned slabs() const;

 private:
    // How many buffers the next slab yields, and how many bytes that is. Zero when the pool is at its
    // ceiling. CALLER HOLDS THE LOCK: it reads how many buffers exist.
    unsigned plan_slab(size_t & bytesize) const;

    // Pin a slab and hand out a buffer from it. Called with NO lock held - pinning 16 MiB takes about
    // 4 ms, and holding the lock across it would stall every release() that lands meanwhile.
    common::ResponseCode add_slab(size_t bytesize, unsigned per_slab, StagingBuffer & out);

    // A buffer that is free right now, or false. CALLER HOLDS THE LOCK.
    bool hand_out(StagingBuffer & out);

    // A pool that has no buffer and can never make one - no size, or a window of none. Told apart
    // from "at the ceiling" by the pool being EMPTY: nothing was ever handed out, so no release can
    // ever come, and acquire() would wait for something that cannot happen.
    common::ResponseCode barren() const;

    // Holding the device keeps its primary context alive for longer than every buffer allocated
    // through it. Releasing that context underneath pinned memory is undefined, and nothing else
    // here would prevent it.
    const std::shared_ptr<device::Device> _device;
    const Params _params;

    mutable std::mutex _mutex;
    std::condition_variable _ready;
    bool _stopped = false;

    // Held as owners rather than freed by hand in the destructor, so a throw part-way through
    // registration cannot leak a slab, and so the rule that they are released is a property of the
    // type.
    //
    // The size rides along because OwnedPinned does not carry it and slab_at() must answer it. A
    // parallel vector would be the same two facts with a way to disagree.
    struct SlabRecord
    {
        device::OwnedPinned memory;
        size_t bytesize = 0;
    };

    std::vector<SlabRecord> _slabs;

    // Every buffer that exists, in creation order. Never shrinks, so an index stays valid.
    std::vector<StagingBuffer> _buffers;

    // Indices of the buffers that are free, oldest first. A buffer is either here, in flight, or
    // retired.
    std::deque<unsigned> _free;

    // Buffers taken out of circulation, and why the first one was. A buffer counted here never
    // returns, so once every buffer is retired no acquire() can ever succeed.
    unsigned _retired = 0;
    common::ResponseCode _retired_code = common::ResponseCode::Success;

    // Slabs that must outlive this pool, by slab index. A retired buffer was cut from one of them and
    // the device may still be reading it.
    std::set<unsigned> _leaked_slabs;
};

// Keeps pools alive past the worker that made them.
//
// An object-storage read is given the pool's pinned memory and the plugin fills it asynchronously. A
// sent request cannot be cancelled, and removing a client only parks it - the client's destructor,
// which is what waits for the SDK, runs later, when the backend is cleaned up. A pool freed with its
// worker would therefore be freed while the plugin may still be writing.
//
// So the streamer holds one of these, declared before the backend cleanup so it is destroyed after it.
// By then every client has been destroyed and every read has reported, and the pinned memory can go.
//
// A Deque because workers register from their own threads and it is already a tested type - this needs
// a thread-safe container, not a queue, and only ever pushes.
using StagingPoolRetainer = utils::Deque<std::shared_ptr<StagingPool>>;

} // namespace runai::llm::streamer::impl
