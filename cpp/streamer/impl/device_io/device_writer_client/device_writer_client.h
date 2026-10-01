#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>

#include "common/device/device.h"
#include "streamer/impl/device_io/device_writer/device_writer.h"
#include "streamer/impl/device_io/event_pool/event_pool.h"
#include "streamer/impl/device_io/staging_pool/staging_pool.h"

namespace runai::llm::streamer::impl
{

// One worker's staging buffers and the devices it has named, over the streamer's shared DeviceWriter.
//
// ONE PER WORKER THREAD, which is what keeps it free of locks. NOT THREAD SAFE: it is 160 bytes and
// 50 ns to build, so a second one is cheaper than a lock on the per-buffer path.
//
// The pool is here rather than in the writer because it is sized by the worker's in-flight window,
// and only the worker knows that - both async workers decide it inside capacity(). Per-worker pools
// cost no more pinned memory than one shared one: a pool grows on demand, and the windows already
// sum to the bound the streamer intends.
//
// ONE BUFFER POOL whatever devices this worker names, because its window does not grow with them and
// pinned memory reaches every context.
//
// ONE EVENT POOL PER DEVICE, because an event does NOT reach every context: recording one device's
// event on another's stream is CUDA_ERROR_INVALID_HANDLE. Measured on 4x B200. Splitting the events -
// a few hundred bytes each - is what lets the buffers, at 16 MiB each, stay shared.
//
// NOTHING EXISTS until the first take(): no pinned memory, no stream, no thread, no driver. So a
// worker can own one unconditionally instead of deciding whether it might ever see a device.
class DeviceWriterClient
{
 public:
    using Completion = DeviceWriter::Completion;

    // Buffer geometry. The ceiling is not here: it is the worker's window, so the constructor takes it.
    //
    // Keep slab_bytesize SMALL - a few buffers, around 16 MiB, never the window. The first take()
    // blocks until its slab is pinned, so a slab the size of the window puts the whole registration
    // on the critical path: ~34 ms for a 128 MiB window at 3.8 GB/s before one byte is read.
    //
    // Splitting is nearly free - one 1 GiB registration against 64 x 16 MiB measured 1.00x on a B200
    // and 1.04x on an H200. Below 16 MiB the per-call cost shows (3.43 GB/s against 3.94 at 64 MiB),
    // which is the only reason not to use one buffer per slab.
    struct Buffers
    {
        size_t buffer_bytesize = 0;   // one read, and one copy
        size_t slab_bytesize = 0;     // registration unit, rounded down to whole buffers
    };

    // `max_buffers` is this worker's in-flight window, so take() never comes back empty: a worker
    // cannot ask for an (N+1)th buffer its own window has not already paid for.
    //
    // Build this in capacity(), which is where the window is decided.
    //
    // A client that never calls take() owns NO pool - which is what the synchronous reader's issuer
    // is, since there the buffers belong to the reading threads and only the copy is shared.
    DeviceWriterClient(std::shared_ptr<DeviceWriter> writer, Buffers buffers, unsigned max_buffers);

    ~DeviceWriterClient();

    DeviceWriterClient(const DeviceWriterClient &) = delete;
    DeviceWriterClient & operator=(const DeviceWriterClient &) = delete;

    // Open a device without taking a buffer from it.
    //
    // take() does this implicitly, so a client that reads never calls it. One that does NOT read -
    // the synchronous reader's issuer, whose buffers come from elsewhere - has no other first touch,
    // and write() deliberately refuses a device it has not seen.
    common::ResponseCode open(common::Device device);

    // A buffer to read into, for bytes bound for `device`. Opens it on first use,
    // which also builds the pool - pinned memory needs a context, and this is the first call with one.
    //
    // An INVALID buffer means every one is in flight. Not an error: an async engine treats it as one
    // more reason to wait for completions instead of submitting.
    common::ResponseCode take(common::Device device, StagingBuffer & out);

    // Copies `bytesize` bytes from the front of `buffer` to `destination`, then takes the buffer back
    // once that copy has landed. Returns as soon as the copy is ENQUEUED.
    //
    // `on_done` is called exactly when this returns Success, from the waiter's thread. On any error
    // the return value is the only report.
    //
    // A device that take() never opened is reported rather than opened: opening here would hide a
    // worker writing to a device it never read for.
    common::ResponseCode write(common::Device device,
                               const StagingBuffer & buffer,
                               size_t bytesize,
                               void * destination,
                               Completion on_done);

    // The same, for a buffer from someone else's pool. The synchronous reader keeps a pool per
    // reading thread and shares one issuer, so the buffer arrives from a pool this object does not
    // own - and must go back to that one.
    common::ResponseCode write(common::Device device,
                               const std::shared_ptr<StagingPool> & pool,
                               const StagingBuffer & buffer,
                               size_t bytesize,
                               void * destination,
                               Completion on_done);

    // Give back a buffer that was taken but never written - a read that failed, or a workload that
    // went away. A buffer lost here shrinks the pool for the life of the worker.
    void release(const StagingBuffer & buffer);

    // The memory a StagingBuffer::slab index names, for a reader that offers it to the kernel.
    //
    // ASKED WHEN NEEDED rather than carried on the buffer: a slab's geometry is used once per slab, so
    // copying it into every buffer and every in-flight chunk would carry it through millions of reads
    // to spend it a handful of times. Invalid before the pool exists.
    StagingPool::Slab slab_at(unsigned index) const;

    // Diagnostics.
    unsigned devices() const;
    unsigned buffers() const;
    unsigned events(common::Device device) const;

 private:
    // The channel for a device, opening it on first use. The first call also builds the pool.
    common::ResponseCode channel_for(common::Device device, DeviceWriter::Channel & out);

    // Make `device`'s context current on this thread. A driver call without one fails with
    // CUDA_ERROR_INVALID_CONTEXT, and a context is per THREAD - which is why this belongs here, in
    // the per-thread object, rather than in the shared writer.
    //
    // Only on a change, which is almost never: a submission names one device.
    common::ResponseCode bind(common::Device device, const DeviceWriter::Channel & channel);

    const std::shared_ptr<DeviceWriter> _writer;
    const Buffers _buffers;
    const unsigned _max_buffers;

    // Touched only by this worker's thread. The writer behind them is shared, and locks only inside
    // open() - once per device, never per buffer.
    std::shared_ptr<StagingPool> _pool;
    std::map<common::Device, DeviceWriter::Channel> _channels;
    std::map<common::Device, std::shared_ptr<EventPool>> _events;

    // Which device's context this thread currently holds. Unset until the first bind.
    bool _bound = false;
    common::Device _bound_device;
};

} // namespace runai::llm::streamer::impl
