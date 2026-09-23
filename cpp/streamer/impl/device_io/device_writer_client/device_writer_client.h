#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>

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
    DeviceWriterClient(std::shared_ptr<DeviceWriter> writer, Buffers buffers, unsigned max_buffers);

    ~DeviceWriterClient();

    DeviceWriterClient(const DeviceWriterClient &) = delete;
    DeviceWriterClient & operator=(const DeviceWriterClient &) = delete;

    // A buffer to read into, for bytes bound for `device_ordinal`. Opens the device on first use,
    // which also builds the pool - pinned memory needs a context, and this is the first call with one.
    //
    // An INVALID buffer means every one is in flight. Not an error: an async engine treats it as one
    // more reason to wait for completions instead of submitting.
    common::ResponseCode take(unsigned device_ordinal, StagingBuffer & out);

    // Copies `bytesize` bytes from the front of `buffer` to `destination`, then takes the buffer back
    // once that copy has landed. Returns as soon as the copy is ENQUEUED.
    //
    // `on_done` is called exactly when this returns Success, from the waiter's thread. On any error
    // the return value is the only report.
    //
    // An ordinal that take() never opened is reported rather than opened: opening here would hide a
    // worker writing to a device it never read for.
    common::ResponseCode write(unsigned device_ordinal,
                               const StagingBuffer & buffer,
                               size_t bytesize,
                               void * destination,
                               Completion on_done);

    // Give back a buffer that was taken but never written - a read that failed, or a workload that
    // went away. A buffer lost here shrinks the pool for the life of the worker.
    void release(const StagingBuffer & buffer);

    // Diagnostics.
    unsigned devices() const;
    unsigned buffers() const;
    unsigned events(unsigned device_ordinal) const;

 private:
    // The channel for an ordinal, opening it on first use. The first call also builds the pool.
    common::ResponseCode channel_for(unsigned ordinal, DeviceWriter::Channel & out);

    // Make `ordinal`'s context current on this thread. A driver call without one fails with
    // CUDA_ERROR_INVALID_CONTEXT, and a context is per THREAD - which is why this belongs here, in
    // the per-thread object, rather than in the shared writer.
    //
    // Only on a change, which is almost never: a submission names one device.
    common::ResponseCode bind(unsigned ordinal, const DeviceWriter::Channel & channel);

    const std::shared_ptr<DeviceWriter> _writer;
    const Buffers _buffers;
    const unsigned _max_buffers;

    // Touched only by this worker's thread. The writer behind them is shared, and locks only inside
    // open() - once per device, never per buffer.
    std::shared_ptr<SharedStagingPool> _pool;
    std::map<unsigned, DeviceWriter::Channel> _channels;
    std::map<unsigned, std::shared_ptr<EventPool>> _events;

    // Which device's context this thread currently holds. Unset until the first bind.
    bool _bound = false;
    unsigned _bound_ordinal = 0;
};

} // namespace runai::llm::streamer::impl
