#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include "device/device.h"
#include "device/owned/owned.h"
#include "streamer/impl/device_io/staging_pool/staging_pool.h"
#include "streamer/impl/device_io/stream_waiter/stream_waiter.h"

namespace runai::llm::streamer::impl
{

// Everything one reader needs to get bytes onto a device: the staging buffers, a stream per device,
// and a waiter per stream.
//
// ONE PER READER - an async engine, or the synchronous threadpool - because the pool is sized by
// what that reader can keep in flight, and a second pool would be memory nobody asked for.
//
// The reader sees three calls:
//
//     open(ordinal)   once, when it learns which device a submission names
//     take()          per buffer: one to read into, or nothing when they are all in flight
//     write(...)      per buffer: hand it back with where its bytes belong
//
// It never sees a stream, an event or the pool.
//
// open() is separate so that the per-BUFFER calls take no lock of this class's own. Everything a
// copy needs is behind the handle, and the only concurrency left on that path is the pool's, which
// SharedStagingPool already handles.
//
// ONE COPY PER BUFFER. Reading works in blocks and a tensor never appears in the read path, so one
// read has one contiguous destination. There is nothing to coalesce and no ordering to require.
//
// NOTHING EXISTS until the first take(): no pinned memory, no stream, no thread, no driver call. A
// load whose destinations are all host memory never reaches here at all.
class DeviceWriter
{
 public:
    // Called when the bytes are on the device, or with why they are not.
    using Completion = std::function<void(common::ResponseCode)>;

    // One device, with its stream and its waiter. Opaque, and valid for the writer's lifetime:
    // targets are never removed, and a std::map keeps its references stable across inserts.
    using Channel = const void *;

    DeviceWriter(std::shared_ptr<device::Backend> backend, StagingPool::Params params);

    // Stops every waiter, so every buffer is back before the pool is destroyed.
    //
    // PRECONDITION: no take() or write() may be in flight, and no channel may be used afterwards.
    // The reader stops reading before it drops its writer, which every caller does anyway. This is
    // the ordinary rule that an object cannot be destroyed while a call is running on it - stated
    // because the stream is freed here, and a write() racing with this would use a dead handle.
    //
    // A copy already ENQUEUED is fine and needs no such care: the waiter drains first, and cuda.h
    // says a destroyed stream releases its resources only once its work has completed.
    ~DeviceWriter();

    DeviceWriter(const DeviceWriter &) = delete;
    DeviceWriter & operator=(const DeviceWriter &) = delete;

    // Opens a device and returns the handle to use for it. Called once per device, not per buffer.
    //
    // This is also what builds the pool, because pinned memory is allocated through a device's
    // context and this is the first call that knows one. Later devices reuse the same buffers:
    // both pinned modes give memory that every context can reach.
    common::ResponseCode open(unsigned device_ordinal, Channel & out);

    // A buffer to read into. An INVALID buffer means every one is in flight: not an error, and the
    // reader treats it as one more reason to wait for I/O instead of submitting.
    //
    // Takes only the pool's lock. It needs the channel because the pool hangs off it, which also
    // makes "open before you take" impossible to get wrong rather than an error to report.
    common::ResponseCode take(Channel channel, StagingBuffer & out);

    // Copies `bytesize` bytes from the front of `buffer` to `destination` on `channel`, then gives
    // the buffer back once that copy has landed.
    //
    // Returns as soon as the copy is ENQUEUED.
    //
    // `on_done` is called EXACTLY WHEN this returns Success, later and from the waiter's thread.
    // On any error the return value is the only report, so a caller is never told twice and never
    // left waiting for a completion that is not coming.
    //
    // The buffer is ours from the call onwards and comes back whatever happens - a buffer lost on
    // an error path is a deadlock that arrives later.
    //
    // The single exception is a NULL channel, which is a caller bug rather than a case to plan for:
    // open() sets the handle to null and fills it only on success, so a null one means its error
    // was ignored. Reaching it here needs a valid channel to have obtained the buffer and a
    // different, null one passed to write. There is no pool to return the buffer to without a
    // channel, so the caller keeps it, and the log says so.
    common::ResponseCode write(Channel channel,
                               const StagingBuffer & buffer,
                               size_t bytesize,
                               void * destination,
                               Completion on_done);

    // Diagnostics.
    unsigned devices() const;
    unsigned buffers() const;

 private:
    // One device's stream and the waiter that drains it. Created the first time that device is
    // opened, and never removed - which is what lets a Channel stay valid for the writer's life.
    struct Target
    {
        // No destructor and no hand-written moves: unique_ptr frees the stream exactly once and
        // leaves a moved-from Target holding nothing, which matters because targets are moved
        // into a map.
        //
        // Member order does NOT matter here, which is worth saying because it looks as though it
        // should. Destroying a stream that still has work is defined: cuda.h says the call returns
        // at once and the stream's resources are freed when its work completes. And the waiter
        // never touches the stream after the copy is enqueued - it waits on the EVENT, which is a
        // separate object. Inverting these two members changes nothing.
        std::shared_ptr<device::Device> device;
        device::OwnedStream stream;
        std::unique_ptr<StreamWaiter> waiter;

        // The one pool, shared by every device. Reached through the handle, so the per-buffer path
        // needs neither a lock nor an atomic: a caller cannot hold a handle without having gone
        // through open(), and open() holds the lock while it writes this.
        std::shared_ptr<SharedStagingPool> pool;
    };

    // Everything that is BUILT ONCE and afterwards only read, together with the lock that guards
    // the building. Its own type so the lock's job is visible: it protects creation, and is never
    // taken on the per-buffer path.
    class Channels
    {
     public:
        ~Channels();

        // Opens `ordinal` if it is not open, building the pool on the first call. Returns the
        // target to use for it.
        common::ResponseCode open(device::Backend & backend,
                                  const StagingPool::Params & params,
                                  unsigned ordinal,
                                  Target ** out);

        // Devices opened, and buffers the pool has created. Diagnostics, and the only reason
        // anything outside asks this type a question at all - the pool itself never leaves.
        unsigned count() const;
        unsigned created() const;

     private:
        mutable std::mutex _mutex;

        // Every waiter holds a share of this, so the pool cannot go while one is still returning
        // buffers to it. That used to rest on the order these two are declared in - a rule a
        // reordered member would break in silence.
        std::shared_ptr<SharedStagingPool> _pool;

        std::map<unsigned, Target> _targets;
    };

    const std::shared_ptr<device::Backend> _backend;
    const StagingPool::Params _params;
    Channels _channels;
};

} // namespace runai::llm::streamer::impl
