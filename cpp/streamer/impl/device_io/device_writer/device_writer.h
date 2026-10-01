#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include "common/device/device.h"
#include "device/device.h"
#include "device/owned/owned.h"
#include "streamer/impl/device_io/staging_pool/staging_pool.h"
#include "streamer/impl/device_io/stream_waiter/stream_waiter.h"

namespace runai::llm::streamer::impl
{

// The copy path onto a device: a stream per device, and a thread per stream waiting on it.
//
// ONE PER STREAMER. A device gets one stream however many workers copy to it, which costs nothing:
// H2D throughput was flat from 1 to 16 streams on an H200 and a B200.
//
// It owns NO buffers - those belong to a worker, sized by that worker's window. Hence the pool
// argument to write(): this class cannot know whose buffer it has. See DeviceWriterClient.
//
// NOTHING EXISTS until the first open(): no stream, no thread, no driver.
//
// The backend is a FACTORY because obtaining one is itself a driver call - dlopen 1.5 ms plus cuInit
// 117 ms in a process that has not used CUDA, against 0.15 ms in one that has (vLLM, through torch).
// A host-only load then pays neither, and a machine with no GPU logs no missing driver.
class DeviceWriter
{
 public:
    // Called when the bytes are on the device, or with why they are not.
    using Completion = std::function<void(common::ResponseCode)>;

    // One device, with its stream and its waiter. Opaque, and valid for the writer's lifetime:
    // targets are never removed, and a std::map keeps its references stable across inserts.
    using Channel = const void *;

    // Asked for the backend on the first open() OF THAT TYPE and never again. Returning null means
    // there is no such device on this machine, which open() reports as DeviceUnavailable.
    using BackendFactory = std::function<std::shared_ptr<device::Backend>()>;

    // The backend for a device type. One writer serves every type, so the type a submission names
    // picks the implementation - see device/backends.
    using BackendLookup = std::function<BackendFactory(common::DeviceType)>;

    explicit DeviceWriter(BackendLookup backends);

    // Stops every waiter, so every buffer is back in its pool before this returns.
    //
    // A write() cannot race with it: every client holds a share of the writer, so this runs only
    // after the last one is gone. An already ENQUEUED copy is fine too - the waiter drains first,
    // and cuda.h says a destroyed stream frees its resources once its work completes.
    ~DeviceWriter();

    DeviceWriter(const DeviceWriter &) = delete;
    DeviceWriter & operator=(const DeviceWriter &) = delete;

    // Opens a device, or returns the channel already opened for it. Idempotent: workers of the same
    // device share its stream and its waiter.
    common::ResponseCode open(common::Device device, Channel & out);

    // So a client can allocate pinned memory, which needs a context. Null for a null channel.
    std::shared_ptr<device::Device> device(Channel channel) const;

    // What one copy needs, and what has to come back when it lands. The EVENT comes from the caller
    // because it belongs to the context that made it - a buffer serves any device, an event does not.
    using Copy = StreamWaiter::Copy;

    // Copies `bytesize` bytes from the front of `copy.buffer` to `destination`, then returns the
    // buffer and the event to their pools once that copy has landed. Returns as soon as the copy is
    // ENQUEUED.
    //
    // `on_done` is called exactly when this returns Success, from the waiter's thread. On any error
    // the return value is the only report, so a caller is never told twice.
    //
    // The buffer and the event are ours from here and come back whatever happens - one lost on an
    // error path is a deadlock that arrives later. The one exception is a missing pool: nowhere to
    // give them back to.
    common::ResponseCode write(Channel channel,
                               Copy copy,
                               size_t bytesize,
                               void * destination,
                               Completion on_done);

    // Diagnostics.
    unsigned devices() const;

 private:
    // One device's stream and the waiter that drains it. Created the first time that device is
    // opened, and never removed - which is what lets a Channel stay valid for the writer's life.
    struct Target
    {
        // No destructor and no hand-written moves: unique_ptr frees the stream exactly once and a
        // moved-from Target holds nothing, which matters because targets are moved into a map.
        //
        // Member order does NOT matter, though it looks as though it should: destroying a stream
        // with pending work is defined (cuda.h), and the waiter waits on the event, not the stream.
        std::shared_ptr<device::Device> device;
        device::OwnedStream stream;
        std::unique_ptr<StreamWaiter> waiter;
    };

    // Built once, then only read, with the lock that guards the building. Its own type so the lock's
    // job is visible: it protects creation and is never taken on the per-buffer path.
    class Channels
    {
     public:
        ~Channels();

        // Calls the lookup only for a type it has not opened yet - each driver is reached once per
        // writer, whatever the ordinal.
        common::ResponseCode open(const BackendLookup & backends, common::Device device, Target ** out);

        unsigned count() const;

     private:
        mutable std::mutex _mutex;

        // One backend per device type, obtained on the first open() of that type and then reused.
        std::map<common::DeviceType, std::shared_ptr<device::Backend>> _opened;

        // KEYED BY DEVICE, not by ordinal: two types can both have an ordinal 0, and they are not the
        // same device.
        std::map<common::Device, Target> _targets;
    };

    const BackendLookup _backends;
    Channels _channels;
};

} // namespace runai::llm::streamer::impl
