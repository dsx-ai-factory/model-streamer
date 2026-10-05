#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include "streamer/impl/device_io/device_writer_client/device_writer_client.h"
#include "streamer/impl/device_io/staging_pool/staging_pool.h"
#include "utils/draining_worker/draining_worker.h"

namespace runai::llm::streamer::impl
{

// One thread PER DEVICE that issues the H2D copies for the synchronous reader.
//
// The reading threads do not touch the driver: they read into their own pinned buffer and hand the
// copy over. Two things follow from that, and they are the reason this exists.
//
// EVERY POOL KEEPS ONE CONSUMER. A reader thread owns its own small staging pool, so nothing
// contends for buffers; the event pools and the channels live behind this thread, so nothing
// contends for those either. The only shared object is the queue below, which is a primitive we
// already have rather than a lock spread across three maps.
//
// AND ONE THREAD ENQUEUES PER DEVICE. A stream is per device, so sixteen readers issuing directly
// would serialise inside the driver's per-context lock and each would need its own bound context.
// Here one thread per device binds ONCE, never switches context, and enqueues for all the readers.
//
// Per device rather than one for everything, because checkpoint/restore runs a single streamer over
// every GPU on the node. One thread would then alternate between contexts on every copy and carry
// all of them against a single enqueue budget - about 5.5 us a copy, against ~47,000 copies a second
// when the source is page cache. Thin enough to remove rather than measure. A single-device load is
// unaffected: one device, one thread, exactly as before.
//
// Threads are created on the first submit FOR THAT DEVICE, so a load naming one GPU starts one.
//
// The cost is a second hop: reader -> this thread -> StreamWaiter -> reader. Each is cheap while the
// consuming thread is busy and about 6 us when it has parked, so this thread parking between copies
// is the thing to watch if the chunk rate ever gets high.
//
// NOTHING EXISTS until the first submit(): no thread, no channel, no driver call. A load whose
// destinations are all host memory never reaches here.
class DeviceIssuer
{
 public:
    using Completion = DeviceWriterClient::Completion;

    // `writer` is the streamer's copy path. The client below owns NO staging pool - every buffer
    // arrives from the reader that read into it - but it DOES own an event pool per device, because
    // a copy needs an event from the context it runs in.
    //
    // The event ceiling is NOT a parameter, and not a policy. Events cost a driver object each and are
    // created on demand, so the number below bounds a LEAK rather than reserving anything - and it
    // cannot be derived here anyway: this issuer is shared by readers whose windows are their own, and
    // an object storage worker learns its window from its plugin long after this exists.
    //
    // Too low is the only harmful direction: a copy refused for want of an event is a failure no
    // caller can act on, and the honest worst case is large - object storage sizes its window from a
    // bandwidth-delay product, so `s3_concurrency x window` reaches six figures at the configuration's
    // limits.
    //
    // So this is set where only a LEAK can reach it. A copy in flight holds a staging buffer, so
    // hitting this needs 65536 buffers live on one device at once: 128 GiB of pinned memory at the
    // 2 MiB minimum block, 512 GiB at the 8 MiB object chunk. No pool can get there. The events
    // themselves are created on demand, so a ceiling nobody reaches costs nothing.
    static constexpr unsigned MaxCopiesInFlight = 65536;

    explicit DeviceIssuer(std::shared_ptr<DeviceWriter> writer);

    // Waits for what is queued, so every copy has been issued before this returns. The StreamWaiter
    // then drains its own queue, which is what puts the buffers back.
    ~DeviceIssuer();

    DeviceIssuer(const DeviceIssuer &) = delete;
    DeviceIssuer & operator=(const DeviceIssuer &) = delete;

    // Hand over a full buffer. Returns as soon as it is QUEUED - not enqueued on the stream, and not
    // copied - so a reader is free to go back to reading.
    //
    // `on_done` is called from the StreamWaiter's thread once the bytes are on the device, or with
    // why they are not. It is ALWAYS called, including when the copy could not be issued at all:
    // unlike DeviceWriter::write, which reports through its return value, there is nobody left to
    // return a code to by the time this runs.
    //
    // The buffer goes back to `pool` whatever happens, AND ALWAYS BEFORE on_done. A reader treats the
    // completion as permission to ask for the next buffer, so reporting first hands it a pool that is
    // still one short. With three buffers per reader and a blocking acquire(), that is a stall rather
    // than an error - the same mistake cost an async worker a wrongly failed chunk, where the pool
    // merely came back empty and said so.
    void submit(common::Device device,
                std::shared_ptr<StagingPool> pool,
                const StagingBuffer & buffer,
                size_t bytesize,
                void * destination,
                Completion on_done);

    // Any device's thread is running. Diagnostics.
    bool running() const;

    // Copies issued across every device. Diagnostics.
    unsigned issued() const;

    // Devices that have a thread. Diagnostics.
    unsigned devices() const;

 private:
    struct Request
    {
        std::shared_ptr<StagingPool> pool;
        StagingBuffer buffer;
        size_t bytesize = 0;
        void * destination = nullptr;
        Completion on_done;
    };

    // One device's queue, thread and driver state.
    //
    // The client is per lane, so it keeps ONE consumer - its own thread - and needs no lock. Same
    // reason a reader's staging pool is per thread. It owns no buffers: every buffer arrives from the
    // reader that read into it. Its channel and event pool are for this device alone.
    //
    // SHARED with the worker's handler, which is the thread that uses it. Stopping a DrainingWorker
    // drains what is queued, so the client must outlive the worker; a share says that, where member
    // order would only imply it.
    struct Lane
    {
        common::Device device;
        std::shared_ptr<DeviceWriterClient> client;
        std::unique_ptr<utils::DrainingWorker<Request>> worker;
    };

    // This device's lane, built on first use. Null with `code` set when the device cannot be opened.
    Lane * lane_for(common::Device device, common::ResponseCode & code);

    void issue(common::Device device, DeviceWriterClient & client, Request && request);

    const std::shared_ptr<DeviceWriter> _writer;

    // Guards the map only. Taken once per device, never per copy: submit() finds the lane, releases
    // this, and pushes to the lane's own queue. A std::map keeps references stable, and lanes are
    // never removed.
    mutable std::mutex _mutex;
    std::map<common::Device, Lane> _lanes;
};

} // namespace runai::llm::streamer::impl
