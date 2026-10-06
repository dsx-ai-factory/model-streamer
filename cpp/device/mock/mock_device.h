#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <vector>

#include "device/device.h"

namespace runai::llm::streamer::device
{

// A test double for Device: no driver, no GPU, and the test decides what fails and when.
//
// Its own target, marked testonly, so Bazel refuses to link it into anything shipped.
//
// Host memory is real, from malloc rather than pinned, so a test can write through a staging buffer
// and read what landed. Streams and events are tokens: what matters about them here is that they
// are created, used and destroyed in the right number and the right order.
//
// THREAD SAFE, unlike the real backend's needs: a StreamWaiter calls event_synchronize from its own
// thread while the test thread is still acquiring buffers. Counters are atomic so a test may read
// them without waiting for the threads to be quiet.
class MockDevice : public Device
{
 public:
    common::ResponseCode bind_thread() override;
    common::ResponseCode get_attribute(Attribute attribute, int & value) const override;
    common::ResponseCode memory_info(size_t & free_bytes, size_t & total_bytes) const override;

    common::ResponseCode host_alloc(size_t bytesize, void ** ptr) override;
    common::ResponseCode host_free(void * ptr) override;
    common::ResponseCode device_alloc(size_t bytesize, void ** ptr) override;
    common::ResponseCode device_free(void * ptr) override;

    common::ResponseCode stream_create(StreamHandle & stream) override;
    common::ResponseCode stream_destroy(StreamHandle stream) override;
    common::ResponseCode stream_synchronize(StreamHandle stream) override;
    common::ResponseCode stream_query(StreamHandle stream, Status & status) override;
    common::ResponseCode stream_wait_event(StreamHandle stream, EventHandle event) override;

    common::ResponseCode event_create(EventHandle & event) override;
    common::ResponseCode event_destroy(EventHandle event) override;
    common::ResponseCode event_record(EventHandle event, StreamHandle stream) override;
    common::ResponseCode event_query(EventHandle event, Status & status) override;
    common::ResponseCode event_synchronize(EventHandle event) override;

    common::ResponseCode memcpy_h2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream) override;
    common::ResponseCode memcpy_d2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream) override;
    common::ResponseCode memset_async(void * dst, unsigned char value, size_t bytesize, StreamHandle stream) override;

    // --- what a test asserts on ---

    std::atomic<unsigned> bind_calls{0};
    std::atomic<unsigned> host_allocs{0};       // one per SLAB, not per buffer
    std::atomic<unsigned> host_frees{0};
    std::atomic<unsigned> streams_created{0};
    std::atomic<unsigned> streams_destroyed{0};
    std::atomic<unsigned> events_created{0};
    std::atomic<unsigned> events_destroyed{0};
    std::atomic<unsigned> event_syncs{0};
    std::atomic<unsigned> stream_syncs{0};
    std::atomic<unsigned> copies{0};

    std::vector<size_t> host_alloc_sizes;   // in order, so slab sizing is checkable
    std::vector<void *> live_host;          // allocated and not yet freed

    // --- what a test controls ---

    // Fail the Nth host_alloc, counting from 1. Zero means never.
    unsigned fail_host_alloc_at = 0;

    // THROW from the Nth host_alloc instead of returning a code, counting from 1. Zero means never.
    // A driver call reaches an ASSERT, which is fatal, so a caller that only handles return codes
    // still has to survive an exception from here.
    unsigned throw_host_alloc_at = 0;

    // Make every event_record fail, with the copy itself still enqueued. That is the one case where a
    // copy is on the stream and no event marks it, so nothing can wait for it.
    std::atomic<bool> fail_event_record{false};

    // Make every stream_synchronize fail, so a caller that cannot wait for the stream either has no
    // way left to know when a copy stops reading from its buffer.
    std::atomic<bool> fail_stream_synchronize{false};

    // Fails bind_thread, as a dead or unusable context does. bind_calls still counts the attempt, so
    // a test can show the waiter retried rather than giving up after the first failure.
    std::atomic<bool> fail_bind_thread{false};

    // Make every event_synchronize report a failed copy.
    std::atomic<bool> fail_event_synchronize{false};

    // Fail every event_synchronize from the Nth onwards, counting from 1. Zero means never. One copy
    // failing part way through a batch is not the same case as all of them: the reader must keep the
    // answers it already gave for the blocks that landed before it.
    std::atomic<unsigned> fail_event_synchronize_from{0};

    // HOLD every event_synchronize until release_copies(). A copy that never retires keeps its
    // staging buffer, which is the only way a test can make a pool run dry - the mock otherwise
    // returns a buffer before the next read asks for one, so a reader never waits for the link.
    void hold_copies();
    void release_copies();

    // Fail every event_create from the Nth onwards, counting from 1. Zero means never.
    unsigned fail_event_create_from = 0;

    // Set when event_record was given an event this device did not create, or a stream it does not
    // own. A real driver answers CUDA_ERROR_INVALID_HANDLE: an event belongs to the context that made
    // it, unlike the pinned memory it marks. Modelled because the mock not modelling it hid a bug
    // that only a second GPU could show.
    std::atomic<unsigned> foreign_records{0};

    // An event answers NotReady until marked ready, so a reaping order can be forced.
    void set_ready(EventHandle event, bool ready);
    void set_all_ready(bool ready);

 private:
    // Guards the hold below. Its own condition, not _mutex's: event_synchronize runs on the waiter's
    // thread and must not hold the state lock while it sleeps.
    std::mutex _hold_mutex;
    std::condition_variable _hold;
    bool _copies_held = false;

    mutable std::mutex _mutex;
    std::map<EventHandle, bool> _ready;     // the events THIS device created
    std::set<StreamHandle> _streams;        // and the streams

    // PROCESS-WIDE, so no two devices ever mint the same handle. Per-device counters would collide,
    // and the affinity check above would then accept a foreign handle that happened to match.
    static std::atomic<uintptr_t> _next_token;
};

// A test double for Backend: hands out MockDevices by ordinal and remembers them, as the real one
// caches a retained context per device.
class MockBackend : public Backend
{
 public:
    explicit MockBackend(unsigned device_count = 4);

    Capabilities capabilities() const override;
    common::ResponseCode device_count(unsigned & count) const override;
    common::ResponseCode open_device(unsigned ordinal, std::shared_ptr<Device> & device) override;

    // The device handed out for an ordinal, so a test can read its counters. Null if never opened.
    std::shared_ptr<MockDevice> opened(unsigned ordinal) const;

    unsigned opens = 0;                 // calls, not distinct devices
    unsigned fail_open_device_at = 0;   // fail the Nth open_device, counting from 1. Zero means never

 private:
    const unsigned _count;
    std::map<unsigned, std::shared_ptr<MockDevice>> _devices;
};

} // namespace runai::llm::streamer::device
