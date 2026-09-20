#pragma once

#include <cstddef>
#include <map>
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
// NOT THREAD SAFE. Tests drive it from their own thread.
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

    unsigned bind_calls = 0;
    unsigned host_allocs = 0;            // one per SLAB, not per buffer
    unsigned host_frees = 0;
    unsigned events_created = 0;
    unsigned events_destroyed = 0;
    unsigned event_syncs = 0;
    unsigned copies = 0;

    std::vector<size_t> host_alloc_sizes;   // in order, so slab sizing is checkable
    std::vector<void *> live_host;          // allocated and not yet freed

    // --- what a test controls ---

    // Fail the Nth host_alloc, counting from 1. Zero means never.
    unsigned fail_host_alloc_at = 0;

    // An event answers NotReady until marked ready, so a reaping order can be forced.
    void set_ready(EventHandle event, bool ready);
    void set_all_ready(bool ready);

 private:
    std::map<EventHandle, bool> _ready;
    unsigned _next_token = 1;
};

} // namespace runai::llm::streamer::device
