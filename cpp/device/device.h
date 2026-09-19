#pragma once

#include <cstddef>
#include <memory>

#include "common/response_code/response_code.h"

namespace runai::llm::streamer::device
{

// A stream and an event as the core sees them. Opaque so that the vendor's own types stay
// inside the backend: nothing above this header includes a vendor header.
using StreamHandle = void *;
using EventHandle  = void *;

// Whether asynchronous work has finished. A query never blocks.
enum class Status { Ready, NotReady };

// Device properties the staging design depends on.
enum class Attribute
{
    UnifiedAddressing,

    // True where host and device share page tables. Pinning host memory can buy nothing there,
    // so it is worth knowing before paying for it.
    PageableAccessUsesHostPageTables,
};

// What a backend can do, answered once when it is opened.
//
// Asked rather than discovered. A capability that is missing should change the plan, not fail
// a transfer that has already started.
struct Capabilities
{
    bool pinned_host_memory = false;
    bool device_to_device   = false;
};

// One open device, holding the context every call below runs in.
//
// Shared by all worker threads. The context is retained for the streamer's lifetime, so the
// pinned buffers allocated against it stay valid and reusable for that long.
class Device
{
 public:
    virtual ~Device() = default;

    // A worker thread does not inherit the device context. Each one binds before its first call.
    virtual common::ResponseCode bind_thread() = 0;

    virtual common::ResponseCode get_attribute(Attribute attribute, int & value) const = 0;

    // Free and total device memory. For diagnostics - an allocation failure is far easier to read
    // with these numbers next to it.
    virtual common::ResponseCode memory_info(size_t & free_bytes, size_t & total_bytes) const = 0;

    // Pinned host memory for staging. The backend allocates and frees; the pool, its size and
    // when a buffer is reused are the core's, because none of that is vendor specific.
    virtual common::ResponseCode host_alloc(size_t bytesize, void ** ptr) = 0;
    virtual common::ResponseCode host_free(void * ptr) = 0;

    // Device memory. Unused while the caller owns every destination.
    virtual common::ResponseCode device_alloc(size_t bytesize, void ** ptr) = 0;
    virtual common::ResponseCode device_free(void * ptr) = 0;

    // Streams are non-blocking: ours must not serialise against the stream the caller happens
    // to be using. No flag, because no caller wants the other behaviour.
    virtual common::ResponseCode stream_create(StreamHandle & stream) = 0;
    virtual common::ResponseCode stream_destroy(StreamHandle stream) = 0;
    virtual common::ResponseCode stream_synchronize(StreamHandle stream) = 0;
    virtual common::ResponseCode stream_query(StreamHandle stream, Status & status) = 0;

    // Orders one stream behind another without blocking the host. Overlapping a copy with a
    // collective depends on this.
    virtual common::ResponseCode stream_wait_event(StreamHandle stream, EventHandle event) = 0;

    // Events carry no timing. Timed events cost more and nothing here reads a duration.
    virtual common::ResponseCode event_create(EventHandle & event) = 0;
    virtual common::ResponseCode event_destroy(EventHandle event) = 0;
    virtual common::ResponseCode event_record(EventHandle event, StreamHandle stream) = 0;

    // How a staging buffer is reclaimed: the reader asks whether the copy out of it has landed,
    // and reads on when it has not.
    virtual common::ResponseCode event_query(EventHandle event, Status & status) = 0;
    virtual common::ResponseCode event_synchronize(EventHandle event) = 0;

    // Must not block the calling thread. A backend may be slow; it may not be wrong about this.
    virtual common::ResponseCode memcpy_h2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream) = 0;
    virtual common::ResponseCode memcpy_d2d_async(void * dst, const void * src, size_t bytesize, StreamHandle stream) = 0;

    // Fills alignment padding without a round trip through the host.
    virtual common::ResponseCode memset_async(void * dst, unsigned char value, size_t bytesize, StreamHandle stream) = 0;

    // The vendor's own text for the last failure, for the log next to our own code. Never null.
    virtual const char * last_error() const = 0;
};

// The loaded driver, and the devices it can open.
class Backend
{
 public:
    virtual ~Backend() = default;

    virtual Capabilities capabilities() const = 0;

    // Zero when no device can be used. Not the same question as whether the driver library was
    // found: a stub library resolves every symbol and then reports no device.
    virtual common::ResponseCode device_count(unsigned & count) const = 0;

    // Opens the device by its ordinal, which the caller names. The ordinal is not read back from
    // a destination pointer: the context and the stream are chosen before the copy, not once a
    // pointer is in hand.
    virtual common::ResponseCode open_device(unsigned ordinal, std::shared_ptr<Device> & device) = 0;
};

} // namespace runai::llm::streamer::device
