/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "device/device.h"
#include "device/owned/owned.h"

namespace runai::llm::streamer::impl
{

// Events for ONE device, created in its context and reused.
//
// Separate from the staging pool because the two are portable in different ways. Pinned host memory
// is reachable from every context, so one pool of it serves a worker whatever devices it names. An
// EVENT is not: it belongs to the context that created it, and recording it on another device's
// stream fails with CUDA_ERROR_INVALID_HANDLE. Measured on 4x B200, where the copy to a second
// device failed on exactly that.
//
// So a worker has one pool of buffers and one pool of events PER DEVICE. Events are a few hundred
// bytes each, against 16 MiB for a buffer - splitting the cheap thing is what keeps the expensive
// one shared.
//
// ONE CONSUMER, SEVERAL PRODUCERS, like the staging pool: the worker acquires, and a StreamWaiter
// releases from its own thread.
//
// The caller must have this device's context current - acquire() creates events when it has to.
class EventPool
{
 public:
    EventPool(std::shared_ptr<device::Device> device, unsigned max_events);

    // Every event must already be back, for the same reason a staging pool's buffers must: the
    // streamer drains before teardown, and the waiter is stopped first.
    ~EventPool();

    EventPool(const EventPool &) = delete;
    EventPool & operator=(const EventPool &) = delete;

    // A free event, creating one when none is free and the ceiling allows. NEVER waits.
    //
    // Success with a NULL event means every one is in flight, which the ceiling makes impossible when
    // it is the caller's window - the same rule the staging pool's ceiling follows.
    common::ResponseCode acquire(device::EventHandle & out);

    void release(device::EventHandle event);

    unsigned created() const;

 private:
    const std::shared_ptr<device::Device> _device;
    const unsigned _max;

    mutable std::mutex _mutex;
    std::vector<device::OwnedEvent> _owned;      // every event this pool made
    std::vector<device::EventHandle> _free;      // and the ones not in flight
};

} // namespace runai::llm::streamer::impl
