/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "device/device.h"

namespace runai::llm::streamer::device
{

// Owners for the handles a Device hands out.
//
// Every one of them is a raw void * that only the device can release, so without an owner they
// leak - silently, and only in a process that lives long enough to notice. Each deleter keeps its
// own share of the device, because releasing is a call on it and it must still be there.
//
// unique_ptr rather than a hand-written destructor: it frees exactly once, leaves a moved-from
// holder empty, and needs no move constructor to be written correctly by hand.

struct StreamDeleter
{
    std::shared_ptr<Device> device;

    void operator()(void * stream) const
    {
        if (device != nullptr && stream != nullptr)
        {
            device->stream_destroy(stream);
        }
    }
};

struct EventDeleter
{
    std::shared_ptr<Device> device;

    void operator()(void * event) const
    {
        if (device != nullptr && event != nullptr)
        {
            device->event_destroy(event);
        }
    }
};

struct PinnedDeleter
{
    std::shared_ptr<Device> device;

    void operator()(void * memory) const
    {
        if (device != nullptr && memory != nullptr)
        {
            device->host_free(memory);
        }
    }
};

using OwnedStream = std::unique_ptr<void, StreamDeleter>;
using OwnedEvent  = std::unique_ptr<void, EventDeleter>;
using OwnedPinned = std::unique_ptr<void, PinnedDeleter>;

} // namespace runai::llm::streamer::device
