/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/libaio_probe/libaio_probe.h"

#include <libaio.h>

#include <cerrno>
#include <cstring>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// Availability does not depend on depth, so ask for the smallest context there is. The engine asks
// for the configured depth, halves on refusal, and reports its own failure.
constexpr int ProbeEvents = 1;

// EAGAIN means the node's aio budget is used up - /proc/sys/fs/aio-max-nr is node wide and shared
// with every other pod, so an operator can raise it. Anything else (ENOSYS above all) means this
// kernel has no aio at all, which nobody can change from here.
common::ResponseCode reason_for(int error)
{
    return error == EAGAIN ? common::ResponseCode::FileAccessError : common::ResponseCode::UnknownError;
}

} // namespace

LibaioCapability probe_libaio()
{
    LibaioCapability capability;

    // io_setup needs a zeroed context and returns -errno rather than setting errno - both measured.
    io_context_t ctx = nullptr;

    const int ret = io_setup(ProbeEvents, &ctx);
    if (ret < 0)
    {
        capability.error = reason_for(-ret);
        LOG(WARNING) << "libaio is not available: io_setup failed: " << std::strerror(-ret)
                     << (-ret == EAGAIN ? ". /proc/sys/fs/aio-max-nr is reached on this node, and it"
                                          " is shared with everything else running on it"
                                        : "");
        return capability;
    }

    // Destroyed at once. The probe answers whether a context can exist, and holding one would take an
    // event from the node's budget for as long as the process lives.
    const int destroyed = io_destroy(ctx);
    if (destroyed < 0)
    {
        // Nothing to do about it, and it does not change the answer - a context WAS created. Logged
        // because it would otherwise be a silent leak of one event from a node-wide budget.
        LOG(WARNING) << "io_destroy failed while probing libaio: " << std::strerror(-destroyed);
    }

    capability.available = true;

    LOG(INFO) << "libaio is available";
    return capability;
}

LibaioCapability LibaioProbe::capability()
{
    std::unique_lock<std::mutex> lock(_mutex);

    if (!_probed)
    {
        _capability = probe_libaio();
        _probed = true;
    }

    return _capability;
}

void LibaioProbe::mark_unavailable(common::ResponseCode reason)
{
    std::unique_lock<std::mutex> lock(_mutex);

    // Already out of the chain, so keep the first reason - it says what actually went wrong, while
    // later failures are consequences of already being disabled.
    //
    // The _probed test is not redundant: an un-probed capability is ALSO !available, since that is
    // how the struct default-constructs. Without it the first demotion on a host where the probe has
    // not run yet would be read as "already unavailable" and its reason thrown away.
    if (_probed && !_capability.available)
    {
        return;
    }

    // Marks it probed even if it never was, so a later capability() cannot probe afresh and answer
    // "available" after a real failure.
    _probed = true;
    _capability.available = false;
    _capability.error = reason;

    // Reports WHAT was disabled and why, and says nothing about the host - same reasoning as
    // IoUringProbe::mark_unavailable. The "a one-event context worked, so the host supports aio"
    // part was a claim this function never checks, and its caller already logs "libaio is available
    // but a context for depth N could not be built" on the line before.
    LOG(WARNING) << "Disabling libaio for the rest of this process: " << reason
                 << ". A context of the configured depth could not be built, and the node's aio"
                 << " budget does not grow on its own, so retrying will not succeed later";
}

LibaioProbe & LibaioProbe::instance()
{
    static LibaioProbe probe;
    return probe;
}

}; // namespace runai::llm::streamer::posix_io
