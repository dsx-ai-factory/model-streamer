/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>

#include "common/response_code/response_code.h"

namespace runai::llm::streamer::posix_io
{

// Whether libaio can be used on this host.
//
// Simpler than the io_uring question, and asked for the same reason. io_uring is refused by container
// seccomp profiles and by kernel.io_uring_disabled, so it must be probed rather than version-checked.
// libaio is not in Docker's blocked set - measured, io_setup returns 0 both with and without the
// default seccomp profile - so it is almost always available.
//
// "Almost" is the reason this exists. io_setup draws on /proc/sys/fs/aio-max-nr, which is node wide
// and shared with every other pod. A node that has exhausted it refuses io_setup with EAGAIN, and
// then libaio is unavailable however healthy the kernel is.
struct LibaioCapability
{
    // A context was created and destroyed here.
    //
    // NOT a promise that a context of the CONFIGURED depth will succeed - the probe asks for one
    // event. The engine halves and retries from the configured depth, so it will normally get
    // something; if even that fails, mark_unavailable() takes libaio out of the chain.
    bool available = false;

    // Why not, when !available. EAGAIN means the node's aio limit is reached, which an operator can
    // raise; anything else means this kernel does not provide aio at all.
    common::ResponseCode error = common::ResponseCode::Success;
};

// Create a one-event context, then destroy it.
//
// Stateless and uncached, so a test exercises the probe rather than a memo of it. Two syscalls.
LibaioCapability probe_libaio();

// The answer, probed once and then remembered.
//
// Cached because it is read on two paths - strategy resolution and every make_io_engine() - and under
// one engine per mount the second runs once per mount. What the host provides cannot change while we
// run, so one probe answers all of them.
//
// A class rather than a free function with a static, so a test can own an instance and exercise
// caching and demotion without leaving process-wide state for the next test. Same shape as
// IoUringProbe on purpose: two probes that answer the same kind of question should not need to be
// read in two different ways.
//
// Thread safe.
class LibaioProbe
{
 public:
    // Probes on first call. Returned by value, because the cached copy can be demoted by another
    // thread and a reference into it would be a data race for a struct this small.
    LibaioCapability capability();

    // Record that building a real engine failed, and stop offering libaio for the rest of the
    // process.
    //
    // The probe only says a context can be made at all; it cannot say one of the configured depth
    // will fit. When that fails, retrying is pointless - the node's aio budget does not grow on its
    // own - and worse than pointless during resolution, which would keep choosing a strategy it
    // cannot build.
    //
    // ONE WAY. Demotes available -> unavailable and never back, so the answer cannot flap.
    void mark_unavailable(common::ResponseCode reason);

    // The process-wide instance, which is the one production uses.
    static LibaioProbe & instance();

 private:
    std::mutex _mutex;
    bool _probed = false;
    LibaioCapability _capability;
};

}; // namespace runai::llm::streamer::posix_io
