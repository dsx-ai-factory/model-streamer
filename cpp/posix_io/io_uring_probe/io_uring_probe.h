/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>

#include "common/response_code/response_code.h"

namespace runai::llm::streamer::posix_io
{

// What io_uring can do on this host.
//
// PROBED, NEVER VERSION-CHECKED. A kernel whose uname is fine can still refuse: container seccomp
// profiles omit io_uring_setup from their allowlist, and kernel.io_uring_disabled turns it off
// outright. Measured in our own dev container - io_uring_setup returns EPERM while
// /proc/sys/kernel/io_uring_disabled reads 0 on a 6.8 kernel, so every version test says "available"
// and is wrong. The only correct question is whether the syscall works.
struct IoUringCapability
{
    // The engine can be built here: a ring was created and IORING_OP_READ is supported.
    //
    // NOT a promise that a ring of the CONFIGURED depth will succeed - the probe uses a small one.
    // Depth can still fail on ENOMEM or RLIMIT_MEMLOCK, and only the engine knows the depth. That is
    // what IoUringProbe::mark_unavailable() is for.
    bool available = false;

    // Why not, when !available. The distinction is operational rather than cosmetic:
    //   FileAccessError - the kernel has io_uring but this process may not use it (seccomp, or
    //                     kernel.io_uring_disabled): an operator can change this
    //   UnknownError    - no io_uring here at all: nobody can
    common::ResponseCode error = common::ResponseCode::Success;

    // IORING_FEAT_EXT_ARG (kernel 5.11): the timeout rides along on io_uring_enter, so a bounded
    // wait costs no submission slot and posts no completion.
    //
    // Below 5.11 it is not merely a cost. liburing submits an IORING_OP_TIMEOUT SQE instead, and
    // doing so flushes everything else staged in the ring - "we will do that on its behalf", in its
    // own words (src/queue.c). Staged reads would then reach the kernel without passing through
    // flush(), and the timeout posts a CQE our harvest cannot tell from a read.
    //
    // So this gates `available` rather than describing it: false means we decline io_uring. It is
    // never true and unused.
    bool timed_wait_is_free = false;

    // IORING_OP_READ_FIXED works here, and a buffer could actually be registered.
    //
    // DESCRIPTIVE, not a gate: a ring without it reads perfectly well with plain IORING_OP_READ. Unlike
    // the two above, false costs us an optimisation rather than correctness.
    //
    // Answered by REGISTERING ONE FOR REAL rather than by a version test. Registration can fail for
    // reasons no version knows - RLIMIT_MEMLOCK above all, which io_uring charges and CUDA pinning does
    // not.
    //
    // NOT a promise that the staging pool will register, for the same reason `available` is not a
    // promise about depth: the trial is a few KiB and a pool is megabytes against a limit. A caller
    // that registers for real has to handle its own failure.
    bool fixed_buffers = false;
};

// Create a small ring, read its features, probe the opcode, tear it down.
//
// Stateless and uncached, so a test exercises the probe itself rather than a memo of it. Cheap but
// not free - a syscall and an mmap.
IoUringCapability probe_io_uring();

// The answer, probed once and then remembered.
//
// Cached because it is consulted on two paths - strategy resolution (5.7) and every
// make_io_engine() - and under one engine per mount (5.2.3) the second of those runs once per mount.
// The host's io_uring support cannot change while we run, so one probe answers all of them.
//
// A class rather than a free function with a static, so a test can own an instance and exercise
// caching and demotion without leaving process-wide state behind for the next test.
//
// Thread safe.
class IoUringProbe
{
 public:
    // Probes on first call. Returned BY VALUE: the cached copy can be demoted by another thread, and
    // a reference into it would be a data race for a struct this small.
    IoUringCapability capability();

    // Record that building a REAL engine failed, and stop offering io_uring for the rest of the
    // process.
    //
    // The probe only says io_uring works at all; it cannot say a ring of the configured depth will
    // fit. When that turns out to fail, retrying is pointless - a ring that did not fit will not fit
    // later, and RLIMIT_MEMLOCK does not rise on its own - and worse than pointless during
    // resolution, which would keep choosing a strategy it cannot instantiate.
    //
    // ONE WAY. Demotes available -> unavailable and never the reverse, so the answer cannot flap and
    // a candidate that has genuinely failed stays out of the chain.
    void mark_unavailable(common::ResponseCode reason);

    // The process-wide instance, which is the one production uses.
    static IoUringProbe & instance();

 private:
    std::mutex _mutex;
    bool _probed = false;
    IoUringCapability _capability;
};

}; // namespace runai::llm::streamer::posix_io
