/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/io_uring_probe/io_uring_probe.h"

#include <liburing.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// Availability does not depend on depth, so probe with the smallest useful ring. The engine asks for
// the configured depth and reports its own failure.
constexpr unsigned ProbeEntries = 8;

// EPERM and EACCES mean "this kernel has io_uring but this process may not use it" - seccomp, or
// kernel.io_uring_disabled. Everything else (ENOSYS above all) means it is simply not here.
common::ResponseCode reason_for(int error)
{
    return (error == EPERM || error == EACCES) ? common::ResponseCode::FileAccessError : common::ResponseCode::UnknownError;
}

// Register one small buffer the way the engine does, and give it straight back.
//
// IORING_OP_READ_FIXED being supported is not enough, for two reasons. The engine registers through a
// sparse table, whose opcodes are younger than both that one and the classic registration - so a
// kernel can offer fixed reads and still refuse the table. And registration charges RLIMIT_MEMLOCK,
// which no kernel version can tell us about.
//
// One page is small enough to pass under any limit that permits registration at all, which is the
// question here - a pool sized against the limit is the caller's problem, not the probe's.
bool trial_registration(struct io_uring * ring, std::string & why_not)
{
    alignas(4096) static unsigned char buffer[4096];

    struct iovec iov;
    iov.iov_base = buffer;
    iov.iov_len = sizeof(buffer);

    int ret = io_uring_register_buffers_sparse(ring, 1);
    if (ret < 0)
    {
        why_not = std::string("io_uring_register_buffers_sparse failed: ") + std::strerror(-ret);
        return false;
    }

    ret = io_uring_register_buffers_update_tag(ring, 0, &iov, nullptr, 1);
    io_uring_unregister_buffers(ring);

    if (ret < 0)
    {
        why_not = std::string("io_uring_register_buffers_update_tag failed: ") + std::strerror(-ret);
        return false;
    }

    return true;
}

} // namespace

IoUringCapability probe_io_uring()
{
    IoUringCapability capability;

    // liburing returns -errno rather than setting errno.
    struct io_uring ring;
    const int ret = io_uring_queue_init(ProbeEntries, &ring, 0);
    if (ret < 0)
    {
        capability.error = reason_for(-ret);
        LOG(WARNING) << "io_uring is not available: io_uring_setup failed: " << std::strerror(-ret)
                     << ". Checking the kernel version instead would have said otherwise - a seccomp"
                     << " profile or kernel.io_uring_disabled blocks the syscall on kernels that"
                     << " support it";
        return capability;
    }

    capability.timed_wait_is_free = (ring.features & IORING_FEAT_EXT_ARG) != 0;

    // Probe against the ring we already have. io_uring_get_probe() would set up and tear down a
    // SECOND ring to answer the same question.
    struct io_uring_probe * probe = io_uring_get_probe_ring(&ring);
    const bool op_read = (probe != nullptr) && io_uring_opcode_supported(probe, IORING_OP_READ);
    const bool op_read_fixed = (probe != nullptr) && io_uring_opcode_supported(probe, IORING_OP_READ_FIXED);
    if (probe != nullptr)
    {
        io_uring_free_probe(probe);
    }

    // Only for a ring we are going to keep. The two gates below decline io_uring outright, and a
    // capability reported for a ring nobody will build is a state the struct should not be able to
    // describe - besides costing a syscall on every host we reject.
    // Why fixed buffers are off, for the one line at the end. Carried rather than logged here: three
    // lines to say one thing is what turns a log nobody reads into the normal state.
    std::string no_fixed_buffers;

    if (op_read && capability.timed_wait_is_free)
    {
        if (!op_read_fixed)
        {
            no_fixed_buffers = "this kernel has no IORING_OP_READ_FIXED";
        }
        else
        {
            capability.fixed_buffers = trial_registration(&ring, no_fixed_buffers);
        }
    }

    io_uring_queue_exit(&ring);

    if (!op_read)
    {
        // A ring we cannot read through is of no use to us. Reachable only below 5.6, well under our
        // floor - but the rule is to probe rather than to trust the version.
        capability.error = common::ResponseCode::UnknownError;
        LOG(WARNING) << "io_uring is not available: a ring was created but IORING_OP_READ is not"
                     << " supported";
        return capability;
    }

    if (!capability.timed_wait_is_free)
    {
        // Every blocking wait we make is a bounded one (AsyncIoWorker passes WaitTimeoutMs), so
        // without IORING_FEAT_EXT_ARG every one of them would submit a timeout SQE - flushing any
        // staged read past flush() and its bookkeeping, and posting a CQE the harvest would route as
        // if it were a read.
        //
        // This is where the design's kernel floor is enforced. 5.8 says the free path is always
        // taken "at 5.7's >= 5.15 floor"; nothing checked it, so the real floor was IORING_OP_READ's
        // 5.6, and 5.6 to 5.10 ran a path no test we can host will ever reach. Declining is what
        // that note asked for - lowering the floor should be a decision, not an accident.
        //
        // UnknownError, like the op_read branch above: no operator can add EXT_ARG to a kernel
        // that predates it. The default chain then picks libaio_direct on its own.
        capability.error = common::ResponseCode::UnknownError;
        LOG(WARNING) << "io_uring is not available: this kernel lacks IORING_FEAT_EXT_ARG (added in"
                     << " 5.11), so every bounded wait would submit a timeout request into our own"
                     << " ring";
        return capability;
    }

    capability.available = true;

    LOG(INFO) << "io_uring is available; fixed buffers "
              << (capability.fixed_buffers ? "yes" : "no (" + no_fixed_buffers + "), so reads use IORING_OP_READ");
    return capability;
}

IoUringCapability IoUringProbe::capability()
{
    std::unique_lock<std::mutex> lock(_mutex);

    if (!_probed)
    {
        _capability = probe_io_uring();
        _probed = true;
    }

    return _capability;
}

void IoUringProbe::mark_unavailable(common::ResponseCode reason)
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

    // Marks it probed even if it never was. Reaching here normally means an engine was built, so the
    // probe has run - but recording it either way stops a later capability() from probing afresh and
    // cheerfully answering "available" after a real failure.
    _probed = true;
    _capability.available = false;
    _capability.error = reason;

    // Cleared with it: fixed buffers describe a ring, and there is no ring any more. Leaving it set
    // would let a caller that checks only this field register against an engine that was never built.
    _capability.fixed_buffers = false;

    // Reports WHAT was disabled and why, and says nothing about the host.
    //
    // It used to add "the probe succeeded, so the host supports io_uring". That is true of the only
    // production caller - make_io_engine consults capability() first and returns early unless it
    // reports available - but this function never checks it, and the guard above deliberately falls
    // through when _probed is false. The claim also belongs to the caller, which logs "io_uring is
    // available but a ring of depth N could not be built" immediately before calling this. Two
    // adjacent lines said the same thing and only one of them had checked it.
    LOG(WARNING) << "Disabling io_uring for the rest of this process: " << reason
                 << ". A ring of the configured depth could not be built, and retrying will not"
                 << " succeed later";
}

IoUringProbe & IoUringProbe::instance()
{
    static IoUringProbe probe;
    return probe;
}

}; // namespace runai::llm::streamer::posix_io
