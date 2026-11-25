/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/libaio_probe/libaio_probe.h"

#include <gtest/gtest.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace runai::llm::streamer::posix_io
{

namespace
{

#ifndef __NR_io_setup
#define __NR_io_setup 206       // x86_64; aarch64 is 0, which the header provides
#endif

// Ask the kernel directly, without libaio and without the code under test.
//
// This is the point of it. If the expectation were probe_libaio().available, the test would assert
// the probe against itself and pass however wrong the probe was. An expectation must never be
// computed by the thing it is checking.
//
// The raw syscall takes a pointer to an aio_context_t, which must be zero going in.
int raw_io_setup_errno()
{
    unsigned long ctx = 0;

    if (::syscall(__NR_io_setup, 1, &ctx) < 0)
    {
        return errno;
    }

    ::syscall(__NR_io_destroy, ctx);
    return 0;
}

} // namespace

// The probe must agree with the kernel, in both directions.
//
// Unlike io_uring there is no seccomp branch to exercise here: Docker's default profile does not
// block io_setup - measured, it returns 0 with and without it - so this normally runs the available
// branch everywhere, including CI. That is the reason this engine exists.
TEST(LibaioProbe, Agrees_With_The_Kernel)
{
    const int raw = raw_io_setup_errno();
    const auto capability = probe_libaio();

    EXPECT_EQ(capability.available, raw == 0)
        << "io_setup errno was " << raw << " (" << std::strerror(raw) << ")";

    if (raw == 0)
    {
        EXPECT_EQ(capability.error, common::ResponseCode::Success);
    }
    else
    {
        EXPECT_NE(capability.error, common::ResponseCode::Success) << "unavailable must carry a reason";

        // A used-up node budget is an operator's problem; a kernel without aio is nobody's. Reporting
        // the second when it is the first sends whoever reads the log to the wrong place.
        const auto expected = (raw == EAGAIN) ? common::ResponseCode::FileAccessError
                                              : common::ResponseCode::UnknownError;
        EXPECT_EQ(capability.error, expected);
    }
}

// The probe must not hold onto an event. /proc/sys/fs/aio-max-nr is node wide and shared with every
// other pod, so a context left open would take from that budget for the life of the process - and
// under one engine per mount the probe is consulted many times.
TEST(LibaioProbe, Probing_Repeatedly_Does_Not_Exhaust_The_Budget)
{
    for (int i = 0; i < 64; ++i)
    {
        const auto capability = probe_libaio();
        ASSERT_EQ(capability.available, probe_libaio().available)
            << "the answer changed on round " << i << ", so a context is being leaked";
    }
}

// The reason survives to the caller: this is what strategy resolution reports when it rejects the
// candidate.
TEST(LibaioProbe, Demotion_Keeps_The_Reason)
{
    LibaioProbe probe;

    probe.mark_unavailable(common::ResponseCode::FileAccessError);

    const auto capability = probe.capability();
    EXPECT_FALSE(capability.available);
    EXPECT_EQ(capability.error, common::ResponseCode::FileAccessError);
}

// One way only. A demoted probe that re-probed would answer "available" again, and resolution would
// keep choosing an engine that has already failed to build.
TEST(LibaioProbe, Demotion_Is_Permanent)
{
    LibaioProbe probe;

    const auto before = probe.capability();
    if (!before.available)
    {
        // Already unavailable here, so demotion cannot be observed to stick. Not a skip: the
        // remaining assertion still holds, and it is the one that matters on such a host.
        probe.mark_unavailable(common::ResponseCode::UnknownError);
        EXPECT_FALSE(probe.capability().available);
        return;
    }

    probe.mark_unavailable(common::ResponseCode::UnknownError);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FALSE(probe.capability().available) << "re-probed after demotion, on call " << i;
    }
}

// The first reason is the informative one - it says what actually went wrong. Later failures are
// consequences of already being disabled.
TEST(LibaioProbe, Second_Demotion_Does_Not_Overwrite)
{
    LibaioProbe probe;

    probe.mark_unavailable(common::ResponseCode::FileAccessError);
    probe.mark_unavailable(common::ResponseCode::UnknownError);

    EXPECT_EQ(probe.capability().error, common::ResponseCode::FileAccessError);
}

// Instances are independent, which is what lets the tests above avoid touching process-wide state.
TEST(LibaioProbe, Instances_Do_Not_Share_State)
{
    LibaioProbe demoted;
    LibaioProbe fresh;

    demoted.mark_unavailable(common::ResponseCode::UnknownError);

    EXPECT_FALSE(demoted.capability().available);
    EXPECT_EQ(fresh.capability().available, probe_libaio().available);
}

}; // namespace runai::llm::streamer::posix_io
