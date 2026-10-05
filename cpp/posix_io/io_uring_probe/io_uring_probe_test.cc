#include "posix_io/io_uring_probe/io_uring_probe.h"

#include <gtest/gtest.h>

#include "posix_io/raw_uring/raw_uring.h"

#include <liburing.h>       // for struct io_uring_params only
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace runai::llm::streamer::posix_io
{

namespace
{

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif

// Ask the kernel directly, without liburing and without the code under test.
//
// This is the whole point: if the expectation were `probe_io_uring().available`, the test would
// assert the probe against itself and pass however wrong the probe was. A skip or an expectation
// must never be computed by the thing it is checking.
int raw_io_uring_setup_errno()
{
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));

    const int fd = ::syscall(__NR_io_uring_setup, 8, &params);
    if (fd < 0)
    {
        return errno;
    }

    ::close(fd);
    return 0;
}

// Whether this kernel offers IORING_FEAT_EXT_ARG, asked of the kernel for the same reason as above:
// an expectation must never be computed by the thing it is checking. probe_io_uring() reports the
// same bit as `timed_wait_is_free`, and reading it from there would make the test vacuous.
bool raw_has_ext_arg()
{
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));

    const int fd = ::syscall(__NR_io_uring_setup, 8, &params);
    if (fd < 0)
    {
        return false;
    }

    ::close(fd);
    return (params.features & IORING_FEAT_EXT_ARG) != 0;
}

} // namespace

// The probe must agree with the kernel - in both directions. In a container under Docker's default
// seccomp profile this runs the unavailable branch; with seccomp=unconfined, the available one.
//
// A working io_uring_setup is necessary but NOT sufficient: below 5.11 the ring exists and we still
// decline it, because a bounded wait there would submit a timeout SQE (see io_uring_probe.h). So the
// expectation is gated on EXT_ARG too, which the probe reports separately. Reading that field back
// from the probe would assert it against itself, so this asks the ring directly.
TEST(IoUringProbe, Agrees_With_The_Kernel)
{
    const int raw = raw_io_uring_setup_errno();
    const auto capability = probe_io_uring();

    EXPECT_EQ(capability.available, raw == 0 && raw_has_ext_arg())
        << "io_uring_setup errno was " << raw << " (" << std::strerror(raw) << ")"
        << ", IORING_FEAT_EXT_ARG " << (raw_has_ext_arg() ? "present" : "absent");

    if (raw == 0 && raw_has_ext_arg())
    {
        EXPECT_EQ(capability.error, common::ResponseCode::Success);
    }
    else
    {
        EXPECT_NE(capability.error, common::ResponseCode::Success) << "unavailable must carry a reason";

        // A blocked syscall is an operator's problem; a missing one is nobody's. Reporting the
        // second when it is the first sends whoever reads the log to the wrong place.
        const auto expected = (raw == EPERM || raw == EACCES) ? common::ResponseCode::FileAccessError
                                                              : common::ResponseCode::UnknownError;
        EXPECT_EQ(capability.error, expected);
    }
}

// Fixed buffers are reported from a REAL registration, so the expectation is a real registration too.
//
// Asked of the kernel with the raw syscall, as above. A version test would answer this wrongly on any
// host with a small RLIMIT_MEMLOCK, which is the case the probe exists to catch.
TEST(IoUringProbe, Reports_Fixed_Buffers_As_The_Kernel_Does)
{
    if (raw_io_uring_setup_errno() != 0)
    {
        GTEST_SKIP() << "no io_uring here, so there is nothing to register against";
    }

    EXPECT_EQ(probe_io_uring().fixed_buffers, raw_can_register_buffer());
}

// Fixed buffers describe a ring, so they cannot outlive one. An unavailable probe claiming them would
// invite a caller that checks only this field to register against an engine that was never built.
//
// Both directions of unavailable are covered: declined at probe time (the host), and demoted later
// (mark_unavailable, when a ring of the configured depth could not be built).
TEST(IoUringProbe, Unavailable_Never_Claims_Fixed_Buffers)
{
    const auto probed = probe_io_uring();
    if (!probed.available)
    {
        EXPECT_FALSE(probed.fixed_buffers) << "declined io_uring still reported fixed buffers";
    }

    IoUringProbe probe;
    ASSERT_TRUE(probe.capability().available || !probe.capability().fixed_buffers);

    probe.mark_unavailable(common::ResponseCode::FileAccessError);

    EXPECT_FALSE(probe.capability().available);
    EXPECT_FALSE(probe.capability().fixed_buffers) << "a demoted probe still reported fixed buffers";
}

// Fixed buffers must NOT gate availability: a ring that cannot register still reads with
// IORING_OP_READ. Getting this wrong would silently drop io_uring on every host with a small
// RLIMIT_MEMLOCK - a large regression bought by an optimisation.
TEST(IoUringProbe, Fixed_Buffers_Do_Not_Gate_Availability)
{
    if (raw_io_uring_setup_errno() != 0 || !raw_has_ext_arg())
    {
        GTEST_SKIP() << "io_uring is unavailable here for reasons that have nothing to do with buffers";
    }

    const auto capability = probe_io_uring();

    EXPECT_TRUE(capability.available)
        << "available must not depend on fixed_buffers, which is " << capability.fixed_buffers;
}

// The reason survives to the caller: mark_unavailable is what strategy resolution reports when it
// rejects the candidate.
TEST(IoUringProbe, Demotion_Keeps_The_Reason)
{
    IoUringProbe probe;

    probe.mark_unavailable(common::ResponseCode::FileAccessError);

    const auto capability = probe.capability();
    EXPECT_FALSE(capability.available);
    EXPECT_EQ(capability.error, common::ResponseCode::FileAccessError);
}

// One way only. On a host where io_uring works, a demoted probe that re-probed would answer
// "available" again - and resolution would keep choosing an engine that has already failed to build.
TEST(IoUringProbe, Demotion_Is_Permanent)
{
    IoUringProbe probe;

    const auto before = probe.capability();
    if (!before.available)
    {
        // Already unavailable here, so demotion cannot be observed to stick. Not a skip: the
        // remaining assertions still hold, and they are the ones that matter on such a host.
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
TEST(IoUringProbe, Second_Demotion_Does_Not_Overwrite)
{
    IoUringProbe probe;

    probe.mark_unavailable(common::ResponseCode::FileAccessError);
    probe.mark_unavailable(common::ResponseCode::UnknownError);

    EXPECT_EQ(probe.capability().error, common::ResponseCode::FileAccessError);
}

// Instances are independent, which is what lets the tests above avoid touching process-wide state.
TEST(IoUringProbe, Instances_Do_Not_Share_State)
{
    IoUringProbe demoted;
    IoUringProbe fresh;

    demoted.mark_unavailable(common::ResponseCode::UnknownError);

    EXPECT_FALSE(demoted.capability().available);
    EXPECT_EQ(fresh.capability().available, probe_io_uring().available);
}

}; // namespace runai::llm::streamer::posix_io
