/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/streamer/streamer.h"

#include "posix_io/mock/mock_io_engine.h"

#include "posix_io/alignment/alignment.h"
#include "posix_io/io_uring_probe/io_uring_probe.h"

#include <unistd.h>

#include <gtest/gtest.h>

#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <cstring>
#include <cstdlib>
#include <atomic>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>

#include "common/exception/exception.h"

#include "utils/logging/logging.h"
#include "utils/random/random.h"
#include "utils/fd/fd.h"
#include "utils/thread/thread.h"
#include "utils/temp/env/env.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::impl
{

namespace
{

// Read the next response off the persistent responder (blocking), with its submission_done flag. There is no
// finish-on-drain in the multi-request API: a submission is complete when its last response carries submission_done ==
// true (the equivalent of the old FinishedError-on-drain).
struct Received
{
    common::Response response;
    bool submission_done = false;
};

Received recv(Streamer & streamer)
{
    // common::Response has no default constructor, so build it first and aggregate-init Received (no
    // default-construct-then-assign).
    bool submission_done = false;
    auto response = streamer.response(0, submission_done);
    return Received{ response, submission_done };
}

// The range indices a submission of n ranges owes: exactly {0, 1, ... n-1}. Compared as a SET, because a
// count (received.size() == n) also passes when a range is answered twice and another dropped, or when an
// index is out of range entirely.
std::set<unsigned> range_indices(unsigned n)
{
    std::set<unsigned> indices;
    for (unsigned i = 0; i < n; ++i)
    {
        indices.insert(i);
    }
    return indices;
}


// The kernel directly - not IoUringProbe and not StrategyResolver, both of which are on the path
// under test here.
bool ring_works()
{
    struct params_stub { char opaque[512]; } params;
    std::memset(&params, 0, sizeof(params));

    const int fd = ::syscall(425 /* __NR_io_uring_setup */, 8, &params);
    if (fd < 0)
    {
        return false;
    }
    ::close(fd);
    return true;
}

// short wait used to assert a fresh/empty responder delivers nothing (it times out rather than blocking)
constexpr unsigned EMPTY_WAIT_MS = 50;

} // namespace


// Say that one strategy cannot be served here, and let the probes answer for the rest.
//
// Needed because every strategy is available on a normal host: sync_buffered always, libaio nearly
// always, io_uring wherever seccomp permits. Without this there is no candidate that reliably fails,
// so a submission that must be refused for want of a reader cannot be built.
//
// These tests named `libaio_direct` for that, which held only while libaio had no engine. They broke
// the day it got one, which is the right way round: a test resting on a feature being MISSING should
// fail when it arrives.
Streamer::Environment without(posix_io::Strategy unavailable)
{
    Streamer::Environment environment;
    environment.availability = [unavailable](posix_io::Strategy strategy)
    {
        return strategy == unavailable ? common::ResponseCode::FsStrategyUnavailable
                                       : common::ResponseCode::Success;
    };
    return environment;
}


TEST(Creation, Default)
{
    Config config;
    Streamer streamer(config);
    // fresh streamer, no request: the persistent responder has nothing, so a timed wait times out (it does
    // NOT report FinishedError - that is teardown-only in the multi-request API)
    bool submission_done = false;
    auto r = streamer.response(EMPTY_WAIT_MS, submission_done);
    EXPECT_EQ(r.ret, common::ResponseCode::TimedOut);
    EXPECT_FALSE(submission_done);
}

TEST(Creation, Sanity)
{
    Streamer streamer;
    bool submission_done = false;
    auto r = streamer.response(EMPTY_WAIT_MS, submission_done);
    EXPECT_EQ(r.ret, common::ResponseCode::TimedOut);
    EXPECT_FALSE(submission_done);
}

// A read failure is attributable to its file, so it must NOT be reported as UnknownError. UnknownError is
// reserved for unrecoverable conditions (corruption, out of memory) and tells the caller to abort
// everything - reporting it for one file's I/O error would poison every other in-flight submission.
//
// A directory is the cheapest real read failure available: open(O_RDONLY) succeeds on it, and the read
// then fails with EISDIR - no fault injection needed.
// S6a's whole point: a real submission served by the io_uring engine rather than the synchronous
// reader, with the same bytes out.
//
// The strategy assertion is what makes this test mean anything. Both paths return identical data, so
// checking only the bytes would pass just as well if the request quietly went to the threadpool.
TEST(Async, ReadsThroughIoUringWhenResolvedToIt)
{
    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered,sync_buffered"));

    const auto data = utils::random::buffer(1 << 20);
    utils::temp::File file(data);

    const unsigned ranges = 8;
    const size_t range_size = data.size() / ranges;

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    for (unsigned i = 0; i < ranges; ++i)
    {
        request[0].ranges.push_back(ReadRange{ i * range_size, range_size, dst.data() + i * range_size });
    }

    Streamer streamer;   // reads RUNAI_STREAMER_FS_STRATEGY through Config

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);

    std::set<unsigned> seen;
    for (unsigned i = 0; i < ranges; ++i)
    {
        const auto received = recv(streamer);
        EXPECT_EQ(received.response.ret, common::ResponseCode::Success);
        EXPECT_EQ(received.response.submission_id, submission_id);
        seen.insert(received.response.index);
    }
    EXPECT_EQ(seen, range_indices(ranges));

    // Which path actually served it. On a host without a ring the list falls through to
    // sync_buffered, and this test then covers the fallback instead - still a real assertion.
    const bool expect_async = ring_works();

    EXPECT_EQ(streamer.fs_strategy(),
              expect_async ? posix_io::Strategy::IoUringBuffered
                           : posix_io::Strategy::SyncBuffered);

    // What was CHOSEN above; what was USED here. Without this, a dispatch that ignored the resolved
    // strategy and sent everything to the threadpool would pass every assertion in this test.
    EXPECT_EQ(streamer.async_pool_used(), expect_async);

    EXPECT_EQ(std::vector<char>(dst.begin(), dst.end()),
              std::vector<char>(data.begin(), data.end()));
}

// The record answers "which reader served which file" for a submission that used two of them. That
// question only exists because a submission can now be split, and nothing else answers it.
TEST(Async, StatsRecordTheStrategyPerFile)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring unavailable, so every file would report the same reader";
    }

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered,sync_buffered"));

    const auto data = utils::random::buffer(4096);

    utils::temp::Dir disk_dir;
    utils::temp::File on_disk(disk_dir.path, utils::random::string(), data);
    utils::temp::File in_memory("/dev/shm", utils::random::string(), data);

    // One directory is memory backed, so it goes to the synchronous reader whatever the strategy says.
    const std::string memory_dir = "/dev/shm";
    Streamer streamer(Config(), Streamer::Environment{
                      .mount = [memory_dir](const std::string & directory) -> posix_io::MountCapability
                      {
                          return posix_io::MountCapability{ makedev(8, 1), directory == memory_dir };
                      },
    });

    std::vector<char> dst1(data.size());
    std::vector<char> dst2(data.size());

    std::vector<FileRanges> request(2);
    request[0].path = on_disk.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst1.data() });
    request[1].path = in_memory.path;
    request[1].ranges.push_back(ReadRange{ 0, data.size(), dst2.data() });

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);

    for (unsigned i = 0; i < 2; ++i)
    {
        EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);
    }

    SubmissionStats stats;
    ASSERT_TRUE(streamer.stats().find(submission_id, stats));
    ASSERT_EQ(stats.files.size(), 2u);

    EXPECT_EQ(stats.files[0].path, on_disk.path);
    EXPECT_EQ(stats.files[0].strategy, posix_io::Strategy::IoUringBuffered);

    EXPECT_EQ(stats.files[1].path, in_memory.path);
    EXPECT_EQ(stats.files[1].strategy, posix_io::Strategy::SyncBuffered)
        << "a memory-backed file must be recorded as read by the synchronous reader";
}

// A submission that never ran must not appear. Otherwise the record would claim work that never
// happened.
TEST(Async, StatsSkipARejectedSubmission)
{
    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("libaio_direct"));

    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    Streamer streamer(Config(), without(posix_io::Strategy::LibaioDirect));

    SubmissionId submission_id = 0;
    ASSERT_NE(streamer.async_request(request, &submission_id), common::ResponseCode::Success);

    EXPECT_TRUE(streamer.stats().submissions().empty());
}

// A submission spanning two mounts: the path from st_dev, through the per-mount group, to a separate
// engine each. Unreachable on a real host without two filesystems - this container has one non-tmpfs
// mount and no CAP_SYS_ADMIN to make another - so the mount probe is injected.
TEST(Async, TwoMountsGetTwoEngines)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring unavailable, so nothing reaches the async pools";
    }

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered,sync_buffered"));
    utils::temp::Env engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 4UL);

    const auto data = utils::random::buffer(1 << 20);

    // Two DIRECTORIES, because the probe answers per directory. Both files in one directory would
    // share a mount, and this test would pass while proving nothing.
    utils::temp::Dir dir_one;
    utils::temp::Dir dir_two;
    utils::temp::File one(dir_one.path, utils::random::string(), data);
    utils::temp::File two(dir_two.path, utils::random::string(), data);

    // Both files really live on one filesystem; the probe says otherwise, which is the point.
    const std::string first = dir_one.path;
    Streamer streamer(Config(), Streamer::Environment{
                      .mount = [first](const std::string & directory) -> posix_io::MountCapability
                      {
                          return posix_io::MountCapability{ directory == first ? makedev(8, 1) : makedev(8, 2), false };
                      },
    });

    std::vector<char> dst1(data.size());
    std::vector<char> dst2(data.size());

    std::vector<FileRanges> request(2);
    request[0].path = one.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst1.data() });
    request[1].path = two.path;
    request[1].ranges.push_back(ReadRange{ 0, data.size(), dst2.data() });

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);

    for (unsigned i = 0; i < 2; ++i)
    {
        EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);
    }

    // What makes this test mean anything. Without per-mount engines both files share one.
    EXPECT_EQ(streamer.async_engines(), 2u);

    EXPECT_EQ(std::vector<char>(dst1.begin(), dst1.end()), std::vector<char>(data.begin(), data.end()));
    EXPECT_EQ(std::vector<char>(dst2.begin(), dst2.end()), std::vector<char>(data.begin(), data.end()));
}

// Two directories on the SAME mount share one engine - groups are keyed on st_dev, not on the path.
TEST(Async, TwoDirectoriesOnOneMountShareAnEngine)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring unavailable, so nothing reaches the async pools";
    }

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered,sync_buffered"));
    utils::temp::Env engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 4UL);

    const auto data = utils::random::buffer(4096);

    utils::temp::Dir dir_one;
    utils::temp::Dir dir_two;
    utils::temp::File one(dir_one.path, utils::random::string(), data);
    utils::temp::File two(dir_two.path, utils::random::string(), data);

    // Different directories, one device.
    Streamer streamer(Config(), Streamer::Environment{
                      .mount = [](const std::string &) -> posix_io::MountCapability
                      {
                          return posix_io::MountCapability{ makedev(8, 1), false };
                      },
    });

    std::vector<char> dst1(data.size());
    std::vector<char> dst2(data.size());

    std::vector<FileRanges> request(2);
    request[0].path = one.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst1.data() });
    request[1].path = two.path;
    request[1].ranges.push_back(ReadRange{ 0, data.size(), dst2.data() });

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);

    for (unsigned i = 0; i < 2; ++i)
    {
        EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);
    }

    EXPECT_EQ(streamer.async_engines(), 1u) << "one filesystem must not get two engines";
}

// The setter is what makes the strategy controllable without an environment variable. It must take
// effect on the submission that follows it - resolution happens on the first request, not at
// construction, precisely so a setter has its chance.
TEST(Async, SetFsStrategyTakesEffect)
{
    utils::temp::UnsetEnv strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"));

    const auto data = utils::random::buffer(1 << 20);
    utils::temp::File file(data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    Streamer streamer;   // the environment says nothing, so the default is the synchronous reader

    ASSERT_EQ(streamer.set_fs_strategy("io_uring_buffered,sync_buffered"), common::ResponseCode::Success);

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    const bool expect_async = ring_works();
    EXPECT_EQ(streamer.fs_strategy(),
              expect_async ? posix_io::Strategy::IoUringBuffered
                           : posix_io::Strategy::SyncBuffered);
    EXPECT_EQ(streamer.async_pool_used(), expect_async);

    EXPECT_EQ(std::vector<char>(dst.begin(), dst.end()),
              std::vector<char>(data.begin(), data.end()));
}

// Set once, like credentials - and rejected after resolution even for a FIRST call, because by then an
// engine exists for the resolved answer. A setter that reported success and changed nothing would be
// worse than one that refuses.
TEST(Async, SetFsStrategyIsRejectedAfterTheFirstRequest)
{
    utils::temp::UnsetEnv strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"));

    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    // Read what resolved rather than naming it. The default list prefers io_uring and falls back, so
    // the answer depends on the host - and this test is about the SETTER's rules, not about which
    // strategy won.
    const auto resolved = streamer.fs_strategy();

    // Something the resolution cannot have produced, so the rejection is about the rule and not about
    // a value that happened to differ.
    const char * different = resolved == posix_io::Strategy::IoUringBuffered ? "sync_buffered"
                                                                             : "io_uring_buffered";

    // Nothing was ever set, so there is no earlier value to conflict with - only the resolution.
    EXPECT_NE(streamer.set_fs_strategy(different), common::ResponseCode::Success);
    EXPECT_EQ(streamer.fs_strategy(), resolved) << "a refused set must not change the strategy";

    // The value already in force is still accepted, since it changes nothing. It is the LIST that was
    // resolved from, not the single strategy that won: the resolver compares candidate lists, so
    // passing the winner's name alone would read as a different request and be refused.
    EXPECT_EQ(streamer.set_fs_strategy(Config::default_fs_strategy_candidates), common::ResponseCode::Success);
}

// A typo must not become a fallback nobody asked for.
TEST(Async, SetFsStrategyRejectsAnUnknownName)
{
    Streamer streamer;
    EXPECT_EQ(streamer.set_fs_strategy("io_uring_bufferd"), common::ResponseCode::InvalidParameterError);
}

// tmpfs is pure memcpy: there is no device to overlap, so depth buys nothing and parallelism does.
// It goes to the 16-thread pool however the strategy resolved - which is the routing rule that needs
// the mount probe at all.
TEST(Async, TmpfsGoesToTheSynchronousPool)
{
    if (::system("test \"$(stat -f -c %T /dev/shm)\" = tmpfs") != 0)
    {
        GTEST_SKIP() << "/dev/shm is not tmpfs here, so there is no memory-backed mount to route from";
    }

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered,sync_buffered"));

    const auto data = utils::random::buffer(1 << 20);
    utils::temp::File file("/dev/shm", utils::random::string(), data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    // The strategy still resolves to io_uring - the mount decides the POOL, not the strategy.
    if (ring_works())
    {
        EXPECT_EQ(streamer.fs_strategy(), posix_io::Strategy::IoUringBuffered);
    }

    EXPECT_FALSE(streamer.async_pool_used())
        << "a tmpfs file must not be read through the ring";

    EXPECT_EQ(std::vector<char>(dst.begin(), dst.end()),
              std::vector<char>(data.begin(), data.end()));
}

// The default list prefers io_uring_direct, so on a host that can serve it the async path is what a
// caller gets without asking. Pinned because the default is a performance decision: it should change
// deliberately, with a measurement, not by someone editing the list for an unrelated reason.
//
// Skipped rather than branched where the ring is missing: the fallback entries are covered by
// StrategyResolver's own tests, and asserting "something else resolved" here would pass for the wrong
// reason on a host where io_uring is merely broken.
TEST(Async, DefaultStrategyPrefersIoUringDirect)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring unavailable, so the default cannot resolve to it here";
    }

    utils::temp::UnsetEnv strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"));

    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    EXPECT_EQ(streamer.fs_strategy(), posix_io::Strategy::IoUringDirect);
    EXPECT_TRUE(streamer.async_pool_used()) << "the default resolves to an async strategy, so the pool must be built";
}

// An unservable list is an error, not a quiet fall-through to the synchronous reader - and it must
// fail the REQUEST, since that is the only place the caller can see it.
// A request that reads nothing must not fail on a reader it will never use. An empty `s3://` entry
// classifies as a file system submission, because is_object_storage_submission ignores files with no
// ranges - so resolving before the empty-submission return refused a no-op object-storage request.
TEST(Async, AnEmptySubmissionDoesNotResolveTheStrategy)
{
    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("libaio_direct"));

    Streamer streamer(Config(), without(posix_io::Strategy::LibaioDirect));

    for (const auto * path : { "s3://bucket/key", "/no/such/file" })
    {
        std::vector<FileRanges> request(1);
        request[0].path = path;   // no ranges: nothing is read, so nothing needs a reader

        SubmissionId submission_id = 0;
        EXPECT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success) << path;
        EXPECT_NE(submission_id, 0u) << path << ": the id is still minted and handed back";
    }
}

TEST(Async, UnservableStrategyFailsTheRequest)
{
    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("libaio_direct"));

    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    Streamer streamer(Config(), without(posix_io::Strategy::LibaioDirect));

    SubmissionId submission_id = 123;   // must be cleared, so a stale id cannot be mistaken for a real one

    // NOT merely "!= Success". Ignoring the resolution failure also produces a non-Success code -
    // dispatch asserts on the unresolved strategy and the catch block reports UnknownError - but that
    // happens AFTER the submission is registered and its responses counted, and UnknownError tells
    // the caller to abort everything rather than just this request. The specific code is what
    // separates a clean refusal from a late collapse.
    EXPECT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::FsStrategyUnavailable);

    // Nothing was committed: no id was minted, so the caller owes nothing and nothing owes it.
    EXPECT_EQ(submission_id, 0u);

    // And no response is waiting - a submission that was never accepted must not have produced one.
    bool done = false;
    EXPECT_EQ(streamer.response(EMPTY_WAIT_MS, done).ret, common::ResponseCode::TimedOut);
}

TEST(Async, ReadFailureIsAttributableNotUnknown)
{
    utils::temp::Dir dir;

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    const size_t size = 128;
    std::vector<unsigned char> dst(size);

    std::vector<FileRanges> request;
    request.push_back(FileRanges{ dir.path, { ReadRange{ 0, size, dst.data() } } });

    EXPECT_EQ(streamer.async_request(request), common::ResponseCode::Success);

    bool done = false;
    const auto response = streamer.response(60000, done);

    EXPECT_NE(response.ret, common::ResponseCode::Success);
    EXPECT_NE(response.ret, common::ResponseCode::UnknownError)
        << "a per-file read failure must not tell the caller to abort everything";
    // the submission still completes - the caller's buffer is released only on this flag
    EXPECT_TRUE(done);
}

TEST(Async, ResponseBlocksUntilResponseArrives)
{
    // response(0) on a streamer with no ready response BLOCKS until one arrives (persistent responder, no
    // finish-on-drain). A background consumer waits; the main thread submits a read; the consumer receives it.
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);
    Streamer streamer(config);

    std::vector<unsigned char> dst(size);
    std::vector<size_t> sizes = { size };

    std::atomic<bool> got_response{false};
    common::ResponseCode ret = common::ResponseCode::UnknownError;
    utils::Thread consumer([&]()
    {
        bool submission_done = false;
        auto r = streamer.response(0, submission_done);   // blocks until the read below completes
        ret = r.ret;
        got_response = true;
    });

    // let the consumer reach the blocking wait, then confirm it is still blocked (nothing submitted yet)
    ::usleep(50 * 1000);
    EXPECT_FALSE(got_response.load());

    EXPECT_EQ(streamer.async_read(file.path, 0, size, dst.data(), 1, sizes.data()), common::ResponseCode::Success);

    consumer.join();
    EXPECT_TRUE(got_response.load());
    EXPECT_EQ(ret, common::ResponseCode::Success);
}

TEST(Sync, Sanity)
{
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    const auto expected = utils::Fd::read(file.path);
    EXPECT_EQ(expected.size(), size);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(2, 30), utils::random::number(2, 30), chunk_size, bulk_size, false /* do not enforce minimum */);
    Streamer streamer(config);

    std::vector<unsigned char> v(size);
    auto result = streamer.sync_read(file.path, 0, size, v.data());
    EXPECT_EQ(result, common::ResponseCode::Success);

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(v[i], expected[i]);
        if (v[i] != expected[i])
        {
            break;
        }
    }
}

TEST(Sync, File_Not_Found_Error)
{
    auto size = utils::random::number(100, 1000);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(2, 30), utils::random::number(2, 30), chunk_size, bulk_size, false /* do not enforce minimum */);
    Streamer streamer(config);
    std::vector<char> v(size);
    auto r = streamer.sync_read(utils::random::string(), 0, size, v.data());
    EXPECT_EQ(r, common::ResponseCode::FileAccessError);
}

TEST(Sync, End_Of_File_Error)
{
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size / 2);
    utils::temp::File file(data);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(2, 30), utils::random::number(2, 30), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    std::vector<char> v(size);

    for (size_t file_offset : {0UL, utils::random::number<size_t>(size, 100 * size)})
    {
        auto r = streamer.sync_read(file.path, file_offset, size, v.data());
        EXPECT_EQ(r, common::ResponseCode::EofError);
    }

    for (size_t file_offset : {utils::random::number<size_t>(size/2, size), utils::random::number<size_t>(size, 100 * size)})
    {
        auto r = streamer.sync_read(file.path, file_offset, utils::random::number<size_t>(1, size/2), v.data());
        EXPECT_EQ(r, common::ResponseCode::EofError);
    }
}

TEST(Sync, Offset)
{
    auto size = 1024;
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    auto offset_end = utils::random::number<size_t>(2, size);
    auto offset_start = utils::random::number<size_t>(offset_end - 1);
    auto size_to_read = offset_end - offset_start;

    const auto expected = utils::Fd::read(file.path);
    EXPECT_EQ(expected.size(), size);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);


    std::vector<unsigned char> v(size_to_read);
    {
        Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);
        Streamer streamer(config);

        auto r = streamer.sync_read(file.path, offset_start, size_to_read, v.data());
        EXPECT_EQ(r, common::ResponseCode::Success);
    }

    for (size_t i = 0; i < size_to_read; ++i)
    {
        EXPECT_EQ(v[i], expected[i + offset_start]);
        if (v[i] != expected[i + offset_start])
        {
            break;
        }
    }
}

TEST(Async, Sanity)
{
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    const auto expected = utils::Fd::read(file.path);
    EXPECT_EQ(expected.size(), size);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    std::vector<unsigned char> dst(size);
    std::vector<size_t> sizes;
    sizes.push_back(size);
    EXPECT_EQ(streamer.async_read(file.path, 0, size, dst.data(), 1, sizes.data()), common::ResponseCode::Success);
    auto received = recv(streamer);
    EXPECT_EQ(received.response.ret, common::ResponseCode::Success);
    EXPECT_EQ(received.response.index, 0);
    EXPECT_TRUE(received.submission_done);   // single-range submission: this is its last response

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(dst[i], expected[i]);
        if (dst[i] != expected[i])
        {
            break;
        }
    }
}

TEST(Async, Requests)
{
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    const auto expected = utils::Fd::read(file.path);
    EXPECT_EQ(expected.size(), size);

    // create internal division
    const unsigned num_chunks = utils::random::number(1, 20);
    EXPECT_LT(num_chunks, size);
    auto chunks = utils::random::chunks(size, num_chunks);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);
    Streamer streamer(config);


    std::vector<unsigned char> dst(size);
    EXPECT_EQ(streamer.async_read(file.path, 0, size, dst.data(), num_chunks, chunks.data()), common::ResponseCode::Success);

    // wait for all the requests to finish
    std::set<int> expected_responses;

    for (unsigned i = 0; i < num_chunks; ++i)
    {
        expected_responses.insert(i);
    }

    unsigned done_count = 0;
    for (unsigned i = 0; i < num_chunks; ++i)
    {
        const auto received = recv(streamer);
        EXPECT_EQ(received.response.ret, common::ResponseCode::Success);
        LOG(SPAM) << "received response of request " << received.response.index;
        EXPECT_EQ(expected_responses.count(received.response.index), 1);
        expected_responses.erase(received.response.index);
        if (received.submission_done) ++done_count;
    }

    EXPECT_TRUE(expected_responses.empty());
    EXPECT_EQ(done_count, 1u);   // submission_done fires exactly once, on the submission's last response

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(dst[i], expected[i]);
        if (dst[i] != expected[i])
        {
            break;
        }
    }
}

TEST(Async, File_Not_Found_Error)
{
    auto size = utils::random::number(100, 1000);

    // create internal division
    const unsigned num_chunks = utils::random::number(1, 20);
    EXPECT_LT(num_chunks, size);
    auto chunks = utils::random::chunks(size, num_chunks);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    std::vector<char> dst(size);
    EXPECT_EQ(streamer.async_read(utils::random::string(), 0, size, dst.data(), num_chunks, chunks.data()), common::ResponseCode::Success);

    unsigned done_count = 0;
    for (unsigned i = 0; i < num_chunks; ++i)
    {
        const auto received = recv(streamer);
        EXPECT_EQ(received.response.ret, common::ResponseCode::FileAccessError);
        if (received.submission_done) ++done_count;
    }
    EXPECT_EQ(done_count, 1u);   // the failed submission still completes: submission_done on its last response
}

TEST(Async, End_Of_File_Error)
{
    auto size = utils::random::number(100, 1000);

    // create internal division
    const unsigned num_chunks = utils::random::number(1, 20);
    EXPECT_LT(num_chunks, size);

    auto chunks = utils::random::chunks(size, num_chunks);

    // write data just for the first chunks

    const auto chunk_size = utils::random::number<size_t>(10, size - 1);
    const auto block_size = utils::random::number<size_t>(1, chunk_size);

    LOG(DEBUG) << "writing only " << chunk_size << " bytes";
    const auto data = utils::random::buffer(chunk_size);
    utils::temp::File file(data);

    Config config(utils::random::number(1, 20), chunk_size, block_size, false /* do not enforce minimum */);
    Streamer streamer(config);

    std::vector<char> dst(size);


    auto request_ret = streamer.async_read(file.path, 0, size, dst.data(), num_chunks, chunks.data());

    EXPECT_EQ(request_ret, common::ResponseCode::Success);

    // wait for all the requests to finish

    unsigned count_successful = 0;
    unsigned done_count = 0;
    for (unsigned i = 0; i < num_chunks; ++i)
    {
        const auto received = recv(streamer);
        LOG(SPAM) << "received response of request " << received.response.index << " : " << received.response.ret;
        if (received.response.ret == common::ResponseCode::Success)
        {
            ++count_successful;
        }
        else
        {
            EXPECT_EQ(received.response.ret, common::ResponseCode::EofError);
        }
        if (received.submission_done) ++done_count;
    }
    EXPECT_LT(count_successful, num_chunks);
    EXPECT_EQ(done_count, 1u);   // submission completes once its last sub-range lands
}

TEST(Async, Zero_Requests)
{
    auto size = utils::random::number(100, 1000);

    // create internal division
    const unsigned num_chunks = utils::random::number(1, 20);
    EXPECT_LT(num_chunks, size);

    auto chunks = utils::random::chunks(size, num_chunks);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 10), utils::random::number(1, 10), chunk_size, bulk_size, false /* do not enforce minimum */);


    Streamer streamer(config);

    std::vector<char> dst(size);

    // A submission with no ranges is accepted, not rejected: it simply reads nothing. Empties are
    // absorbed rather than refused, so there is no InvalidParameterError / EmptyRequestError here any
    // more. It is not registered either, so it owes no responses and there is nothing to receive.
    EXPECT_EQ(streamer.async_read(utils::random::string(), 0, size, dst.data(), 0, chunks.data()), common::ResponseCode::Success);
}

TEST(Async, Zero_Bytes_To_Read)
{
    auto size = utils::random::number(100, 1000);

    // create internal division
    const unsigned num_chunks = utils::random::number(1, 20);
    EXPECT_LT(num_chunks, size);

    auto chunks = utils::random::chunks(size, num_chunks);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);


    Streamer streamer(config);

    std::vector<char> dst(size);

    // Zero ranges is legal and reads nothing. (The path is random and does not exist, which does not
    // matter: with no ranges nothing ever reaches storage.)
    EXPECT_EQ(streamer.async_read(utils::random::string(), 0, 0, dst.data(), 0, chunks.data()), common::ResponseCode::Success);
}

// Replaces the second branch of the old Zero_Bytes_To_Read_Error, which asserted that a zero total with
// non-zero sub ranges was rejected. That contradiction is unrepresentable now - the ranges define the
// span, there is no separate total to disagree with - so this asserts what the new contract says
// instead: zero sized ranges are accepted AND answered, one response each.
TEST(Async, Zero_Sized_Ranges)
{
    const unsigned num_ranges = utils::random::number(1, 20);
    std::vector<size_t> zero_chunks(num_ranges, 0);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    // a real file: a zero sized transfer still opens its file, it just reads nothing from it
    const auto data = utils::random::buffer(utils::random::number(1, 100));
    utils::temp::File file(data);

    std::vector<char> dst(1);
    EXPECT_EQ(streamer.async_read(file.path, 0, 0, dst.data(), num_ranges, zero_chunks.data()),
              common::ResponseCode::Success);

    std::set<unsigned> received;
    for (unsigned i = 0; i < num_ranges; ++i)
    {
        bool done = false;
        const auto r = streamer.response(60000, done);
        EXPECT_EQ(r.ret, common::ResponseCode::Success);
        received.insert(r.index);
        EXPECT_EQ(done, i + 1 == num_ranges);   // completion lands on the last range
    }

    EXPECT_EQ(received, range_indices(num_ranges));
}

TEST(Async, ConcurrentRequests)
{
    // Multiple submissions are now accepted concurrently (no BusyError). Each is demuxed on the
    // shared persistent responder and both are delivered, each completing (submission_done) on its
    // single response.
    auto size = utils::random::number(100, 1000);
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    const auto expected = utils::Fd::read(file.path);
    EXPECT_EQ(expected.size(), size);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);


    Streamer streamer(config);

    // disjoint destination buffers, one per concurrent submission
    std::vector<unsigned char> dst1(size), dst2(size);
    std::vector<size_t> sizes;
    sizes.push_back(size);

    // both requests are accepted - the second does NOT return BusyError
    EXPECT_EQ(streamer.async_read(file.path, 0, size, dst1.data(), 1, sizes.data()), common::ResponseCode::Success);
    EXPECT_EQ(streamer.async_read(file.path, 0, size, dst2.data(), 1, sizes.data()), common::ResponseCode::Success);

    // both submissions are delivered; each is single-range, so each response is its submission's last
    std::set<unsigned> completed;
    for (int i = 0; i < 2; ++i)
    {
        const auto received = recv(streamer);
        EXPECT_EQ(received.response.ret, common::ResponseCode::Success);
        EXPECT_TRUE(received.submission_done);
        completed.insert(received.response.submission_id);
    }
    EXPECT_EQ(completed.size(), 2u);   // two distinct submissions, each ended

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(dst1[i], expected[i]);
        EXPECT_EQ(dst2[i], expected[i]);
        if (dst1[i] != expected[i] || dst2[i] != expected[i])
        {
            break;
        }
    }
}

// Previously this asserted the Assigner's "Input vector sizes mismatch" throw - it passed two paths but
// one entry in every other vector. Mismatched lengths are unrepresentable now (a request is a list of
// FileRanges), so it asserts the surviving property of a submission whose paths disagree: mixing two
// object-storage plugins is rejected up front. A different pair than MixedObjectPluginsRejected below.
TEST(AsyncRequest, InvalidScheme)
{
    const auto size = utils::random::number(100, 1000);
    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    std::vector<unsigned char> dst0(size);
    std::vector<unsigned char> dst1(size);

    std::vector<FileRanges> request;
    request.push_back(FileRanges{ "s3://s3-bucket/file-01.txt", { ReadRange{ 0, static_cast<size_t>(size), dst0.data() } } });
    request.push_back(FileRanges{ "az://az-account/file-02.txt", { ReadRange{ 0, static_cast<size_t>(size), dst1.data() } } });

    EXPECT_EQ(streamer.async_request(request), common::ResponseCode::UnsupportedBackendMix);
}

TEST(AsyncRequest, MixedObjectPluginsRejected)
{
    const auto size = utils::random::number(100, 1000);
    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    std::vector<unsigned char> dst0(size);
    std::vector<unsigned char> dst1(size);

    // a single submission that mixes two object-storage plugins (s3 + gcs) is rejected up front,
    // before any dispatch or plugin load
    std::vector<FileRanges> request;
    request.push_back(FileRanges{ "s3://bucket/a.txt", { ReadRange{ 0, static_cast<size_t>(size), dst0.data() } } });
    request.push_back(FileRanges{ "gs://bucket/b.txt", { ReadRange{ 0, static_cast<size_t>(size), dst1.data() } } });

    EXPECT_EQ(streamer.async_request(request), common::ResponseCode::UnsupportedBackendMix);
}

// A submission must pick ONE backend kind. The streamer serves both across submissions (see
// FilesystemAndObjectStorageSubmissionsCoexist), but within a submission the Assigner divides the work
// with a single backend's worker count and block size, and a workload has to be homogeneous to be routed.
// Rejecting up front replaces a slice-dependent outcome: without the check, this is InvalidParameterError
// when both kinds land in one workload and silently accepted with the wrong block size when they do not.
TEST(AsyncRequest, MixedFilesystemAndObjectStorageRejected)
{
    const auto size = utils::random::number(100, 1000);
    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    Streamer streamer(config);

    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    std::vector<unsigned char> dst0(size);
    std::vector<unsigned char> dst1(size);

    // both orders: the check must not depend on which kind is seen first (a first-file test would pass
    // one of these by accident)
    {
        std::vector<FileRanges> request;
        request.push_back(FileRanges{ file.path, { ReadRange{ 0, static_cast<size_t>(size), dst0.data() } } });
        request.push_back(FileRanges{ "s3://bucket/a.txt", { ReadRange{ 0, static_cast<size_t>(size), dst1.data() } } });

        EXPECT_EQ(streamer.async_request(request), common::ResponseCode::UnsupportedBackendMix);
    }
    {
        std::vector<FileRanges> request;
        request.push_back(FileRanges{ "s3://bucket/a.txt", { ReadRange{ 0, static_cast<size_t>(size), dst0.data() } } });
        request.push_back(FileRanges{ file.path, { ReadRange{ 0, static_cast<size_t>(size), dst1.data() } } });

        EXPECT_EQ(streamer.async_request(request), common::ResponseCode::UnsupportedBackendMix);
    }
}

// A file with no ranges reaches no storage (verify_requests accepts it deliberately, and it yields no
// transfer), so it must take no part in backend selection.
TEST(AsyncRequest, FilesWithoutRangesDoNotSelectTheBackend)
{
    const auto size = utils::random::number(100, 1000);
    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    const auto bulk_size = utils::random::number<size_t>(1, chunk_size);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size, bulk_size, false /* do not enforce minimum */);

    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);

    {
        // An empty object-storage entry must not lock the streamer's plugin: that submission reads
        // nothing, so a later submission using a DIFFERENT plugin is still legitimate. Both submissions
        // here are empty, so neither reaches a pool or loads a plugin - the lock alone is under test.
        Streamer streamer(config);

        std::vector<FileRanges> s3_only;
        s3_only.push_back(FileRanges{ "s3://bucket/empty.txt", {} });
        EXPECT_EQ(streamer.async_request(s3_only), common::ResponseCode::Success);

        std::vector<FileRanges> gcs_only;
        gcs_only.push_back(FileRanges{ "gs://bucket/empty.txt", {} });
        EXPECT_EQ(streamer.async_request(gcs_only), common::ResponseCode::Success);
    }

    {
        // A filesystem submission carrying an empty object-storage entry is NOT a mixed submission: the
        // empty entry contributes no batch, so every workload is still filesystem. It must also not
        // select the object-storage worker count and block size for the assignment.
        Streamer streamer(config);

        std::vector<unsigned char> dst(size);
        std::vector<FileRanges> request;
        request.push_back(FileRanges{ "s3://bucket/empty.txt", {} });
        request.push_back(FileRanges{ file.path, { ReadRange{ 0, static_cast<size_t>(size), dst.data() } } });

        EXPECT_EQ(streamer.async_request(request), common::ResponseCode::Success);

        // drain the single range and check it really read the filesystem file
        bool done = false;
        const auto response = streamer.response(60000, done);
        EXPECT_EQ(response.ret, common::ResponseCode::Success);
        EXPECT_EQ(dst, std::vector<unsigned char>(data.begin(), data.end()));
    }
}

namespace
{

std::set<std::string> paths_of(const std::vector<std::pair<std::string, size_t>> & entries)
{
    std::set<std::string> result;
    for (const auto & entry : entries)
    {
        result.insert(entry.first);
    }
    return result;
}

} // namespace

TEST(ListFiles, FilesystemBasicListingAndSizes)
{
    Streamer streamer;

    utils::temp::Dir dir;
    const auto data_a = utils::random::buffer(utils::random::number(1, 1000));
    const auto data_b = utils::random::buffer(utils::random::number(1, 1000));
    utils::temp::File a(dir.path, "a.bin", data_a);
    utils::temp::File b(dir.path, "b.bin", data_b);

    const auto entries = streamer.list_files(dir.path, true, {}, {});

    EXPECT_EQ(entries.size(), 2u);
    bool found_a = false;
    bool found_b = false;
    for (const auto & entry : entries)
    {
        if (entry.first == a.path) { EXPECT_EQ(entry.second, data_a.size()); found_a = true; }
        if (entry.first == b.path) { EXPECT_EQ(entry.second, data_b.size()); found_b = true; }
    }
    EXPECT_TRUE(found_a);
    EXPECT_TRUE(found_b);
}

TEST(ListFiles, FilesystemRecursive)
{
    Streamer streamer;

    utils::temp::Dir dir;
    utils::temp::File root_file(dir.path, "root.bin", utils::random::buffer(10));
    utils::temp::Dir sub(dir.path, "subdir");
    utils::temp::File nested(sub.path, "nested.bin", utils::random::buffer(10));

    const auto recursive = paths_of(streamer.list_files(dir.path, true, {}, {}));
    const auto non_recursive = paths_of(streamer.list_files(dir.path, false, {}, {}));

    EXPECT_TRUE(recursive.count(root_file.path));
    EXPECT_TRUE(recursive.count(nested.path));

    EXPECT_TRUE(non_recursive.count(root_file.path));
    EXPECT_FALSE(non_recursive.count(nested.path));
}

TEST(ListFiles, FilesystemAllowPattern)
{
    Streamer streamer;

    utils::temp::Dir dir;
    utils::temp::File st(dir.path, "model.safetensors", utils::random::buffer(10));
    utils::temp::File js(dir.path, "config.json", utils::random::buffer(10));

    const auto paths = paths_of(streamer.list_files(dir.path, true, {"*.safetensors"}, {}));

    EXPECT_TRUE(paths.count(st.path));
    EXPECT_FALSE(paths.count(js.path));
}

TEST(ListFiles, FilesystemIgnorePattern)
{
    Streamer streamer;

    utils::temp::Dir dir;
    utils::temp::File st(dir.path, "model.safetensors", utils::random::buffer(10));
    utils::temp::File js(dir.path, "config.json", utils::random::buffer(10));

    const auto paths = paths_of(streamer.list_files(dir.path, true, {}, {"*.json"}));

    EXPECT_TRUE(paths.count(st.path));
    EXPECT_FALSE(paths.count(js.path));
}

TEST(ListFiles, FilesystemNonExistentPathThrows)
{
    Streamer streamer;

    const std::string missing = "./" + utils::random::string() + "/" + utils::random::string();
    try
    {
        streamer.list_files(missing, true, {}, {});
        FAIL() << "expected an exception for a non-existent path";
    }
    catch (const common::Exception & e)
    {
        EXPECT_EQ(e.error(), common::ResponseCode::FileAccessError);
    }
}

TEST(ListFiles, FilesystemEmptyDirectory)
{
    Streamer streamer;

    utils::temp::Dir dir;

    const auto entries = streamer.list_files(dir.path, true, {}, {});
    EXPECT_TRUE(entries.empty());
}

TEST(Async, Scattered_Ranges_And_Destinations)
{
    // The point of the range API: ranges need not be contiguous in the file, need not be ordered, and
    // need not be written to adjacent memory. Every other data test here goes through async_read, which
    // tiles one span of the file into one buffer - so none of them would notice if offsets or
    // destinations were silently paired by position instead of being honoured per range.
    const size_t size = 1000;
    const auto data = utils::random::buffer(size);
    utils::temp::File file(data);
    const auto expected = utils::Fd::read(file.path);
    ASSERT_EQ(expected.size(), size);

    const auto chunk_size = utils::random::number<size_t>(1, 1024);
    Config config(utils::random::number(1, 20), utils::random::number(1, 20), chunk_size,
                  utils::random::number<size_t>(1, chunk_size), false /* do not enforce minimum */);
    Streamer streamer(config);

    // deliberately: descending file order, gaps between ranges, a zero-sized range, and two ranges
    // reading the SAME source bytes (source overlap is legal - only destinations must not overlap)
    std::vector<std::pair<size_t, size_t>> ranges =
    {
        { 700, 120 },
        {  50, 200 },
        { 400,   0 },
        { 700, 120 },
        { 900, 100 },
    };

    // Then a random number of random ranges on top: the five above are the shapes worth naming, but their
    // count is arbitrary, and a fixed count is one a position bug can fit by accident.
    const unsigned extra = utils::random::number(0, 15);
    for (unsigned i = 0; i < extra; ++i)
    {
        const size_t offset = utils::random::number<size_t>(0, size - 1);
        ranges.emplace_back(offset, utils::random::number<size_t>(0, size - offset));
    }

    // each destination is its OWN allocation, not an offset into a shared buffer: this is what proves a
    // destination need not belong to any single buffer. The extra byte is a guard against an over-long write.
    std::vector<std::vector<unsigned char>> dsts;
    for (const auto & range : ranges)
    {
        dsts.emplace_back(range.second + 1, 0xAB);
    }

    FileRanges file_ranges;
    file_ranges.path = file.path;
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        file_ranges.ranges.push_back(ReadRange{ ranges[i].first, ranges[i].second, dsts[i].data() });
    }

    std::vector<FileRanges> request{ file_ranges };
    SubmissionId submission_id = 0;
    EXPECT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);

    std::set<unsigned> received;
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        const auto r = recv(streamer);
        EXPECT_EQ(r.response.ret, common::ResponseCode::Success);
        EXPECT_EQ(r.response.file_index, 0u);
        received.insert(r.response.index);
        EXPECT_EQ(r.submission_done, i + 1 == ranges.size());
    }

    // exactly one response per range, indexed within the file - a dropped or duplicated zero-sized
    // range would shift every later index
    EXPECT_EQ(received, range_indices(ranges.size()));

    for (size_t i = 0; i < ranges.size(); ++i)
    {
        const auto offset = ranges[i].first;
        const auto length = ranges[i].second;
        for (size_t j = 0; j < length; ++j)
        {
            ASSERT_EQ(dsts[i][j], expected[offset + j])
                << "range " << i << " (offset " << offset << " size " << length << ") differs at byte " << j;
        }
        EXPECT_EQ(dsts[i][length], 0xAB) << "range " << i << " wrote past the end of its destination";
    }
}

// libaio has no asynchronous buffered mode, so a file it cannot read directly has to reach the
// synchronous reader BEFORE dispatch - once a workload is on the async pool the worker cannot hand it
// back. Congruence is the first of the two reasons.
//
// The destination here is deliberately one byte out of step with the file offset. Aligning the buffer
// is not enough: a direct read puts file byte F+k at address B+k, so it needs (B - F) % block == 0.
// Without that NO part of the region can be read directly.
TEST(Async, LibaioSkipsAFileItCannotReadDirectly)
{
    const auto data = utils::random::buffer(8192);
    utils::temp::File file(data);

    // Over-allocated so the destination can be moved off a block boundary on purpose.
    std::vector<char> dst(data.size() + posix_io::DirectBlockSize);
    char * misaligned = dst.data() + 1;
    ASSERT_NE((reinterpret_cast<uintptr_t>(misaligned)) % posix_io::DirectBlockSize, 0u);

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), misaligned });

    Streamer::Environment environment;
    environment.availability = [](posix_io::Strategy) { return common::ResponseCode::Success; };
    environment.direct = [](dev_t, const std::string &) { return posix_io::DirectSupport::Yes; };

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("libaio_direct"));
    Streamer streamer(Config(), std::move(environment));

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    EXPECT_FALSE(streamer.async_pool_used())
        << "a file libaio cannot read directly must not reach the async pool at all";

    SubmissionStats stats;
    ASSERT_TRUE(streamer.stats().find(submission_id, stats));
    ASSERT_EQ(stats.files.size(), 1u);
    EXPECT_EQ(stats.files[0].strategy, posix_io::Strategy::SyncBuffered);

    // And the bytes are still right - routing away is a performance decision, never a correctness one.
    EXPECT_EQ(std::vector<uint8_t>(misaligned, misaligned + data.size()), data);
}

// The second reason: the mount cannot serve O_DIRECT at all. Congruence holds here, so this isolates
// the mount from the placement.
TEST(Async, LibaioSkipsAMountWithoutODirect)
{
    const auto data = utils::random::buffer(8192);
    utils::temp::File file(data);

    std::vector<char> dst(data.size() + posix_io::DirectBlockSize);
    char * congruent = dst.data() + ((-reinterpret_cast<uintptr_t>(dst.data()))
                                     % posix_io::DirectBlockSize);
    ASSERT_TRUE(posix_io::is_congruent(0, congruent, posix_io::DirectBlockSize));

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), congruent });

    Streamer::Environment environment;
    environment.availability = [](posix_io::Strategy) { return common::ResponseCode::Success; };
    environment.direct = [](dev_t, const std::string &) { return posix_io::DirectSupport::No; };

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("libaio_direct"));
    Streamer streamer(Config(), std::move(environment));

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    EXPECT_FALSE(streamer.async_pool_used())
        << "libaio without O_DIRECT reads one file at a time, which is worse than the 16-thread pool";

    EXPECT_EQ(std::vector<uint8_t>(congruent, congruent + data.size()), data);
}

// A congruent file on a mount that serves O_DIRECT is the case libaio exists for, so it must reach
// the async pool. Without this the two tests above would pass with routing that always said no.
TEST(Async, LibaioTakesAFileItCanReadDirectly)
{
    const auto data = utils::random::buffer(8192);
    utils::temp::File file(data);

    std::vector<char> dst(data.size() + posix_io::DirectBlockSize);
    char * congruent = dst.data() + ((-reinterpret_cast<uintptr_t>(dst.data()))
                                     % posix_io::DirectBlockSize);

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), congruent });

    Streamer::Environment environment;
    environment.availability = [](posix_io::Strategy) { return common::ResponseCode::Success; };
    environment.direct = [](dev_t, const std::string &) { return posix_io::DirectSupport::Yes; };

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("libaio_direct"));
    Streamer streamer(Config(), std::move(environment));

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    EXPECT_TRUE(streamer.async_pool_used());

    SubmissionStats stats;
    ASSERT_TRUE(streamer.stats().find(submission_id, stats));
    ASSERT_EQ(stats.files.size(), 1u);
    EXPECT_EQ(stats.files[0].strategy, posix_io::Strategy::LibaioDirect);

    EXPECT_EQ(std::vector<uint8_t>(congruent, congruent + data.size()), data);
}

// io_uring keeps a file it cannot read directly, because a buffered read on the ring is STILL
// asynchronous - the worker just opens that one file without O_DIRECT. Only libaio has to route away.
//
// Without this test the routing could reject non-congruent files for every strategy and nothing would
// notice, which would quietly send io_uring work to the 16-thread pool.
TEST(Async, IoUringKeepsAFileItCannotReadDirectly)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring is unavailable here";
    }

    const auto data = utils::random::buffer(8192);
    utils::temp::File file(data);

    std::vector<char> dst(data.size() + posix_io::DirectBlockSize);
    char * misaligned = dst.data() + 1;

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), misaligned });

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_direct"));
    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    EXPECT_TRUE(streamer.async_pool_used())
        << "io_uring falls back to a buffered read on the same ring, so the file stays on the engine";

    EXPECT_EQ(std::vector<uint8_t>(misaligned, misaligned + data.size()), data);
}

// THE ASSERTION S8b EXISTS FOR: reading into destinations the streamer placed congruently must copy
// NOTHING.
//
// A direct read puts file byte F+k at address B+k, so if (B - F) is a multiple of the block, every
// block boundary in the file lines up with one in memory and no edge needs a scratch buffer. When it
// does not line up, every byte is copied through scratch instead - which costs throughput and reports
// no error anywhere. A number, not a log line, is the only way to see it.
//
// This could not be written before. The worker tests drive chunks directly, so they check the
// bouncing logic rather than the placement; only here do a real request, real routing and a real
// engine meet.
TEST(Async, CongruentDestinationsBounceNothing)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring is unavailable here";
    }

    const size_t block = posix_io::DirectBlockSize;
    const auto data = utils::random::buffer(block * 16);
    utils::temp::File file(data);

    // Congruent with file offset 0: the destination itself lands on a block boundary.
    std::vector<char> dst(data.size() + block);
    char * congruent = dst.data() + ((-reinterpret_cast<uintptr_t>(dst.data())) % block);
    ASSERT_TRUE(posix_io::is_congruent(0, congruent, block));

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), congruent });

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_direct"));
    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    const auto counters = streamer.async_counters();
    EXPECT_EQ(counters.bytes_read, data.size());
    EXPECT_EQ(counters.bounced_bytes, 0u)
        << "a congruent destination has no partial edge, so nothing should go through scratch";

    EXPECT_EQ(std::vector<uint8_t>(congruent, congruent + data.size()), data);
}

// The other side of the same assertion, so a zero above cannot come from bouncing being broken or
// never reached. A destination one byte out of step has no aligned part at all, so the worker reads
// the file BUFFERED - and a buffered read never bounces either.
//
// So what this pins is the routing, not the copying: the bytes still arrive, and they arrive without
// scratch, because the worker chose the reader that does not need it.
TEST(Async, ANonCongruentDestinationIsReadBufferedRatherThanBounced)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring is unavailable here";
    }

    const size_t block = posix_io::DirectBlockSize;
    const auto data = utils::random::buffer(block * 4);
    utils::temp::File file(data);

    std::vector<char> dst(data.size() + block);
    char * misaligned = dst.data() + 1;
    ASSERT_FALSE(posix_io::is_congruent(0, misaligned, block));

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), misaligned });

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_direct"));
    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    const auto counters = streamer.async_counters();
    EXPECT_EQ(counters.bytes_read, data.size());
    EXPECT_EQ(counters.bounced_bytes, 0u)
        << "no part of this region can be read directly, so the worker reads it buffered instead of"
           " copying every byte through scratch";

    EXPECT_EQ(std::vector<uint8_t>(misaligned, misaligned + data.size()), data);
}

// The counters are summed over every worker and reach zero when nothing async ran, so a caller cannot
// mistake "no async work" for "async work that copied nothing".
TEST(Async, CountersAreZeroWithoutAnAsyncWorkload)
{
    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("sync_buffered"));
    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    const auto counters = streamer.async_counters();
    EXPECT_EQ(counters.bytes_read, 0u) << "the synchronous reader served this, so no worker counted it";
    EXPECT_EQ(counters.bounced_bytes, 0u);
    EXPECT_EQ(counters.achieved_depth, 0u);
}

// Achieved depth is a high-water mark, so it must survive the reads finishing - by the time anyone
// asks, the live in-flight count is back to zero.
TEST(Async, AchievedDepthOutlivesTheReads)
{
    if (!ring_works())
    {
        GTEST_SKIP() << "io_uring is unavailable here";
    }

    const auto data = utils::random::buffer(4 * 1024 * 1024);
    utils::temp::File file(data);

    std::vector<char> dst(data.size());
    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered"));
    Streamer streamer;

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
    EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);

    const auto counters = streamer.async_counters();
    EXPECT_GE(counters.achieved_depth, 1u) << "at least one read was outstanding at some point";
    EXPECT_EQ(counters.bytes_read, data.size());
}

// direct_block_for answers from the INJECTED probe when a test set one.
//
// Every measurement site has to honour the same seam. This one did not, so a test could answer for
// the router's groups() and reads_directly and still get the build machine's real block back from
// this API -
// a number that changes with the filesystem the tests happen to run on.
//
// The two mounts answer differently on purpose. A request spanning both must report the LARGER, since
// congruence at a power of two implies congruence at every smaller one, so one number satisfies both.
TEST(Streamer, Direct_Block_For_Uses_The_Injected_Probe)
{
    const auto data = utils::random::buffer(100);

    // Two directories, because the probe answers per directory - two files in one would share a mount
    // and the larger-of-the-two check would prove nothing.
    utils::temp::Dir dir_one;
    utils::temp::Dir dir_two;
    utils::temp::File one(dir_one.path, utils::random::string(), data);
    utils::temp::File two(dir_two.path, utils::random::string(), data);

    const std::string first = dir_one.path;
    const dev_t device_one = makedev(8, 1);
    const dev_t device_two = makedev(8, 2);

    // Values no real mount would report, so a number that leaked in from the machine is obvious.
    constexpr size_t BlockOne = 8192;
    constexpr size_t BlockTwo = 32768;

    Streamer streamer(Config(), Streamer::Environment{
                      .mount = [first, device_one, device_two](const std::string & directory) -> posix_io::MountCapability
                      {
                          return posix_io::MountCapability{ directory == first ? device_one : device_two, false };
                      },
                      .direct = {},
                      .direct_block = [device_one, BlockOne, BlockTwo](dev_t device, const std::string &) -> size_t
                      {
                          return device == device_one ? BlockOne : BlockTwo;
                      },
    });

    size_t block = 0;
    EXPECT_EQ(streamer.direct_block_for({ one.path }, block), common::ResponseCode::Success);
    EXPECT_EQ(block, BlockOne);

    EXPECT_EQ(streamer.direct_block_for({ two.path }, block), common::ResponseCode::Success);
    EXPECT_EQ(block, BlockTwo);

    EXPECT_EQ(streamer.direct_block_for({ one.path, two.path }, block), common::ResponseCode::Success);
    EXPECT_EQ(block, BlockTwo) << "a request spanning both mounts must be laid out at the larger";
}

// A mount whose engine dies is read by the SYNCHRONOUS reader from then on, and the bytes are right.
//
// This is the end of the chain the pieces below only cover separately: the worker marks its engine
// dead, tells the streamer which mount it served, and the router's groups() stops routing it there.
// Each link was where the bugs in this area lived, so the test drives all three.
//
// The engine is injected because a real one fails only when its ring or context is gone, which a test
// cannot arrange - and what follows the failure is the part worth pinning.
TEST(Async, ADeadEngineDropsItsMountToTheSynchronousReader)
{
    const auto data = utils::random::buffer(8192);
    utils::temp::File first(data);
    utils::temp::File second(data);

    Streamer::Environment environment;
    environment.availability = [](posix_io::Strategy) { return common::ResponseCode::Success; };
    environment.direct = [](dev_t, const std::string &) { return posix_io::DirectSupport::No; };

    // Every engine this streamer builds refuses to issue anything - the ring is gone before the first
    // read goes out.
    environment.engine = [](posix_io::Strategy, const posix_io::AsyncIoConfig & config)
        -> std::unique_ptr<posix_io::IoEngine>
    {
        posix_io::Limits limits;
        limits.max_read_bytesize = posix_io::max_read_bytesize();
        limits.offset_alignment = posix_io::DirectBlockSize;
        limits.buffer_alignment = posix_io::DirectBlockSize;

        auto engine = std::make_unique<posix_io::MockIoEngine>(config.depth, limits);
        engine->set_flush_result(common::ResponseCode::FsAsyncEngineError);
        return engine;
    };

    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered"));
    Streamer streamer(Config(), std::move(environment));

    // First submission: it reaches the dead engine and fails with the engine's own code - not
    // UnknownError, which would tell the caller to abort everything over one broken ring.
    {
        std::vector<char> dst(data.size());
        std::vector<FileRanges> request(1);
        request[0].path = first.path;
        request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

        SubmissionId submission_id = 0;
        ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
        EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::FsAsyncEngineError);
    }

    // Second submission on the same mount: no longer routed to the async pool at all, and it reads.
    {
        std::vector<char> dst(data.size());
        std::vector<FileRanges> request(1);
        request[0].path = second.path;
        request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

        SubmissionId submission_id = 0;
        ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
        EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success)
            << "the storage is healthy - only the ring was lost, so this must still read";

        SubmissionStats stats;
        ASSERT_TRUE(streamer.stats().find(submission_id, stats));
        ASSERT_EQ(stats.files.size(), 1u);
        EXPECT_EQ(stats.files[0].strategy, posix_io::Strategy::SyncBuffered)
            << "the mount must have dropped to the synchronous reader";

        EXPECT_EQ(std::vector<uint8_t>(dst.begin(), dst.end()), data)
            << "a demotion is a performance decision, never a correctness one";
    }
}

// Two mounts of different types get the depth their type asks for, from one setting.
//
// The engine factory delegates to the real one and records the depth it was asked for - the only place
// the resolution is observable, since everything downstream is the ring's own size. A mock engine
// cannot be used here: it never completes, so the read would never return.
//
// Which means a REAL ring is built: the injected availability makes the resolver pick io_uring
// whatever the host says, so without one the factory returns nullptr and this fails. Skipping is
// silent, so RUNAI_STREAMER_REQUIRE_IO_URING - which CI passes - turns the skip into a failure rather
// than hiding a broken CI host.
TEST(Async, QueueDepthIsResolvedPerMount)
{
    const auto ring = posix_io::IoUringProbe::instance().capability();
    if (!ring.available)
    {
        const char * const required = std::getenv("RUNAI_STREAMER_REQUIRE_IO_URING");
        if (required != nullptr && std::string(required) == "1")
        {
            FAIL() << "io_uring is unavailable (" << ring.error << ") but "
                   << "RUNAI_STREAMER_REQUIRE_IO_URING=1 says this host has it";
        }
        GTEST_SKIP() << "io_uring unavailable (" << ring.error << "); this test builds a real ring";
    }

    const auto data = utils::random::buffer(8192);
    utils::temp::Dir dir_one;
    utils::temp::Dir dir_two;
    utils::temp::File nfs_file(dir_one.path, utils::random::string(), data);
    utils::temp::File local_file(dir_two.path, utils::random::string(), data);

    const std::string nfs_dir = dir_one.path;

    utils::temp::Env depth(std::string("RUNAI_STREAMER_FS_QUEUE_DEPTH"), std::string("512,nfs=64"));
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 1UL);
    utils::temp::Env strategy(std::string("RUNAI_STREAMER_FS_STRATEGY"), std::string("io_uring_buffered"));

    // FS_MAX_ENGINES is deliberately NOT set. Engines are grouped by depth and the limit applies
    // inside each group, so the default of one already gives 64 and 512 an engine each.
    utils::temp::UnsetEnv engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"));

    std::mutex recorded_mutex;
    std::vector<unsigned> recorded;

    Streamer::Environment environment;
    environment.availability = [](posix_io::Strategy) { return common::ResponseCode::Success; };
    environment.mount = [nfs_dir](const std::string & directory) -> posix_io::MountCapability
    {
        const bool is_nfs = directory == nfs_dir;
        return posix_io::MountCapability{ is_nfs ? makedev(8, 1) : makedev(8, 2), false,
                                          is_nfs ? "nfs4" : "ext4" };
    };
    environment.engine = [&recorded, &recorded_mutex](posix_io::Strategy s, const posix_io::AsyncIoConfig & config)
    {
        {
            const auto guard = std::unique_lock<std::mutex>(recorded_mutex);
            recorded.push_back(config.depth);
        }
        return posix_io::make_io_engine(s, config);
    };

    Streamer streamer(Config(), std::move(environment));

    for (const auto & path : { nfs_file.path, local_file.path })
    {
        std::vector<char> dst(data.size());
        std::vector<FileRanges> request(1);
        request[0].path = path;
        request[0].ranges.push_back(ReadRange{ 0, data.size(), dst.data() });

        SubmissionId submission_id = 0;
        ASSERT_EQ(streamer.async_request(request, &submission_id), common::ResponseCode::Success);
        EXPECT_EQ(recv(streamer).response.ret, common::ResponseCode::Success);
        EXPECT_EQ(std::vector<uint8_t>(dst.begin(), dst.end()), data);
    }

    const auto guard = std::unique_lock<std::mutex>(recorded_mutex);
    ASSERT_EQ(recorded.size(), 2u) << "one engine per mount, so two engines";

    // In order, not as a set: the submissions are drained one at a time, so the nfs4 engine is always
    // built first. A set would also pass with the two depths swapped, which is the very thing this
    // asserts.
    EXPECT_EQ(recorded, (std::vector<unsigned>{ 64, 512 }))
        << "the nfs4 mount is submitted first and takes the nfs entry; the ext4 mount takes the default";
}

}; // namespace runai::llm::streamer::impl
