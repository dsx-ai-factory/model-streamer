/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/pools/backend_pools.h"

#include <gtest/gtest.h>

#include <sys/sysmacros.h>   // makedev

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "common/response_code/response_code.h"
#include "streamer/impl/config/config/config.h"
#include "utils/temp/env/env.h"

namespace runai::llm::streamer::impl
{

using Pool = BackendPools::Pool;

namespace
{
// Empty workloads (size() == 0) are a no-op in execute(), so this filesystem worker just exercises lazy
// pool creation/reuse without needing the full batch/reader machinery.
struct RunWorker : utils::Worker<Workload>
{
    void execute(Workload && workload, std::atomic<bool> & stopped) override { workload.execute(stopped); }
    void drain(std::atomic<bool> &) override {}
    bool idle() const override { return true; }
};

std::unique_ptr<utils::Worker<Workload>> run()
{
    return std::make_unique<RunWorker>();
}

// The object-storage pool is a per-worker pool; these tests exercise pool lifecycle, not worker behavior,
// so its factory builds a trivial no-op worker (a real ObjectStorageWorker would build a plugin client from
// the first workload's params). This lets the tests push an empty Workload{} harmlessly.
struct NoopWorker : utils::Worker<Workload>
{
    void execute(Workload &&, std::atomic<bool> &) override {}
    void drain(std::atomic<bool> &) override {}
    bool idle() const override { return true; }
};

// Two factories, because the two are no longer the same type: the filesystem async factory takes the
// mount's measured block, the object-storage one takes nothing.
std::unique_ptr<utils::Worker<Workload>> noop_async_factory(dev_t /* device */, size_t /* block */,
                                                           unsigned /* depth */, bool /* register */)
{
    return std::make_unique<NoopWorker>();
}

std::unique_ptr<utils::Worker<Workload>> noop_factory()
{
    return std::make_unique<NoopWorker>();
}
} // namespace

// The default is ONE engine per queue depth: the throughput case for splitting is unmeasured, and
// io_uring itself is off by default for the same reason. So a streamer that reads three mounts at one
// depth still builds one engine unless someone raises the variable.
TEST(BackendPools, DefaultsToOneEngineForAllMounts)
{
    utils::temp::UnsetEnv max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"));

    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(8, 2), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(259, 0), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});

    EXPECT_EQ(pools.async_engines(), 1u) << "the default cap is 1";
}

// The case the per-depth cap exists for: RUNAI_STREAMER_FS_QUEUE_DEPTH="512,nfs=64" with the default
// cap of 1, on a submission that meets two ext4 mounts before the NFS one.
//
// With a single process-wide cap the two ext4 mounts consume every engine and NFS shares one built at
// 512 - the configured 64 never applies, and only the discovery order decides that. Per depth, NFS has
// its own bucket, so it always gets an engine at 64 and the ext4 mounts share with each other, which
// costs them nothing they configured.
TEST(BackendPools, AConfiguredDepthSurvivesTheDiscoveryOrder)
{
    utils::temp::UnsetEnv max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"));

    std::vector<unsigned> built;
    auto recording = [&built](dev_t, size_t, unsigned depth, bool) -> std::unique_ptr<utils::Worker<Workload>>
    {
        built.push_back(depth);
        return std::make_unique<NoopWorker>();
    };

    BackendPools pools(run, recording, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* ext4 */, true /* register buffers */, Workload{});
    pools.push_async(makedev(8, 2), 0 /* block: not probed in this test */, 512 /* ext4 */, true /* register buffers */, Workload{});
    pools.push_async(makedev(0, 42), 0 /* block: not probed in this test */, 64 /* nfs */, true /* register buffers */, Workload{});

    EXPECT_EQ(pools.async_engines(), 2u) << "one engine per depth, not one per process";
    EXPECT_EQ(pools.shared_engine_mounts(), 1u) << "the second ext4 mount shares, the NFS one does not";

    EXPECT_EQ(built, (std::vector<unsigned>{ 512u, 64u }))
        << "an engine must be built at the depth NFS resolved, and only one at 512";
}

// Mounts that disagree about registered buffers must not share an engine, even at the default cap of
// one. Registration is bound into an engine when it is built, so a shared engine would let whichever
// mount arrived first decide for the other - and it pays on some file systems and costs on others, so
// either way round is wrong for somebody.
//
// Reachable with the DEFAULT configuration: a plain RUNAI_STREAMER_FS_QUEUE_DEPTH gives every mount the
// same depth, which before this was the only thing keeping them apart.
TEST(BackendPools, MountsThatDisagreeAboutRegistrationDoNotShareAnEngine)
{
    utils::temp::UnsetEnv max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"));

    std::vector<bool> built;
    auto recording = [&built](dev_t, size_t, unsigned, bool registers) -> std::unique_ptr<utils::Worker<Workload>>
    {
        built.push_back(registers);
        return std::make_unique<NoopWorker>();
    };

    BackendPools pools(run, recording, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0, 512 /* same depth */, true /* ext4 */, Workload{});
    pools.push_async(makedev(0, 42), 0, 512 /* same depth */, false /* nfs */, Workload{});

    EXPECT_EQ(pools.async_engines(), 2u) << "one engine each, though the depth is the same";
    EXPECT_EQ(built, (std::vector<bool>{ true, false })) << "each engine kept its own mount's answer";
}

// The engine count alone would still pass if the workloads went to the wrong engine, so this follows
// the work itself: each engine counts what it ran, and the NFS workload must land on the 64 one.
TEST(BackendPools, WorkloadsRunOnTheEngineOfTheirDepth)
{
    utils::temp::UnsetEnv max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"));

    struct CountingWorker : utils::Worker<Workload>
    {
        explicit CountingWorker(std::atomic<unsigned> & ran) : _ran(ran) {}
        void execute(Workload &&, std::atomic<bool> &) override { ++_ran; }
        void drain(std::atomic<bool> &) override {}
        bool idle() const override { return true; }
        std::atomic<unsigned> & _ran;
    };

    std::atomic<unsigned> ran_at_512{0};
    std::atomic<unsigned> ran_at_64{0};

    auto counting = [&](dev_t, size_t, unsigned depth, bool) -> std::unique_ptr<utils::Worker<Workload>>
    {
        return std::make_unique<CountingWorker>(depth == 64 ? ran_at_64 : ran_at_512);
    };

    BackendPools pools(run, counting, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* ext4 */, true /* register buffers */, Workload{});
    pools.push_async(makedev(8, 2), 0 /* block: not probed in this test */, 512 /* ext4 */, true /* register buffers */, Workload{});
    pools.push_async(makedev(0, 42), 0 /* block: not probed in this test */, 64 /* nfs */, true /* register buffers */, Workload{});

    for (int i = 0; i < 500 && (ran_at_512.load() + ran_at_64.load()) < 3; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(ran_at_64.load(), 1u) << "the NFS workload must run on the engine built at its depth";
    EXPECT_EQ(ran_at_512.load(), 2u);
}

// The cap bounds each depth separately, so two depths at a cap of 2 may reach four engines. That is
// the cost of the guarantee: the worst case is the cap times the number of distinct depths.
TEST(BackendPools, TheCapIsPerDepth)
{
    utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 2UL);

    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(8, 2), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(8, 3), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 2u) << "the third mount at 512 shares";

    pools.push_async(makedev(0, 42), 0 /* block: not probed in this test */, 64 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(0, 43), 0 /* block: not probed in this test */, 64 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(0, 44), 0 /* block: not probed in this test */, 64 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 4u) << "the 64 bucket has its own two, and its third mount shares";

    EXPECT_EQ(pools.shared_engine_mounts(), 2u);
}

// Raised, each mount gets its own engine - which is the isolation the split exists for.
TEST(BackendPools, EnginePerMountUpToTheCap)
{
    utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 4UL);

    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 1u);

    pools.push_async(makedev(8, 2), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 2u);

    pools.push_async(makedev(259, 0), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 3u);
}

// A cap too large for `unsigned` must not wrap to zero. A zero cap is not a small one: no engine is
// ever built, so every mount takes the sharing branch and least_loaded_async asserts on an empty
// bucket - fatal in every build.
//
// 2^32 exactly, because that is the smallest value that truncates to zero rather than to something
// merely wrong.
TEST(BackendPools, AnOversizedEngineCapDoesNotWrapToZero)
{
    utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 4294967296UL);

    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 1u) << "an engine must still be created";
}

// An engine costs a thread and a ring, so the limit takes the same ceiling as the concurrencies -
// and it applies per depth, so an unbounded value would be multiplied by the depths configured.
// Capped through Config::to_concurrency, which is also what warns; env-vars.md promises that warning.
TEST(BackendPools, TheEngineCapIsBounded)
{
    {
        utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 100000UL);
        BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);
        EXPECT_EQ(pools.max_async_engines(), Config::max_concurrency);
    }

    {
        utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 4UL);
        BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);
        EXPECT_EQ(pools.max_async_engines(), 4u) << "a value below the ceiling is untouched";
    }
}

// The same mount keeps the same engine however often it is pushed to - assignment is stable, because
// completion routing and window credit live with the engine.
TEST(BackendPools, SameMountReusesItsEngine)
{
    utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 4UL);

    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    for (int i = 0; i < 5; ++i)
    {
        pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    }

    EXPECT_EQ(pools.async_engines(), 1u) << "one mount must never build a second engine";
}

// Past the cap, mounts SHARE rather than queueing for a free engine - queueing would put a second
// head-of-line problem at the assignment layer.
TEST(BackendPools, PastTheCapMountsShare)
{
    utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 2UL);

    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(8, 2), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 2u);

    // The third and fourth mounts must not create engines, and must not be refused either.
    pools.push_async(makedev(8, 3), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    pools.push_async(makedev(8, 4), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 2u) << "the cap must bound engines, not reject work";
}

// Engines are lazy like the synchronous pool: a streamer that never reads a mount never builds a ring
// or a thread for it.
TEST(BackendPools, AsyncEnginesAreLazy)
{
    utils::temp::Env max_engines(std::string("RUNAI_STREAMER_FS_MAX_ENGINES"), 4UL);

    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    EXPECT_EQ(pools.async_engines(), 0u);
    EXPECT_EQ(pools.pools_created(), 0u);

    pools.push_async(makedev(8, 1), 0 /* block: not probed in this test */, 512 /* depth */, true /* register buffers */, Workload{});
    EXPECT_EQ(pools.async_engines(), 1u);
    EXPECT_EQ(pools.pools_created(), 1u) << "and it counts among the pools";
}

TEST(BackendPools, FilesystemPoolCreatedLazilyOnPush)
{
    BackendPools pools(run, noop_async_factory, noop_factory, /*filesystem_size=*/2, /*object_storage_size=*/3);

    EXPECT_EQ(pools.pools_created(), 0u);

    pools.push(Pool::FileSystem, Workload{});
    EXPECT_EQ(pools.pools_created(), 1u);

    // reusing the filesystem pool does not create another
    pools.push(Pool::FileSystem, Workload{});
    EXPECT_EQ(pools.pools_created(), 1u);
}

TEST(BackendPools, ObjectStoragePoolCreatedByPluginLock)
{
    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    EXPECT_EQ(pools.pools_created(), 0u);

    // locking the plugin builds the object-storage pool
    EXPECT_EQ(pools.lock_object_plugin(BackendPools::Plugin::S3), common::ResponseCode::Success);
    EXPECT_EQ(pools.pools_created(), 1u);

    // a repeated lock of the same plugin does not create another, and the pool now accepts workloads
    EXPECT_EQ(pools.lock_object_plugin(BackendPools::Plugin::S3), common::ResponseCode::Success);
    pools.push(Pool::ObjectStorage, Workload{});
    EXPECT_EQ(pools.pools_created(), 1u);
}

TEST(BackendPools, BothKindsCreateTwoPools)
{
    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    pools.push(Pool::FileSystem, Workload{});
    EXPECT_EQ(pools.lock_object_plugin(BackendPools::Plugin::Azure), common::ResponseCode::Success);
    EXPECT_EQ(pools.pools_created(), 2u);
}

TEST(BackendPools, ObjectPluginLockedToOne)
{
    BackendPools pools(run, noop_async_factory, noop_factory, 2, 3);

    // first object-storage plugin wins; the same plugin is accepted; a different one is rejected
    EXPECT_EQ(pools.lock_object_plugin(BackendPools::Plugin::GCS), common::ResponseCode::Success);
    EXPECT_EQ(pools.lock_object_plugin(BackendPools::Plugin::GCS), common::ResponseCode::Success);
    EXPECT_EQ(pools.lock_object_plugin(BackendPools::Plugin::S3), common::ResponseCode::UnsupportedBackendMix);

    // the rejected plugin did not create a second pool
    EXPECT_EQ(pools.pools_created(), 1u);
}


// Every plugin builds one worker per unit of concurrency, each owning its own client.
TEST(BackendPools, ObjectStoragePoolRunsAWorkerPerUnit)
{
    for (const auto plugin : { BackendPools::Plugin::S3, BackendPools::Plugin::GCS, BackendPools::Plugin::Azure })
    {
        std::atomic<unsigned> built{0};
        auto counting = [&built]() -> std::unique_ptr<utils::Worker<Workload>>
        {
            ++built;
            return std::make_unique<NoopWorker>();
        };

        BackendPools pools(run, noop_async_factory, counting, 2, /*object_storage_size=*/3);
        ASSERT_EQ(pools.lock_object_plugin(plugin), common::ResponseCode::Success);

        EXPECT_EQ(built.load(), 3u);
    }
}

}; // namespace runai::llm::streamer::impl
