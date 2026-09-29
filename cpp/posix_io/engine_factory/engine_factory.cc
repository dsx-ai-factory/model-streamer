/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/engine_factory/engine_factory.h"

#include <utility>

#include "common/exception/exception.h"
#include "posix_io/io_uring_engine/io_uring_engine.h"
#include "posix_io/io_uring_probe/io_uring_probe.h"
#include "posix_io/libaio_engine/libaio_engine.h"
#include "posix_io/libaio_probe/libaio_probe.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

std::unique_ptr<IoEngine> make_io_uring_engine(Strategy strategy, const AsyncIoConfig & config)
{
    auto & probe = IoUringProbe::instance();

    // Consulted rather than re-derived. Both this and strategy resolution ask "can this host do
    // io_uring", and two mechanisms answering independently can disagree - resolution picking a
    // strategy this then refuses to build. One source of truth, cached, so the several engines built
    // under one-engine-per-mount cost one probe between them.
    const auto capability = probe.capability();
    if (!capability.available)
    {
        LOG(WARNING) << "Cannot build " << strategy << ": io_uring is unavailable here ("
                     << capability.error << ")";
        return nullptr;
    }

    try
    {
        return std::make_unique<IoUringEngine>(config);
    }
    catch (const common::Exception & e)
    {
        // The probe said io_uring works, and it does - a ring of THIS depth is what failed, on ENOMEM
        // or RLIMIT_MEMLOCK. Retrying cannot help, and leaving the strategy in the chain would have
        // resolution keep choosing it. Demote once, permanently.
        LOG(ERROR) << "io_uring is available but a ring of depth " << config.depth
                   << " could not be built: " << e.error();
        probe.mark_unavailable(e.error());
        return nullptr;
    }
}


std::unique_ptr<IoEngine> make_libaio_engine(const AsyncIoConfig & config)
{
    auto & probe = LibaioProbe::instance();

    // Consulted rather than re-derived, exactly as the io_uring builder does. Resolution and this
    // both ask "can this host do libaio", and two mechanisms answering independently can disagree -
    // resolution picking a strategy this then refuses to build.
    const auto capability = probe.capability();
    if (!capability.available)
    {
        LOG(WARNING) << "Cannot build " << Strategy::LibaioDirect << ": libaio is unavailable here ("
                     << capability.error << ")";
        return nullptr;
    }

    try
    {
        return std::make_unique<LibaioEngine>(config);
    }
    catch (const common::Exception & e)
    {
        // A one-event context worked, so aio is here - a context of THIS depth is what failed, after
        // halving all the way down. The node's aio budget does not grow on its own, so retrying
        // cannot help and leaving the strategy in the chain would have resolution keep choosing it.
        LOG(ERROR) << "libaio is available but a context for depth " << config.depth
                   << " could not be built: " << e.error();
        probe.mark_unavailable(e.error());
        return nullptr;
    }
}

} // namespace

std::unique_ptr<IoEngine> make_io_engine(Strategy strategy, const AsyncIoConfig & config)
{
    ASSERT(is_async(strategy)) << "Cannot build an engine for " << strategy
                               << " - it is served by the synchronous pool, not by an IoEngine";

    switch (strategy)
    {
    case Strategy::IoUringDirect:
    case Strategy::IoUringBuffered:
        // Both strategies use the same builder, because both use the same ring.
        //
        // O_DIRECT is a property of the open file, not of the ring. Each request carries its own
        // FileRef, and the engine reads that request's `direct` flag when it decides whether to set
        // IOSQE_ASYNC. There is no ring-wide setting to choose here.
        //
        // Under IoUringDirect the worker still decides file by file. It reads a file buffered if the
        // file offset is not congruent with the destination, or if the mount refuses O_DIRECT. So one
        // engine serves both kinds of file even when the strategy is the direct one.
        //
        // The two strategies must succeed or fail together. StrategyResolver reports them as equally
        // available, so if this refused one of them, resolution would succeed and the first workload
        // would then fail with no engine.
        return make_io_uring_engine(strategy, config);

    case Strategy::LibaioDirect:
        return make_libaio_engine(config);

    default:
        return nullptr;
    }
}

}; // namespace runai::llm::streamer::posix_io
