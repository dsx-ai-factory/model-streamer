/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace runai::llm::streamer::posix_io
{

// How the filesystem is read. Three of these are served by an async IoEngine, one by the existing
// 16-thread pool - so the enum spans both, and lives here rather than with the engine.
//
// One value per combination, not an engine axis crossed with a cache axis: libaio + buffered is
// always refused (it degenerates to a serial reader), and a flat enum cannot spell it.
//
// SyncBuffered is the 16-thread pool, unchanged. Not "Threadpool": the thread count is configurable,
// and BackendPools holds three pools, so that name would say neither which nor how many.
enum class Strategy
{
    IoUringDirect,
    IoUringBuffered,
    LibaioDirect,
    SyncBuffered,
};

// Served by an IoEngine? False only for SyncBuffered, which goes to the synchronous pool.
bool is_async(Strategy strategy);

// Opened O_DIRECT? The strategy's default - a mount that cannot do it falls back, and the caller can
// override the cache mode per file.
bool is_direct(Strategy strategy);

// Lowercase, as written in RUNAI_STREAMER_FS_STRATEGY.
const char * name(Strategy strategy);

// False if the name is not one of ours. The caller decides whether that is fatal.
bool strategy_from_name(const std::string & name, Strategy & out);

// Parse RUNAI_STREAMER_FS_STRATEGY: an ordered preference list, best first, e.g.
// "io_uring_direct,io_uring_buffered,sync_buffered". The first candidate the host can provide wins.
//
// Throws InvalidParameterError on an unknown name, a duplicate, an empty entry or an empty list.
// Strict on purpose: a typo must not silently become a fallback nobody asked for.
std::vector<Strategy> parse_candidates(const std::string & value);

std::ostream & operator<<(std::ostream &, Strategy);

}; // namespace runai::llm::streamer::posix_io
