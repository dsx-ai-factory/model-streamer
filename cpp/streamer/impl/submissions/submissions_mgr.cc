/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/submissions/submissions_mgr.h"

#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

std::chrono::steady_clock::time_point SubmissionsMgr::default_now()
{
    return std::chrono::steady_clock::now();
}

SubmissionsMgr::SubmissionsMgr(Clock now) :
    _now(std::move(now))
{}

SubmissionId SubmissionsMgr::generate()
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);

    SubmissionId id;
    do
    {
        id = _next_id++;              // unsigned wrap is well-defined
        if (id == 0) id = _next_id++; // 0 is reserved (Response default / "none")
    }
    while (_submissions.count(id) != 0);   // after a 2^64 wrap, skip a still-live id

    return id;
}

void SubmissionsMgr::add(SubmissionId submission_id, unsigned expected, size_t total_bytes)
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    // emplace (not operator[]=) so a colliding id surfaces instead of silently clobbering a live
    // submission. generate() already skips live ids, so this can only fire on an accounting bug.
    const bool inserted = _submissions.emplace(submission_id, Submission{ expected, total_bytes, _now() }).second;
    ASSERT(inserted) << "Submission id " << submission_id << " already registered";
}

SubmissionsMgr::Result SubmissionsMgr::consume(SubmissionId submission_id)
{
    Result result;

    const auto guard = std::unique_lock<std::mutex>(_mutex);

    auto it = _submissions.find(submission_id);
    // A response must belong to a live submission: add() precedes any of its responses, and the
    // entry is erased only on its last consume. An unknown id means the submission over-delivered
    // (more than `expected` responses) - an accounting bug. ASSERT logs and throws (surfaces as
    // UnknownError at the C API), rather than silently masking it.
    ASSERT(it != _submissions.end()) << "Consumed a response for unknown submission " << submission_id;

    if (--it->second.remaining == 0)
    {
        const auto & sub = it->second;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(_now() - sub.submit_time).count();

        result.outcome = Result::Outcome::Completed;
        result.total_bytes = sub.total_bytes;
        result.elapsed_ms = ms;
        result.throughput_bps = (ms > 0) ? (sub.total_bytes * 1000 / static_cast<size_t>(ms)) : 0;

        _submissions.erase(it);
    }

    return result;
}

size_t SubmissionsMgr::size() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _submissions.size();
}

} // namespace runai::llm::streamer::impl
