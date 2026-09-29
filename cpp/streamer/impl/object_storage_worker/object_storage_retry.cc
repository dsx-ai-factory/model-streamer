/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/object_storage_worker/object_storage_retry.h"

#include <algorithm>
#include <cstdint>
#include <random>

namespace runai::llm::streamer::impl
{

ObjectStorageRetry::ObjectStorageRetry(Duration timeout) :
    ObjectStorageRetry(timeout, retry_delay)
{}

ObjectStorageRetry::ObjectStorageRetry(Duration timeout, DelayGenerator delay_generator) :
    _timeout(timeout),
    _delay_generator(delay_generator)
{}

bool ObjectStorageRetry::begin_attempt(State & state, TimePoint now) const
{
    if (!state._deadline.has_value() && _timeout > Duration::zero())
    {
        state._deadline = now + _timeout;
    }

    // retry_count == 0 is the initial backend attempt, which is allowed even when retry is disabled.
    return state._retry_count == 0 ||
           (state._deadline.has_value() && now < state._deadline.value());
}

std::optional<ObjectStorageRetry::Schedule> ObjectStorageRetry::schedule(
    State & state,
    Handle handle,
    TimePoint now)
{
    if (!state._deadline.has_value() || now >= state._deadline.value())
    {
        return std::nullopt;
    }

    const unsigned next_retry_count = state._retry_count + 1;
    const auto delay = _delay_generator(next_retry_count);
    const auto retry_at = now + delay;
    if (retry_at >= state._deadline.value())
    {
        return std::nullopt;
    }

    _delayed.emplace(retry_at, handle);
    state._retry_count = next_retry_count;
    return Schedule{ delay, next_retry_count };
}

std::optional<ObjectStorageRetry::Handle> ObjectStorageRetry::pop_due(TimePoint now)
{
    const auto due = next_due();
    if (!due.has_value() || due.value() > now)
    {
        return std::nullopt;
    }

    const auto handle = _delayed.begin()->second;
    _delayed.erase(_delayed.begin());
    return handle;
}

std::optional<ObjectStorageRetry::TimePoint> ObjectStorageRetry::next_due() const
{
    return _delayed.empty()
        ? std::nullopt
        : std::optional<TimePoint>(_delayed.begin()->first);
}

bool ObjectStorageRetry::enabled() const
{
    return _timeout > Duration::zero();
}

bool ObjectStorageRetry::has_pending() const
{
    return !_delayed.empty();
}

void ObjectStorageRetry::clear()
{
    _delayed.clear();
}

unsigned ObjectStorageRetry::retry_count(const State & state) const
{
    return state._retry_count;
}

std::chrono::milliseconds ObjectStorageRetry::retry_delay(unsigned retry_count)
{
    // Full jitter over exponential backoff: [0, min(100ms * 2^(n-1), 1s)]. This is the earliest retry
    // time, not an exact schedule: while other chunks are in flight, the worker may be blocked waiting for
    // another completion, so the actual delay can be longer. The per-chunk deadline is the hard bound.
    constexpr uint64_t base_ms = 100;
    constexpr uint64_t cap_ms = 1000;
    const unsigned shift = std::min(retry_count > 0 ? retry_count - 1 : 0, 4u);
    const uint64_t upper_ms = std::min<uint64_t>(cap_ms, base_ms << shift);

    thread_local std::mt19937_64 generator(std::random_device{}());
    std::uniform_int_distribution<uint64_t> distribution(0, upper_ms);
    return std::chrono::milliseconds(distribution(generator));
}

}; // namespace runai::llm::streamer::impl
