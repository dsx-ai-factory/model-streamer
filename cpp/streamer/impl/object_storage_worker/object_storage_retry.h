/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <map>
#include <optional>

#include "common/backend_api/response/response.h"

namespace runai::llm::streamer::impl
{

// Owns the application-level retry policy for one ObjectStorageWorker: per-chunk deadlines and retry
// counts, full-jitter backoff, and the delayed queue. The worker remains responsible for backend I/O and
// capacity accounting; it only releases a failed attempt's capacity when schedule() succeeds.
class ObjectStorageRetry
{
 public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;
    using Handle = common::backend_api::ObjectRequestId_t;
    using DelayGenerator = std::chrono::milliseconds (*)(unsigned retry_count);

    // Stored beside the owning object chunk. Its internals are private so retry state can only be changed by
    // this class; retry_count() exposes the count for diagnostics.
    class State
    {
     public:
        State() = default;

     private:
        std::optional<TimePoint> _deadline;
        unsigned _retry_count = 0;

        friend class ObjectStorageRetry;
    };

    struct Schedule
    {
        std::chrono::milliseconds delay;
        unsigned retry_count;
    };

    explicit ObjectStorageRetry(Duration timeout = Duration::zero());

    // Custom delay generation is exposed for deterministic unit tests.
    ObjectStorageRetry(Duration timeout, DelayGenerator delay_generator);

    // Starts the per-chunk deadline on the first backend attempt and checks whether a later retry may still
    // be submitted. The first attempt is always allowed, including when retries are disabled.
    bool begin_attempt(State & state, TimePoint now = Clock::now()) const;

    // Schedule a failed chunk after backoff. Returns nullopt when retries are disabled, its deadline has
    // expired, or the complete backoff would reach the deadline. State changes only when scheduling succeeds.
    std::optional<Schedule> schedule(State & state, Handle handle, TimePoint now = Clock::now());

    // Return and remove the next retry whose backoff elapsed. A retry becomes eligible at its due time, but
    // the worker may pop it later if it is blocked waiting for another backend completion.
    std::optional<Handle> pop_due(TimePoint now = Clock::now());

    std::optional<TimePoint> next_due() const;

    bool enabled() const;
    bool has_pending() const;
    void clear();
    unsigned retry_count(const State & state) const;

 private:
    static std::chrono::milliseconds retry_delay(unsigned retry_count);

    Duration _timeout;
    DelayGenerator _delay_generator;
    std::multimap<TimePoint, Handle> _delayed;
};

}; // namespace runai::llm::streamer::impl
