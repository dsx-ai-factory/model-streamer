/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <semaphore.h>

namespace runai::llm::streamer::utils
{

struct Semaphore
{
    Semaphore(unsigned value);
    ~Semaphore();

    // increment and decrement the semaphore
    void post();
    void wait();

    // non-blocking decrement: returns true if the semaphore was acquired (was > 0), false otherwise
    bool try_wait();

    // wait up to timeout_ms for the semaphore; returns true if acquired (decremented),
    // false on timeout. Uses sem_clockwait(CLOCK_MONOTONIC) on glibc >= 2.30 (immune to
    // wall-clock jumps); falls back to sem_timedwait(CLOCK_REALTIME) on older glibc, where
    // an NTP or admin wall-clock jump can skew the wait duration.
    bool wait_for(unsigned timeout_ms);

    // get the semaphore value
    unsigned value();

 private:
    sem_t _sem;
};

} // namespace runai::llm::streamer::utils
