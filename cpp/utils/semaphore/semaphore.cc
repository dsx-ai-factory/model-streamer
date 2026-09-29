/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/semaphore/semaphore.h"

#include <features.h>

#include <cerrno>
#include <ctime>
#include <chrono>

#include "utils/logging/logging.h"

// Prefer CLOCK_MONOTONIC (immune to host wall-clock changes) via sem_clockwait, added in
// glibc 2.30. The repo floor is glibc 2.29, so fall back to sem_timedwait/CLOCK_REALTIME there.
#if defined(__GLIBC_PREREQ)
#  if __GLIBC_PREREQ(2, 30)
#    define RUNAI_HAVE_SEM_CLOCKWAIT 1
#  endif
#endif

namespace runai::llm::streamer::utils
{

namespace
{

// Convert a chrono time_point to a normalized timespec (tv_nsec in [0, 1e9)); chrono does the
// second/nanosecond split, so there is no hand-rolled tv_nsec carry.
template <typename TimePoint>
struct timespec to_timespec(TimePoint tp)
{
    const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(tp);
    const auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(tp - secs);

    struct timespec ts;
    ts.tv_sec = static_cast<time_t>(secs.time_since_epoch().count());
    ts.tv_nsec = static_cast<long>(nsecs.count());
    return ts;
}

} // namespace

Semaphore::Semaphore(unsigned value)
{
    PASSERT(sem_init(&_sem, 0, value) == 0) << "Failed creating semaphore";
}

Semaphore::~Semaphore()
{
    CHECK(sem_destroy(&_sem) == 0) << "Failed destroying semaphore";
}

void Semaphore::post()
{
    PASSERT(sem_post(&_sem) == 0) << "Could not increment semaphore";
}

void Semaphore::wait()
{
    // restart if interrupted by signal
    int ret{};
    while ((ret = sem_wait(&_sem)) == -1 && errno == EINTR)
    {
        LOG(SPAM) << "Received EINTR while waiting on semaphore " << reinterpret_cast<void*>(&_sem);
        continue;
    }

    PASSERT(ret == 0) << "Could not decrement semaphore";
}

bool Semaphore::try_wait()
{
    // restart if interrupted by signal; non-blocking otherwise
    int ret{};
    while ((ret = sem_trywait(&_sem)) == -1 && errno == EINTR)
    {
        continue;
    }

    if (ret == 0)
    {
        return true;
    }

    // EAGAIN means the semaphore was already zero (nothing to acquire)
    PASSERT(errno == EAGAIN) << "try wait on semaphore failed with errno " << errno;
    return false;
}

bool Semaphore::wait_for(unsigned timeout_ms)
{
    int ret{};

#ifdef RUNAI_HAVE_SEM_CLOCKWAIT
    // steady_clock is CLOCK_MONOTONIC on glibc, so its epoch matches sem_clockwait(CLOCK_MONOTONIC).
    // Immune to host wall-clock changes (NTP / admin adjustments).
    auto deadline = to_timespec(std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms));
    while ((ret = sem_clockwait(&_sem, CLOCK_MONOTONIC, &deadline)) == -1 && errno == EINTR)
    {
        LOG(SPAM) << "Received EINTR while timed-waiting on semaphore " << reinterpret_cast<void*>(&_sem);
        continue;
    }
#else
    // Fallback (glibc < 2.30): CLOCK_REALTIME. A wall-clock jump skews the idle timeout, but a
    // real post() still wakes immediately, so response delivery is never affected.
    auto deadline = to_timespec(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
    while ((ret = sem_timedwait(&_sem, &deadline)) == -1 && errno == EINTR)
    {
        LOG(SPAM) << "Received EINTR while timed-waiting on semaphore " << reinterpret_cast<void*>(&_sem);
        continue;
    }
#endif

    if (ret == 0)
    {
        return true;
    }

    if (errno == ETIMEDOUT)
    {
        return false;
    }

    PASSERT(false) << "timed wait on semaphore failed with errno " << errno;
    return false;
}

unsigned Semaphore::value()
{
    int value;
    PASSERT(sem_getvalue(&_sem, &value) == 0) << "Could not get semaphore value";

    // in Linux, `value` is (0) and not a negative value
    // for more information read `man sem_getvalue`
    return static_cast<unsigned>(value);
}

} // namespace runai::llm::streamer::utils
