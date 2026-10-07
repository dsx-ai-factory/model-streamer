/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <type_traits> // For std::is_constructible
#include <typeinfo>
#include <utility>

#include "utils/semaphore/semaphore.h"
#include "common/response_code/response_code.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::common
{

// SharedQueue to hold and return ResponseType objects
// Initialized with the expected number of responses

// ResponseType object must be initialized with a common::ResponseCode
// ResponseType object must have a member 'ret' of type common::ResponseCode
// ResponseType must be movable
// ResponseType must be streamable to std::ostream

// Implements producer-consumer design:
//    push: ready responses are pushed to the queue and notify the semaphore
//    pop : waits on the semaphore and returns the first response in the queue
//          if no responses are expected returns a ResponseType indicating FinishedError

// Designed for multi producers that push responses and a single consumer that is waiting for responses

// Who owns the "finished" decision for this queue:
//  - FINISH_ON_DRAIN: the queue owns it. pop() returns FinishedError once drained
//    (_running == 0 && empty). Used by the backend responders, whose _running/increment
//    track outstanding reads. This is the default and preserves the original behavior.
//  - PERSISTENT: the caller owns it. A drained queue is NOT terminal - pop() blocks (or
//    times out) waiting for future submissions; FinishedError is returned only on stop()
//    (teardown). Used by the user-facing multi-request responder, whose completion is tracked
//    outside the queue (the streamer's per-submission registry).
enum class QueueMode
{
    FINISH_ON_DRAIN,
    PERSISTENT,
};

template <typename ResponseType>
struct SharedQueue
{
    // Static assertion to ensure ResponseType has the expected 'ret' member and constructor.
    // This is a basic check; more sophisticated checks might use SFINAE or concepts in C++20.
    static_assert(std::is_constructible<ResponseType, common::ResponseCode>::value,
                  "ResponseType must be constructible from common::ResponseCode for error states.");

    SharedQueue(unsigned running, QueueMode mode = QueueMode::FINISH_ON_DRAIN);

    ~SharedQueue();

    void increment(unsigned running);

    // Blocking + timed pop.
    //  - returns the next response when one is available;
    //  - returns TimedOut if timeout_ms elapses first (timeout_ms == 0 blocks indefinitely);
    //  - returns FinishedError on stop(), and (FINISH_ON_DRAIN only) when drained.
    // See QueueMode for how the mode affects the drained state.
    ResponseType pop(unsigned timeout_ms = 0);

    void push(ResponseType && response);
    void push(ResponseType && response, size_t bytesize);

    void stop();

    bool finished() const;

    // return throughput in bytes per second
    size_t bytes_per_second() const;

    common::ResponseCode valid() const;

 private:
    // whether the queue owns the "finished" decision (see QueueMode)
    const QueueMode _mode;

    // expected number of responses
    unsigned _running;

    // responses for completed requests
    std::deque< ResponseType > _responses;

    // signals for responses in the queue
    utils::Semaphore _ready;

    // mutex to make the queue thread safe
    mutable std::mutex _mutex;

    std::atomic<bool> _stopped;

    std::atomic<size_t> _total_bytesize;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;

    bool _successful = true;

    std::atomic<bool> _unexpected_push_error;
};


// --- Implementation ---

template <typename ResponseType>
SharedQueue<ResponseType>::SharedQueue(unsigned running, QueueMode mode) :
    _mode(mode),
    _running(running),
    _ready(0), // Semaphore initialized to 0
    _stopped(false),
    _total_bytesize(0),
    _start_time(std::chrono::steady_clock::now()),
    _unexpected_push_error(false)
{
    LOG(DEBUG) << "created responder for " << running << " running requests (Type: " << typeid(ResponseType).name() << ")";
}

template <typename ResponseType>
SharedQueue<ResponseType>::~SharedQueue()
{
    LOG(DEBUG) << "Responder shutdown";
}

template <typename ResponseType>
void SharedQueue<ResponseType>::increment(unsigned running)
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    _running += running;
    LOG(DEBUG) << "Responder incremented, new running count: " << _running;
}

template <typename ResponseType>
ResponseType SharedQueue<ResponseType>::pop(unsigned timeout_ms)
{
    // stop() is terminal in both modes.
    if (_stopped.load(std::memory_order_acquire)) // Use acquire for atomic read
    {
        LOG(DEBUG) << "responder stopped (Type: " << typeid(ResponseType).name() << ")";
        return ResponseType(common::ResponseCode::FinishedError);
    }

    // FINISH_ON_DRAIN owns "finished": a drained queue is terminal at entry.
    // PERSISTENT does not - it blocks / times out below, leaving completion to the caller.
    if (_mode == QueueMode::FINISH_ON_DRAIN && finished())
    {
        LOG(DEBUG) << "responder does not expect any more responses (Type: " << typeid(ResponseType).name() << ")";
        return ResponseType(common::ResponseCode::FinishedError);
    }

    if (timeout_ms == 0)
    {
        _ready.wait(); // block indefinitely until a post (push or stop)
    }
    else if (!_ready.wait_for(timeout_ms))
    {
        return ResponseType(common::ResponseCode::TimedOut);
    }

    const auto guard = std::unique_lock<std::mutex>(_mutex);

    // stop() is the only non-push post; if it woke us, report FinishedError. Otherwise a push-post
    // always has a matching item (single consumer), so the queue is non-empty below.
    if (_stopped.load(std::memory_order_relaxed))
    {
        LOG(DEBUG) << "responder stopped while waiting or after acquiring lock (Type: " << typeid(ResponseType).name() << ")";
        return ResponseType(common::ResponseCode::FinishedError);
    }

    ASSERT(!_responses.empty()) << "responder is empty after notification. Current running " << _running << " (Type: " << typeid(ResponseType).name() << ")";

    ResponseType response = std::move(_responses.front());
    LOG(SPAM) << "Sending response: " << response; // Requires ResponseType to be streamable
    _responses.pop_front();

    return response;
}

template <typename ResponseType>
void SharedQueue<ResponseType>::push(ResponseType && response)
{
    bool posted_to_semaphore = false;
    {
        const auto guard = std::unique_lock<std::mutex>(_mutex);

        if (_stopped.load(std::memory_order_relaxed))
        {
            LOG(DEBUG) << "Responder stopped, ignoring pushed response (Type: " << typeid(ResponseType).name() << ")";
            // ignore responses
            return;
        }

        // Assuming ResponseType has a member 'ret' of type common::ResponseCode
        _successful  = _successful && (response.ret == common::ResponseCode::Success);

        if (_running > 0) // Check _running before decrementing
        {
            LOG(SPAM) << response << " ; " << _running << " running requests (Type: " << typeid(ResponseType).name() << ")";
            _responses.push_back(std::move(response)); // Use std::move
            --_running;
            posted_to_semaphore = true;

            // Throughput logging moved out of the queue (G4): on a PERSISTENT responder
            // _running hits 0 between every submission and _start_time/_total_bytesize are
            // cumulative, so a queue-level throughput number is meaningless. Per-submission
            // throughput is logged by the streamer on request_done instead.
        }
        else
        {
            LOG(ERROR) << "Received unexpected response (no running requests) " << response << " (Type: " << typeid(ResponseType).name() << ")";
            _unexpected_push_error.store(true, std::memory_order_relaxed);
        }
    } // Mutex guard released

    if (posted_to_semaphore)
    {
        _ready.post(); // Signal that a response is ready
    }
}

template <typename ResponseType>
void SharedQueue<ResponseType>::push(ResponseType && response, size_t bytesize)
{
    _total_bytesize.fetch_add(bytesize, std::memory_order_relaxed);
    push(std::move(response)); // Forward to the other push method
}

template <typename ResponseType>
bool SharedQueue<ResponseType>::finished() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    // A responder is finished when all expected responses have been accounted for (_running == 0)
    // AND there are no more responses waiting in the queue to be popped.
    return (_running == 0 && _responses.empty());
}

template <typename ResponseType>
void SharedQueue<ResponseType>::stop()
{
    bool needs_post = false;
    {
        LOG(DEBUG) << "Stopping responder..." << " (Type: " << typeid(ResponseType).name() << ")";
        const auto guard = std::unique_lock<std::mutex>(_mutex);
        if (!_stopped.exchange(true, std::memory_order_acq_rel)) { // Ensure stop is only processed once
            // If there are threads potentially blocked in pop() waiting on _ready,
            // we need to signal them to wake up and see the _stopped flag.
            needs_post = true; // Post regardless of _running, to unblock any waiter
        }
    }

    if (needs_post)
    {
        // For our "single consumer" design, one post is okay.
        _ready.post();
        LOG(DEBUG) << "Responder stopped and semaphore posted (Type: " << typeid(ResponseType).name() << ")";
    }
    else
    {
        LOG(DEBUG) << "Responder already stopped or stop initiated by another thread (Type: " << typeid(ResponseType).name() << ")";
    }
}

template <typename ResponseType>
size_t SharedQueue<ResponseType>::bytes_per_second() const
{
    // This function might be called from push() which holds the mutex.
    //  _start_time is only set in constructor
    // _total_bytesize is atomic
    const auto time_ = std::chrono::steady_clock::now();
    const auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(time_ - _start_time);
    const auto milliseconds = duration.count();

    if (milliseconds == 0)
    {
        return 0; // Avoid division by zero
    }

    LOG(DEBUG) << "Read " << utils::logging::human_readable_size(_total_bytesize.load(std::memory_order_relaxed))
               << " in " << milliseconds << " milliseconds (Type: " << typeid(ResponseType).name() << ")";
    return static_cast<double>(_total_bytesize.load(std::memory_order_relaxed)) / (static_cast<double>(milliseconds) / 1000.0);
}

template <typename ResponseType>
common::ResponseCode SharedQueue<ResponseType>::valid() const
{
    return _unexpected_push_error.load(std::memory_order_relaxed) ? common::ResponseCode::UnknownError : common::ResponseCode::Success;
}

} // namespace runai::llm::streamer::common
