/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

#include "utils/threadpool/threadpool.h"          // Worker<Request>
#include "utils/capacity_queue/capacity_queue.h"

namespace runai::llm::streamer::utils
{

// CapacityWorker - a Worker (see threadpool.h) for asynchronous backends. It owns a CapacityQueue (the
// in-flight window) and implements execute/drain/idle as the submit+drain interleave pattern, so a
// concrete worker only fills the backend hooks (see the protected section). `Chunk` is the unit submitted
// to the backend and held in the window.
//
//  * execute submits a request into the window, draining internally to make room until the request is
//    fully submitted, then drains once more to free a slot - so it returns ready for the next request and
//    can never stall on a full window.
//  * drain progresses the tail one batch at a time (so a newly arrived request can interleave - the pool's
//    worker_routine re-checks the queue between drains).
//  * idle is true once nothing is pending or in flight (and while the window hasn't been created yet).
//
// The window is created lazily via the capacity() hook, which receives the request and returns the window
// size. If capacity() (or the queue allocation) throws - e.g. an empty request, or the backend client cannot
// be built - the base catches it, hands the request to discard() so the worker can finalize it, and retries
// on the next request; the queue is not created until a request can size it, so nothing is chunked before the
// window exists. The pattern lives here once; concrete workers implement capacity/discard/enqueue/submit/
// drain_batch. Backends that don't need a window (filesystem) don't derive from this at all - they use
// ThreadPool's stateless Handler constructor.
template <typename Request, typename Chunk>
class CapacityWorker : public Worker<Request>
{
 public:
    void execute(Request && request, std::atomic<bool> & stopped) final override
    {
        if (_queue == nullptr)   // window not up yet: try to size it from this request
        {
            try
            {
                _queue = std::make_unique<CapacityQueue<Chunk>>(capacity(request));
            }
            catch (...)
            {
                // capacity() (or the queue allocation) could not bring the window up. Let the worker resolve
                // this request - never drop it, or it never completes and the consumer hangs - then retry the
                // window on the next request (the queue is still null).
                discard(std::move(request));
                return;
            }
        }

        enqueue(std::move(request));                    // window is up -> chunk the request into _queue
        while (!_queue->empty())                        // pending = this request's not-yet-submitted chunks
        {
            pump();                                     // submit up to capacity
            if (!_queue->empty())
            {
                drain_batch(stopped);                   // window full -> free slots, then submit more
            }
        }
        if (_queue->inflight() >= _queue->capacity())
        {
            drain_batch(stopped);                       // fully submitted but full -> free a slot for the next
        }
    }

    void drain(std::atomic<bool> & stopped) final override
    {
        if (_queue == nullptr)   // only reached via the pool when !idle(), so normally non-null
        {
            return;
        }
        drain_batch(stopped);                           // one batch, then the pool re-checks for new work
        pump();                                         // submit whatever the freed slots now allow
    }

    bool idle() const final override
    {
        return _queue == nullptr || (_queue->idle() && !has_deferred_work());
    }

 protected:
    // Window size (in Chunk cost units), computed on the first request that can size it. `first_request` lets
    // a worker whose window is only knowable then set it up (e.g. build the client and read max_inflight_bytes).
    // May THROW if the window cannot be sized from this request; the base then calls discard(request) and
    // retries on the next request.
    virtual std::size_t capacity(const Request & first_request) = 0;

    // Resolve a request that could not be admitted because the window failed to come up (capacity() or the
    // queue allocation threw). The worker must finalize it - push its responses - so the consumer never hangs.
    // NOT called once the window is up (admitted requests go through enqueue()).
    //
    // Pure virtual by design: a worker whose capacity() cannot throw still implements it (as an empty body),
    // but must do so consciously. A silent no-op default would be a trap - a subclass whose capacity() throws
    // yet forgets to override it would drop the request and hang the consumer, with no compile-time signal.
    virtual void discard(Request && request) = 0;

    // Split `request` into chunks, _queue->enqueue(...) each, and register it as in flight with whatever
    // per-request/per-task tracking drain_batch needs to route completions to it and finalize it.
    virtual void enqueue(Request && request) = 0;

    // Submit one chunk to the backend (fire the async read).
    virtual void submit(const Chunk & chunk) = 0;

    // Let workers move deferred work into the capacity queue immediately before selecting the next chunks.
    // A worker can combine this with enqueue_front() to prioritize ready retries over new work.
    virtual void pre_pump() {}

    // Process the completions ready now: for each, _queue->complete(cost), route it to its owning request,
    // and finalize+drop the request once its last chunk lands. Blocks for at least one completion when the
    // window is full (nothing else to do); otherwise takes whatever is ready.
    virtual void drain_batch(std::atomic<bool> & stopped) = 0;

    // Asynchronous workers may own scheduled work outside CapacityQueue (for example, jittered retries).
    // Returning true keeps the pool draining even while the main window is temporarily empty.
    virtual bool has_deferred_work() const { return false; }

    std::unique_ptr<CapacityQueue<Chunk>> _queue;   // created lazily once a request can size the window (see execute)

 private:
    void pump()
    {
        pre_pump();
        while (auto chunk = _queue->try_take())
        {
            submit(*chunk);
        }
    }
};

}; // namespace runai::llm::streamer::utils
