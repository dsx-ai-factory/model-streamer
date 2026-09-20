#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <deque>
#include <mutex>

#include "utils/logging/logging.h"
#include "utils/deque/deque.h"
#include "utils/thread/thread.h"
#include "utils/semaphore/semaphore.h"

namespace runai::llm::streamer::utils
{

// Worker - the per-worker contract for ThreadPool's per-worker mode (below). Intentionally minimal and
// Chunk-agnostic so the pool can hold it as unique_ptr<Worker<Request>> without knowing anything about a
// worker's internals. The pool owns the queue and the stop flag and drives these three calls; the worker
// owns whatever per-thread state it needs (e.g. an in-flight window) and never touches the queue.
//
// CapacityWorker<Request, Chunk> (in capacity_worker.h) is the reusable implementation of this interface
// for async backends: it owns a CapacityQueue and implements execute/drain/idle as the submit+drain
// interleave pattern, leaving concrete workers just the backend hooks. Backends that don't need a window
// (filesystem) don't use Worker at all - they use ThreadPool's stateless Handler constructor instead.
template <typename Request>
struct Worker
{
    virtual ~Worker() = default;

    // Submit `request` into the worker (draining internally to make room until it is fully submitted),
    // then drain until one more completion frees room; returns ready for the next request.
    virtual void execute(Request && request, std::atomic<bool> & stopped) = 0;

    // One batch: process the completions ready now, finalize finished work. Used to progress the tail
    // when there is no next request to execute.
    virtual void drain(std::atomic<bool> & stopped) = 0;

    // True when the worker has nothing pending and nothing in flight.
    virtual bool idle() const = 0;
};

// ThreadPool - a fixed set of worker threads over a shared FIFO queue. Two modes, both sharing the queue,
// thread management, push() and the stop-and-join teardown; they differ only in what each thread runs:
//
//  * Stateless (Handler ctor): each thread pops one request and calls a shared handler - the classic
//    thread pool. Used for synchronous work (filesystem reads; the s3/gcs plugins' own internal pools).
//
//  * Per-worker (WorkerFactory ctor): each thread owns a Worker (built by the factory) and drives its own
//    loop - pull the next request without blocking and hand it to the worker, else drain the worker's tail
//    a batch at a time, else park on the blocking pop. This lets a worker interleave a newly arrived
//    request with async work still in flight (see worker_routine). Used for object storage, where the
//    workers are CapacityWorkers. The factory builds concrete workers (upcast to unique_ptr<Worker<Request>>),
//    so the pool stays agnostic of the worker's queue/chunk types.
template <typename Request>
struct ThreadPool
{
    using Handler = std::function<void(Request &&, std::atomic<bool> &)>;
    using WorkerFactory = std::function<std::unique_ptr<Worker<Request>>()>;

    ThreadPool(const Handler & handler, unsigned size) :
        stopped(false),
        _handler(handler)
    {
        _threads.reserve(size);
        for (unsigned i = 0; i < size; ++i)
        {
            _threads.emplace_back(std::bind(routine, std::ref(*this)));
        }
    }

    // Per-worker mode: each thread owns a Worker (built by `factory`) and pulls from the shared queue
    // itself, so it can interleave a newly arrived request while draining earlier ones. The queue and the
    // stop flag stay inside the pool; the worker only ever sees requests.
    ThreadPool(const WorkerFactory & factory, unsigned size) :
        stopped(false)
    {
        _workers.reserve(size);
        _threads.reserve(size);
        for (unsigned i = 0; i < size; ++i)
        {
            _workers.emplace_back(factory());
            _threads.emplace_back(std::bind(worker_routine, std::ref(*this), _workers.back().get()));
        }
    }

    ~ThreadPool()
    {
        // Set the abort flag BEFORE stopping the deque: a worker that enters drain_batch() then observes
        // `stopped` and takes the fast abort_all path instead of a full backend completion wait. Then stop the
        // deque to wake any thread parked in the blocking pop() so it can exit. (Threads are joined by
        // utils::Thread's destructor when _threads is destroyed, after this body.)
        stopped = true;
        _deque.stop(_threads.size());
    }

    void push(Request && request)
    {
        _deque.push(std::move(request));
    }

    // Requests accepted but not yet taken by a worker. A load measure for choosing between pools -
    // NOT a completion measure: a request a worker has taken is no longer counted here even though
    // its work may still be outstanding.
    unsigned pending() const
    {
        return _deque.size();
    }

    static void routine(ThreadPool & pool)
    {
        while (true)
        {
            Request request;
            if (!pool._deque.pop(/* out */ request))
            {
                break;
            }

            try
            {
                pool._handler(std::move(request), pool.stopped);
            }
            catch (...)
            {
                LOG(WARNING) << "Failed handling request";
            }
        }
    }

    // Per-worker loop: pull the next request without blocking and submit it (interleaving it with the
    // worker's in-flight work); with no next request but work still in flight, drain one batch and
    // re-check; only when idle park on the blocking pop (which also returns false on shutdown -> exit).
    static void worker_routine(ThreadPool & pool, Worker<Request> * worker)
    {
        while (true)
        {
            try
            {
                Request request;
                if (pool._deque.try_pop(/* out */ request))
                {
                    worker->execute(std::move(request), pool.stopped);
                }
                else if (!worker->idle())
                {
                    worker->drain(pool.stopped);
                }
                else
                {
                    if (!pool._deque.pop(/* out */ request))
                    {
                        break;
                    }
                    worker->execute(std::move(request), pool.stopped);
                }
            }
            catch (...)
            {
                LOG(WARNING) << "Failed handling request in worker";
            }
        }
    }

    std::atomic<bool> stopped;

 private:
    Handler _handler;
    Deque<Request> _deque;
    std::vector<std::unique_ptr<Worker<Request>>> _workers;
    std::vector<Thread> _threads;
};

} // namespace runai::llm::streamer::utils
