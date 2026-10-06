/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <utility>

#include "utils/thread/thread.h"
#include "utils/deque/deque.h"

namespace runai::llm::streamer::utils
{

// One thread consuming messages in order, which FINISHES what is queued before it stops.
//
// The difference from ThreadPool is the teardown, and it matters when the messages are promises
// rather than work. ThreadPool's destructor calls Deque::stop(), which drops whatever is still
// queued - correct when an entry is "work to do" and the streamer is being abandoned. It is wrong
// when an entry is "a completion somebody is waiting for", because dropping one leaves that caller
// waiting forever.
//
// The drain is the sentinel and nothing else: the queue is FIFO, so everything pushed before it is
// handled before the thread sees it, and join() is then the whole wait. There is no "stopped" state
// that push has to check.
//
// PRECONDITION: no push may be in flight when stop() is called. Every user drains its own work
// first - an engine stops issuing before it stops its waiter - so this costs nothing, and paying
// for a guarantee nobody needs would put a lock on every push. A push after stop is a caller bug:
// its message is queued behind a sentinel nobody will read, and the ASSERT below says so.
//
// The thread starts on the FIRST push, so an object that is never used costs nothing.
//
// Message must be default-constructible and movable.
template <typename Message>
class DrainingWorker
{
 public:
    using Handler = std::function<void(Message &&)>;

    explicit DrainingWorker(Handler handler) :
        _handler(std::move(handler))
    {
    }

    ~DrainingWorker()
    {
        stop();
    }

    DrainingWorker(const DrainingWorker &) = delete;
    DrainingWorker & operator=(const DrainingWorker &) = delete;

    // Hands a message to the worker, starting the thread the first time.
    void push(Message && message)
    {
        ASSERT(!_stopped.load()) << "Pushing to a DrainingWorker that has already been stopped";

        std::call_once(_once, [this]()
            {
                _thread = Thread(std::bind(&DrainingWorker::run, this));
                _started.store(true);
            });

        _queue.push(Entry{false, std::move(message)});
    }

    // Handles what is already queued, then ends the thread. Safe to call more than once.
    void stop()
    {
        if (_stopped.exchange(true))
        {
            return;
        }

        if (!_started.load())
        {
            return;   // never used, so there is nothing to drain or join
        }

        _queue.push(Entry{true, Message{}});
        _thread.join();
    }

    bool running() const
    {
        return _started.load() && !_stopped.load();
    }

    // Messages handed to the handler. The sentinel is not one of them.
    unsigned handled() const
    {
        return _handled.load();
    }

 private:
    struct Entry
    {
        bool last = false;
        Message message;
    };

    void run()
    {
        while (true)
        {
            Entry entry;
            if (!_queue.pop(entry) || entry.last)
            {
                return;
            }

            _handler(std::move(entry.message));
            ++_handled;
        }
    }

    const Handler _handler;

    Deque<Entry> _queue;
    std::atomic<unsigned> _handled{0};

    std::once_flag _once;
    std::atomic<bool> _started{false};
    std::atomic<bool> _stopped{false};
    Thread _thread;
};

} // namespace runai::llm::streamer::utils
