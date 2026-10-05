#pragma once

#include <deque>
#include <mutex>
#include <utility>

#include "utils/logging/logging.h"
#include "utils/semaphore/semaphore.h"

namespace runai::llm::streamer::utils
{

template <typename Message>
struct Deque
{
    void push(Message && message)
    {
        {
            const auto lock = std::unique_lock<std::mutex>(_mutex);

            ASSERT(!_stopped) << "Pushing a message to an already stopped queue";

            _deque.push_back(std::move(message));
        }

        _sem.post(); // notify about the new message
    }

    bool pop(/* out */ Message & message)
    {
        _sem.wait(); // wait for a message

        const auto lock = std::unique_lock<std::mutex>(_mutex);

        if (_stopped)
        {
            return false;
        }

        /* out */ message = std::move(_deque.front());
        _deque.pop_front();

        return true;
    }

    // Non-blocking pop: return false immediately when there is no message (or the deque is stopped),
    // otherwise pop the front message and return true. Lets a worker check for new work without parking.
    bool try_pop(/* out */ Message & message)
    {
        if (!_sem.try_wait())
        {
            return false; // no pending token -> empty and not woken by stop()
        }

        const auto lock = std::unique_lock<std::mutex>(_mutex);

        if (_stopped)
        {
            _sem.post(); // return the stop token so a blocking pop() still observes the shutdown
            return false;
        }

        /* out */ message = std::move(_deque.front());
        _deque.pop_front();

        return true;
    }

    // any unresolved messages in the deque will be dropped
    void stop(unsigned times) // `times` is the number of times to increment the semaphore
    {
        {
            const auto lock = std::unique_lock<std::mutex>(_mutex);

            if (_deque.size() != 0)
            {
                LOG(DEBUG) << "Stopping a `Deque` with unresolved messages";
            }
            _stopped = true;
        }

        for (unsigned i = 0; i < times; ++i)
        {
            _sem.post();
        }
    }

    unsigned size() const // get the current size of the deque
    {
        const auto lock = std::unique_lock<std::mutex>(_mutex);

        return _deque.size();
    }

 private:
    Semaphore _sem = 0; // no messages are available
    std::deque<Message> _deque;
    bool _stopped = false;
    mutable std::mutex _mutex; // guarding `_deque` and `_stopped`
};

} // namespace runai::llm::streamer::utils
