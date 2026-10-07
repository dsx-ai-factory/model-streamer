/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <utility>

namespace runai::llm::streamer::utils
{

// CapacityQueue - a credit-bounded pending queue.
//
// Holds items that are waiting to be submitted and bounds how much "work" may be
// in flight at once. The owner submits items by calling try_take() (which reserves
// the item's cost against the capacity) and releases credit by calling complete()
// when an item finishes, which lets the next pending item become takeable.
//
// cost and capacity are in caller-defined units: bytes for the S3 in-flight window
// (bandwidth-delay product), or an operation count for threadpool-bounded backends.
//
// NOT THREAD SAFE - by design. It is meant to be owned by a single worker that
// enqueues, takes and completes on one thread; the cross-thread completion signal
// (e.g. an async callback) is delivered through a separate thread-safe queue, and
// the worker calls complete() from its own loop. Keeping this object single-owner
// makes it a deterministic, independently testable unit.
template <typename T>
class CapacityQueue
{
 public:
    explicit CapacityQueue(size_t capacity) :
        _capacity(capacity)
    {}

    // Add an item to be submitted later.
    // cost - the amount of capacity this item consumes while in flight.
    void enqueue(T item, size_t cost)
    {
        _pending.push_back(Entry{std::move(item), cost});
    }

    // Add a priority item ahead of normally queued work. Intended for work that was already submitted once
    // and became ready to retry; capacity is still reserved only when try_take() returns the item.
    void enqueue_front(T item, size_t cost)
    {
        _pending.push_front(Entry{std::move(item), cost});
    }

    // Return the next item to submit and reserve its cost against the capacity, or
    // std::nullopt if the window is currently full.
    //
    // When nothing is in flight, one item is always returned even if its cost
    // exceeds the whole capacity - otherwise an item larger than the window could
    // never be submitted (deadlock).
    std::optional<T> try_take()
    {
        if (_pending.empty())
        {
            return std::nullopt;
        }

        const size_t cost = _pending.front().cost;

        if (_inflight != 0 && _inflight + cost > _capacity)
        {
            return std::nullopt;
        }

        Entry entry = std::move(_pending.front());
        _pending.pop_front();
        _inflight += entry.cost;
        return std::optional<T>(std::move(entry.item));
    }

    // Release the credit previously reserved for a completed item. Pass the same
    // cost that try_take() reserved for it. Clamps at zero defensively.
    void complete(size_t cost)
    {
        _inflight = (_inflight >= cost) ? _inflight - cost : 0;
    }

    // Drop every pending item and release all reserved credit, leaving the queue idle()
    // in one step. For aborting a window whose pending set may exceed the capacity, where
    // draining via try_take()/complete() would stop at the full-window boundary. The
    // owner must abandon any tracking tied to the dropped items - their costs are gone.
    void clear()
    {
        _pending.clear();
        _inflight = 0;
    }

    // Drop every PENDING item and leave the in-flight credit alone. For an abort that must wait for
    // what the backend already holds: a pending item never reached the backend, so no pointer of ours
    // escaped with it, while an in-flight one may still be written into. Only a real completion may
    // release that credit, which is what makes idle() honest enough to wait on.
    void abort_pending()
    {
        _pending.clear();
    }

    // no items left to submit (some may still be in flight)
    bool empty() const
    {
        return _pending.empty();
    }

    // nothing left to submit and nothing in flight
    bool idle() const
    {
        return _pending.empty() && _inflight == 0;
    }

    size_t inflight() const
    {
        return _inflight;
    }

    size_t pending() const
    {
        return _pending.size();
    }

    size_t capacity() const
    {
        return _capacity;
    }

 private:
    struct Entry
    {
        T item;
        size_t cost;
    };

    size_t _capacity;
    size_t _inflight = 0;
    std::deque<Entry> _pending;   // FIFO; see the selection point in try_take()
};

} // namespace runai::llm::streamer::utils
