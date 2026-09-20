#include "utils/draining_worker/draining_worker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace runai::llm::streamer::utils
{

namespace
{

// One scenario: random traffic from a random number of pushers, stopped at a random moment.
//
// The queue, the lazy start and the drain are all cross-thread, and a mistake in any of them shows
// only sometimes. The shape is randomised and the runner repeats it, so the interleavings differ
// between runs rather than within one.
struct Scenario
{
    unsigned pushers;        // threads calling push()
    unsigned per_pusher;     // messages each one sends
    unsigned handler_delay;  // microseconds spent in the handler, to change the timing
};

Scenario roll(std::mt19937 & rng)
{
    Scenario s;
    s.pushers = 1 + rng() % 4;
    s.per_pusher = 20 + rng() % 200;
    s.handler_delay = rng() % 3;
    return s;
}

std::string run_scenario(const Scenario & scenario)
{
    std::mutex seen_lock;
    std::vector<unsigned> seen;
    std::atomic<unsigned> handled{0};
    std::atomic<bool> handler_reentered{false};
    std::atomic<int> inside{0};

    DrainingWorker<unsigned> worker([&](unsigned && value)
        {
            // One thread only: two handlers running at once would be a broken worker.
            if (inside.fetch_add(1, std::memory_order_acq_rel) != 0)
            {
                handler_reentered.store(true, std::memory_order_release);
            }

            if (scenario.handler_delay != 0)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(scenario.handler_delay));
            }

            {
                const std::lock_guard<std::mutex> guard(seen_lock);
                seen.push_back(value);
            }
            handled.fetch_add(1, std::memory_order_acq_rel);

            inside.fetch_sub(1, std::memory_order_acq_rel);
        });

    std::vector<std::thread> pushers;
    for (unsigned p = 0; p < scenario.pushers; ++p)
    {
        pushers.emplace_back([&, p]()
            {
                for (unsigned i = 0; i < scenario.per_pusher; ++i)
                {
                    unsigned value = p * scenario.per_pusher + i;
                    worker.push(std::move(value));
                }
            });
    }

    // Every pusher finishes before stop: that is the contract, not a convenience. Stopping while a
    // push is in flight would queue a message behind a sentinel nobody reads, which is why push()
    // has no refusal path to test.
    for (auto & thread : pushers)
    {
        thread.join();
    }
    worker.stop();

    if (handler_reentered.load(std::memory_order_acquire))
    {
        return "two messages were handled at the same time";
    }

    const unsigned total = scenario.pushers * scenario.per_pusher;
    if (handled.load() != total)
    {
        return "a pushed message was never handled";
    }
    if (worker.handled() != total)
    {
        return "handled() disagrees with what the handler saw";
    }

    const std::lock_guard<std::mutex> guard(seen_lock);
    if (seen.size() != total)
    {
        return "the handler saw a different number of messages than were pushed";
    }
    if (worker.running())
    {
        return "still running after stop";
    }
    return {};
}

} // namespace

// ONE scenario per run. Repeat it with the runner, which gives a fresh process each time:
//
//     bazel test //utils/draining_worker:draining_worker_stress_test \
//                --runs_per_test=10000 --nocache_test_results
TEST(DrainingWorkerStress, RandomTrafficAndStops)
{
    const char * const run = ::getenv("TEST_RUN_NUMBER");
    const unsigned seed = run != nullptr
        ? static_cast<unsigned>(std::strtoul(run, nullptr, 10))
        : std::random_device{}();

    std::mt19937 rng(seed);
    const Scenario scenario = roll(rng);
    const std::string failure = run_scenario(scenario);

    ASSERT_TRUE(failure.empty())
        << failure << "\n  seed=" << seed
        << " pushers=" << scenario.pushers
        << " per_pusher=" << scenario.per_pusher
        << " handler_delay=" << scenario.handler_delay;
}

} // namespace runai::llm::streamer::utils
