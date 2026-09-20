#include "streamer/impl/staging_pool/staging_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "device/mock/mock_device.h"

namespace runai::llm::streamer::impl
{

namespace
{

// One scenario: random pool shape, random traffic, run on real threads.
//
// The free list is two atomics with no lock, and a broken memory ordering there fails only
// sometimes. One run proves nothing, so the shape is randomised and the whole thing is repeated -
// the point is many different interleavings, not one long one.
struct Scenario
{
    size_t   buffer_bytesize;
    size_t   slab_bytesize;
    unsigned max_buffers;
    unsigned consumers;     // more than one exercises SharedStagingPool
    unsigned rounds;        // total acquires across all consumers
};

Scenario roll(std::mt19937 & rng)
{
    Scenario s;
    s.buffer_bytesize = 64u << (rng() % 4);                       // 64..512
    s.slab_bytesize = s.buffer_bytesize * (1 + rng() % 4);        // 1..4 buffers per slab
    s.max_buffers = 1 + rng() % 8;
    s.consumers = 1 + rng() % 3;
    s.rounds = 2000 + rng() % 4000;
    return s;
}

// A buffer in flight, with the byte pattern its holder wrote into it.
struct InFlight
{
    StagingBuffer buffer;
    unsigned char pattern;
};

// Runs one scenario and returns a message, empty when every invariant held.
std::string run_scenario(const Scenario & scenario, unsigned seed)
{
    auto mock = std::make_shared<device::MockDevice>();

    StagingPool::Params params;
    params.buffer_bytesize = scenario.buffer_bytesize;
    params.slab_bytesize = scenario.slab_bytesize;
    params.max_buffers = scenario.max_buffers;

    const bool shared = scenario.consumers > 1;
    std::unique_ptr<StagingPool> pool;
    if (shared)
    {
        pool = std::make_unique<SharedStagingPool>(mock, params);
    }
    else
    {
        pool = std::make_unique<StagingPool>(mock, params);
    }

    // Held per buffer index. A buffer handed out while already out is the corruption this is
    // looking for: two readers would write the same bytes.
    std::vector<std::atomic<bool>> held(scenario.max_buffers);
    for (auto & flag : held)
    {
        flag.store(false);
    }

    std::mutex handoff_lock;
    std::deque<InFlight> handoff;
    std::atomic<unsigned> acquired{0};
    std::atomic<bool> failed{false};
    std::string failure;
    std::mutex failure_lock;

    const auto fail = [&](const std::string & message)
        {
            const std::lock_guard<std::mutex> guard(failure_lock);
            if (failure.empty())
            {
                failure = message;
            }
            failed.store(true, std::memory_order_release);
        };

    const auto consume = [&](unsigned id)
        {
            std::mt19937 rng(seed * 31 + id);
            while (!failed.load(std::memory_order_acquire))
            {
                const unsigned mine = acquired.fetch_add(1, std::memory_order_acq_rel);
                if (mine >= scenario.rounds)
                {
                    acquired.fetch_sub(1, std::memory_order_acq_rel);
                    return;
                }

                StagingBuffer buffer;
                if (pool->acquire(buffer) != common::ResponseCode::Success)
                {
                    fail("acquire failed");
                    return;
                }

                if (!buffer.valid())
                {
                    // The non-blocking pool says "everything is in flight". Not an error: give the
                    // round back and try again.
                    acquired.fetch_sub(1, std::memory_order_acq_rel);
                    continue;
                }

                if (buffer.index >= scenario.max_buffers)
                {
                    fail("index past the ceiling");
                    return;
                }
                if (buffer.bytesize != scenario.buffer_bytesize)
                {
                    fail("buffer handed out with the wrong size");
                    return;
                }
                if (held[buffer.index].exchange(true))
                {
                    fail("the same buffer was handed out twice at once");
                    return;
                }

                const unsigned char pattern = static_cast<unsigned char>(rng() & 0xff);
                std::memset(buffer.data, pattern, buffer.bytesize);

                const std::lock_guard<std::mutex> guard(handoff_lock);
                handoff.push_back(InFlight{buffer, pattern});
            }
        };

    // One reaper, as in production: it belongs to the engine, not the device.
    const auto reap = [&]()
        {
            unsigned done = 0;
            while (done < scenario.rounds && !failed.load(std::memory_order_acquire))
            {
                InFlight item;
                {
                    const std::lock_guard<std::mutex> guard(handoff_lock);
                    if (handoff.empty())
                    {
                        continue;
                    }
                    item = handoff.front();
                    handoff.pop_front();
                }

                // Nobody else may have written here while it was out.
                for (size_t i = 0; i < item.buffer.bytesize; ++i)
                {
                    if (static_cast<unsigned char>(item.buffer.data[i]) != item.pattern)
                    {
                        fail("a buffer in flight was written by someone else");
                        return;
                    }
                }

                held[item.buffer.index].store(false);
                pool->release(item.buffer);
                ++done;
            }
        };

    std::vector<std::thread> threads;
    std::thread reaper(reap);
    for (unsigned i = 0; i < scenario.consumers; ++i)
    {
        threads.emplace_back(consume, i);
    }
    for (auto & thread : threads)
    {
        thread.join();
    }

    if (shared)
    {
        // A consumer that gave its round back may still be waiting; without this it waits for a
        // reaper that is about to stop.
        static_cast<SharedStagingPool *>(pool.get())->stop();
    }
    reaper.join();

    if (!failure.empty())
    {
        return failure;
    }
    if (pool->created() > scenario.max_buffers)
    {
        return "the pool grew past its ceiling";
    }
    for (unsigned i = 0; i < scenario.max_buffers; ++i)
    {
        if (held[i].load())
        {
            return "a buffer was never returned";
        }
    }
    return {};
}

} // namespace

// ONE scenario per run, seeded differently each time. The repetition is the runner's job:
//
//     bazel test //device/staging:staging_pool_stress_test --runs_per_test=1000 \
//                --nocache_test_results
//
// A fresh process per run means a different heap layout and a different scheduling pattern, which
// a loop inside one process does not give. Bazel also names the run that failed.
TEST(StagingPoolStress, RandomTrafficAcrossThreads)
{
    // TEST_RUN_NUMBER is set by bazel under --runs_per_test, so a failure names a seed that
    // reproduces it. Falling back to random_device keeps a bare run from repeating one scenario.
    const char * const run = ::getenv("TEST_RUN_NUMBER");
    const unsigned seed = run != nullptr
        ? static_cast<unsigned>(std::strtoul(run, nullptr, 10))
        : std::random_device{}();

    std::mt19937 rng(seed);
    const Scenario scenario = roll(rng);
    const std::string failure = run_scenario(scenario, seed);

    ASSERT_TRUE(failure.empty())
        << failure << "\n  seed=" << seed
        << " buffer=" << scenario.buffer_bytesize
        << " slab=" << scenario.slab_bytesize
        << " max_buffers=" << scenario.max_buffers
        << " consumers=" << scenario.consumers
        << " rounds=" << scenario.rounds;
}

} // namespace runai::llm::streamer::impl
