#include "utils/threadpool/threadpool.h"

#include <unistd.h>

#include <gtest/gtest.h>
#include <atomic>
#include <memory>

#include "utils/logging/logging.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::utils
{

namespace
{

// A minimal Worker with no capacity window: execute is synchronous and the worker is always idle. This
// exercises the pool's per-worker routine itself (try_pop / blocking-pop dispatch and clean shutdown),
// independent of the CapacityWorker window machinery (which is covered in capacity_worker_test).
class SyncWorker : public Worker<unsigned>
{
 public:
    explicit SyncWorker(std::atomic<unsigned> & executed) : _executed(executed) {}

    void execute(unsigned && value, std::atomic<bool> &) override { _executed += value; }
    void drain(std::atomic<bool> &) override {}   // nothing is ever in flight
    bool idle() const override { return true; }

 private:
    std::atomic<unsigned> & _executed;
};

} // namespace

TEST(PerWorker, DispatchesEveryRequest)
{
    std::atomic<unsigned> executed{0};

    const unsigned num_workers = utils::random::number(2, 8);

    unsigned total = 0;
    {
        ThreadPool<unsigned> pool(
            [&]() -> std::unique_ptr<Worker<unsigned>>
            {
                return std::make_unique<SyncWorker>(executed);
            },
            num_workers);

        const auto num_requests = utils::random::number(1000, 5000);
        for (unsigned i = 0; i < num_requests; ++i)
        {
            unsigned value = utils::random::number(1, 100);
            total += value;
            pool.push(std::move(value));
        }

        for (int i = 0; i < 5000 && executed.load() < total; ++i)
        {
            ::usleep(1000);
        }
    }   // ~pool stops the deque and joins all workers cleanly

    EXPECT_EQ(executed.load(), total);
}

TEST(Creation, Sanity)
{
    for (auto size : { 1U, 0U, utils::random::number(2, 100), 100U })
    {
        ThreadPool<unsigned> pool([](unsigned, std::atomic<bool> &){}, size);
        (void)pool;
    }
}

TEST(Handle, Sanity)
{
    for (auto size : { 1U, 2U, 10U, utils::random::number(10, 100), 100U })
    {
        constexpr auto Options = 1000;

        std::array<std::atomic<int>, Options> counters{};
        std::atomic<unsigned> total = 0;

        ThreadPool<unsigned> pool([&](unsigned index, std::atomic<bool> & stopped)
            {
                total++;
                counters[index]--;
            }, size);

        const auto count = utils::random::number(50000, 100000);

        for (unsigned i = 0; i < count; ++i)
        {
            auto index = utils::random::number(Options);
            counters[index]++;
            pool.push(std::move(index));
        }

        sleep(1);

        EXPECT_EQ(total, count);

        for (const auto & counter : counters)
        {
            EXPECT_EQ(counter, 0);
        }
    }
}

TEST(Handle, Exception_Thrown_From_Handler)
{
    for (auto size : { 1U, 2U, 10U, utils::random::number(10, 100), 100U })
    {
        std::atomic<unsigned> counter = 0;

        ThreadPool<bool> pool([&](auto _, std::atomic<bool> & stopped)
            {
                counter++;
                throw std::exception();
            }, size);

        const auto count = utils::random::number(50, 100);

        for (unsigned i = 0; i < count; ++i)
        {
            pool.push(utils::random::boolean());
        }

        sleep(1);

        EXPECT_EQ(count, counter);
    }
}

TEST(Handle, Stopped)
{
    for (auto size : { 1U, 2U, 10U, utils::random::number(10, 100), 100U })
    {
        std::atomic<unsigned> total = 0;
        const auto count = utils::random::number(5000, 10000);

        {
            ThreadPool<unsigned> pool([&](auto _, std::atomic<bool> & stopped)
                {
                    while (!stopped)
                    {
                        ::usleep(utils::random::number(10000));
                    }
                    total++;
                }, size);

            for (unsigned i = 0; i < count; ++i)
            {
                pool.push(utils::random::boolean());
            }

            ::usleep(utils::random::number(100000, 1000000));

            EXPECT_EQ(total, 0);
        }

        // verify all threads stopped and each thread performed a single task

        EXPECT_EQ(total, size);
    }
}

} // namespace runai::llm::streamer::utils
