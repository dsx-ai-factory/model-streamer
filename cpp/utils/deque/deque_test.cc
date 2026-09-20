#include "utils/deque/deque.h"

#include <gtest/gtest.h>

namespace runai::llm::streamer::utils
{

TEST(Deque, TryPop)
{
    Deque<int> deque;
    int out = 0;

    // empty -> false, non-blocking
    EXPECT_FALSE(deque.try_pop(out));

    deque.push(42);
    EXPECT_TRUE(deque.try_pop(out));
    EXPECT_EQ(out, 42);
    EXPECT_FALSE(deque.try_pop(out));   // drained again

    deque.push(1);
    deque.push(2);
    deque.push(3);
    EXPECT_TRUE(deque.try_pop(out)); EXPECT_EQ(out, 1);   // FIFO
    EXPECT_TRUE(deque.try_pop(out)); EXPECT_EQ(out, 2);
    EXPECT_TRUE(deque.try_pop(out)); EXPECT_EQ(out, 3);
    EXPECT_FALSE(deque.try_pop(out));
}

TEST(Deque, TryPopAfterStopReturnsFalse)
{
    Deque<int> deque;
    deque.push(7);
    deque.stop(1);   // stopped: unresolved messages are dropped

    int out = 0;
    // stopped -> try_pop returns false (and does not consume the stop token needed by pop())
    EXPECT_FALSE(deque.try_pop(out));
    // a blocking pop() still observes the shutdown and returns false without blocking
    EXPECT_FALSE(deque.pop(out));
}

} // namespace runai::llm::streamer::utils
