/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/random/random.h"

#include <gtest/gtest.h>

#include <map>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::utils::random
{

template <typename T>
struct Number : ::testing::Test
{
    virtual ~Number() = default;
};

template <typename T>
struct Numbers : ::testing::Test
{
    virtual ~Numbers() = default;
};

template <typename T>
struct Sequence : ::testing::Test
{
    virtual ~Sequence() = default;
};

using Types = ::testing::Types<int, unsigned, float, double>;

TYPED_TEST_CASE(Number, Types);
TYPED_TEST_CASE(Numbers, Types);
TYPED_TEST_CASE(Sequence, Types);

TYPED_TEST(Number, Sanity)
{
    const int Min = 1;
    const int Max = 1000;

    for (int i = 0; i < 1000000; ++i)
    {
        const int r = random::number<TypeParam>();

        EXPECT_GE(r, Min);
        EXPECT_LT(r, Max);
    }
}

TYPED_TEST(Number, Max)
{
    const int Max = 10;

    for (int cnt = 0; cnt < 1000000; ++cnt)
    {
        const auto r = random::number<TypeParam>(Max);

        EXPECT_GE(r, 0);
        EXPECT_LT(r, Max);
    }
}

TYPED_TEST(Number, Min_Max)
{
    const int Min = 58;
    const int Max = 217;

    for (int cnt = 0; cnt < 1000000; ++cnt)
    {
        const auto r = random::number<TypeParam>(Min, Max);

        EXPECT_GE(r, Min);
        EXPECT_LT(r, Max);
    }
}

TYPED_TEST(Number, Min_Max__Input_Validation)
{
    const int Min = 58;
    const int Max = 217;

    EXPECT_THROW(random::number<TypeParam>(Max, Min), std::exception);
}

TYPED_TEST(Number, Min_Equals_Max__Succeeds)
{
    const int Min = 217;
    const int Max = 217;

    EXPECT_EQ(random::number<TypeParam>(Max, Min), 217);
    EXPECT_EQ(random::number<TypeParam>(0), 0);
}

TYPED_TEST(Numbers, Count)
{
    const auto count    = random::number(10, 20);
    const auto numbers  = random::numbers<TypeParam>(count);

    EXPECT_EQ(numbers.size(), count);
}

TYPED_TEST(Numbers, Unique)
{
    for (int i = 0; i < 200000; ++i)
    {
        const auto count    = random::number(5, 20);
        const auto numbers  = random::numbers<TypeParam>(count, true);

        for (auto number : numbers)
        {
            EXPECT_EQ(std::count(numbers.begin(), numbers.end(), number), 1);
        }
    }
}

TYPED_TEST(Sequence, Count)
{
    for (auto count : { 0U, 1U, 2U, random::number<unsigned>(10, 100) })
    {
        const auto sequence = random::sequence<TypeParam>(count);

        EXPECT_EQ(sequence.size(), count);

        for (TypeParam i = 0; i < static_cast<TypeParam>(count); ++i)
        {
            EXPECT_EQ(std::count(sequence.begin(), sequence.end(), i), 1);
        }
    }
}

TYPED_TEST(Sequence, Start_End)
{
    for (auto count : { 0U, 1U, 2U, random::number<unsigned>(10, 100) })
    {
        const auto start = random::number<TypeParam>();
        const auto end = start + static_cast<TypeParam>(count);

        const auto sequence = random::sequence<TypeParam>(start, end);

        EXPECT_EQ(sequence.size(), count);

        for (TypeParam i = start; i < end; ++i)
        {
            EXPECT_EQ(std::count(sequence.begin(), sequence.end(), i), 1);
        }
    }
}

TYPED_TEST(Sequence, Start_End_Trim)
{
    for (auto count : { 0U, 1U, 2U, random::number<unsigned>(10, 100) })
    {
        const auto start = random::number<TypeParam>();
        const auto end = start + count;

        for (auto trim : { 0U, random::number<unsigned>(count), count })
        {
            const auto sequence = random::sequence<TypeParam>(start, end, trim);

            EXPECT_EQ(sequence.size(), trim);

            for (const auto & value : sequence)
            {
                EXPECT_GE(value, start);
                EXPECT_LT(value, end);
                EXPECT_EQ(std::count(sequence.begin(), sequence.end(), value), 1);
            }
        }
    }
}

TYPED_TEST(Sequence, Randomness)
{
    const auto start = random::number<TypeParam>();
    const auto end = start + random::number<TypeParam>(100, 200); // to reduce the possibility of collisions

    const auto a = random::sequence<TypeParam>(start, end);
    const auto b = random::sequence<TypeParam>(start, end);

    EXPECT_NE(a, b);
}

TEST(Float, Default)
{
    for (int i = 0; i < 1000000; ++i)
    {
        const auto f = random::flot();

        EXPECT_GE(f, 0.0f);
        EXPECT_LE(f, 1.0f);
    }
}

TEST(Float, Max)
{
    const auto Max = 21.7f;

    bool gto = false;

    for (int i = 0; i < 1000000; ++i)
    {
        const auto f = random::flot(Max);

        if (f > 1.0)
        {
            gto = true;
        }

        EXPECT_GE(f, 0.0f);
        EXPECT_LE(f, Max);
    }

    EXPECT_TRUE(gto);
}

TEST(Float, Min_Max)
{
    const auto Min = -2.17f;
    const auto Max = 21.7f;

    bool gto = false;
    bool ltz = false;

    for (int i = 0; i < 1000000; ++i)
    {
        const auto f = random::flot(Min, Max);

        if (f > 1.0) { gto = true; }
        if (f < 0.0) { ltz = true; }

        EXPECT_GE(f, Min);
        EXPECT_LE(f, Max);
    }

    EXPECT_TRUE(gto);
    EXPECT_TRUE(ltz);
}

TEST(Floats, Count)
{
    const int count = random::number(10, 20);
    const auto flots = random::flots(count);

    EXPECT_EQ(flots.size(), count);
}

TEST(String, Sanity)
{
    const std::string a = random::string();
    const std::string b = random::string();

    EXPECT_NE(a, b);
}

TEST(String, Length)
{
    const unsigned length = random::number(5, 20);
    const std::string str = random::string(length);

    EXPECT_EQ(length, str.size());
}

TEST(Strings, Count)
{
    const unsigned count = random::number(5, 20);
    const std::vector<std::string> strings = random::strings(count);

    EXPECT_EQ(strings.size(), count);
}

TEST(Buffer, Sanity)
{
    const auto first = random::buffer();
    const auto second = random::buffer();

    EXPECT_NE(first, second);
}

TEST(Buffer, Length)
{
    const unsigned length = random::number(5, 200);
    const auto buffer = random::buffer(length);

    EXPECT_EQ(length, buffer.size());
}

TEST(Data, Sanity)
{
    for (auto length :
        {
            1e+0, // 1b
            1e+1, // 10b
            1e+2, // 100b
            1e+3, // 1k
            1e+4, // 10k
            1e+5, // 100k
            1e+6, // 1m
            1e+7, // 10m
            1e+8, // 100m
            1e+9, // 1g
        })
    {
        EXPECT_EQ(random::data(length).size(), length);
    }
}

TEST(Seed, Sanity)
{
    const unsigned Max = 1000000000;

    {
        const unsigned a = random::number(Max);
        const unsigned b = random::number(Max);
        EXPECT_NE(a, b);
    }

    const unsigned seed = random::number(1000);

    {
        random::seed(seed);
        const unsigned a = random::number(Max);
        random::seed(seed);
        const unsigned b = random::number(Max);
        EXPECT_EQ(a, b);
    }

    {
        random::seed(seed);
        const std::string a = random::string();
        random::seed(seed);
        const std::string b = random::string();
        EXPECT_EQ(a, b);
    }
}

TEST(Boolean, Sanity)
{
    bool t = false;
    bool f = false;

    for (int i = 0; i < 1000; ++i)
    {
        if (random::boolean())
        {
            t = true;
        }
        else
        {
            f = true;
        }
    }

    EXPECT_TRUE(t);
    EXPECT_TRUE(f);
}

TEST(Chance, _1_0)
{
    bool t = false;
    bool f = false;

    for (int i = 0; i < 10000; ++i)
    {
        if (random::chance(1.0))
        {
            t = true;
        }
        else
        {
            f = true;
        }
    }

    EXPECT_TRUE(t);
    EXPECT_FALSE(f);
}

TEST(Chance, _0_0)
{
    bool t = false;
    bool f = false;

    for (int i = 0; i < 10000; ++i)
    {
        if (random::chance(0.0))
        {
            t = true;
        }
        else
        {
            f = true;
        }
    }

    EXPECT_FALSE(t);
    EXPECT_TRUE(f);
}

TEST(Chance, _0_x)
{
    for (auto p : { 0.1, 0.5, 0.9 })
    {
        bool t = false;
        bool f = false;

        for (int i = 0; i < 100000; ++i)
        {
            if (random::chance(p))
            {
                t = true;
            }
            else
            {
                f = true;
            }
        }

        EXPECT_TRUE(t);
        EXPECT_TRUE(f);
    }
}

TEST(Booleans, Count)
{
    const auto count = random::number(10, 100);
    const auto bools = random::booleans(count);

    EXPECT_EQ(bools.size(), count);
}

TEST(Booleans, Ensure)
{
    {
        // sanity

        for (int i = 0; i < 10000; ++i)
        {
            const auto bools = random::booleans(2, true);
            EXPECT_NE(bools.at(0), bools.at(1));
        }
    }

    {
        // complex

        for (int i = 0; i < 10000; ++i)
        {
            const auto count = random::number(10, 100);
            const auto bools = random::booleans(count, true);

            const auto t = std::find(bools.begin(), bools.end(), true);
            const auto f = std::find(bools.begin(), bools.end(), false);

            EXPECT_TRUE(t != bools.end());
            EXPECT_TRUE(f != bools.end());
        }
    }
}

TEST(Choice, Sanity)
{
    const auto options = random::numbers();

    std::map<unsigned, bool> found;

    for (int i = 0; i < 1000; ++i)
    {
        found[random::choice<unsigned>(options)] = true;
    }

    for (auto option : options)
    {
        EXPECT_TRUE(found[option]);
    }
}

TEST(Choices, Count)
{
    const auto count = random::number(5, 10);
    const auto choices = random::choices<int>({ 1, 2, 3 }, count);

    EXPECT_EQ(choices.size(), count);
}

TEST(Subset, Set)
{
    std::set<unsigned> s;

    for (unsigned i = random::number(5, 10); i > 0; --i)
    {
        s.insert(random::number());
    }

    const auto sub = random::subset(s);

    EXPECT_GT(sub.size(), 0);
    EXPECT_LE(sub.size(), s.size());

    for (auto o : sub)
    {
        EXPECT_TRUE(s.find(o) != s.end());
    }
}

TEST(Subset, Vector)
{
    std::vector<unsigned> v = random::numbers();

    const auto sub = random::subset(v);

    EXPECT_GT(sub.size(), 0);
    EXPECT_LE(sub.size(), v.size());

    for (auto o : sub)
    {
        EXPECT_TRUE(std::find(v.begin(), v.end(), o) != v.end());
    }
}

TEST(Chunks, Input_Validation)
{
    for (const auto count : { 1u, random::number<unsigned>(1, 10), 10u })
    {
        for (const auto total : { 0u, random::number<unsigned>(0, count - 1) })
        {
            EXPECT_THROW(random::chunks(total, count), std::exception);
        }
    }
}

TEST(Chunks, Sanity)
{
    for (auto total : { 1u, random::number<unsigned>(1, 10), random::number<unsigned>(10, 100), 100u })
    {
        for (auto count : { 1u, random::number<unsigned>(1u, total), total })
        {
            const auto chunks = random::chunks(total, count);

            EXPECT_EQ(chunks.size(), count);

            unsigned sum = 0;
            for (auto chunk : chunks)
            {
                EXPECT_GT(chunk, 0);
                sum += chunk;
            }

            EXPECT_EQ(sum, total);
        }
    }
}

TEST(Chunks, Huge)
{
    auto total = 100000000u; // 100MB
    auto count = 1000;
    const auto chunks = random::chunks(total, count);

    EXPECT_EQ(chunks.size(), count);

    unsigned sum = 0;
    for (auto chunk : chunks)
    {
        sum += chunk;
    }

    EXPECT_EQ(sum, total);
}

} // namespace runai::llm::streamer::utils::random
