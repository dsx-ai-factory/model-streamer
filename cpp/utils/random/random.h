/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <time.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <cstdint>

#define DEFAULT_COUNT() (number(10, 100))

namespace runai::llm::streamer::utils::random
{

void seed(unsigned s = ::time(nullptr));

// number

unsigned number(); // [1, 1000)
unsigned number(unsigned max); // [0, max)
unsigned number(unsigned min, unsigned max); // [min, max)

// the following methods return the desired type (int / float / double) but the values are actually unsigned integers

template <typename T, typename... Args>
T number(Args ... args)
{
    return static_cast<T>(number(args...));
}

template <typename T = unsigned>
std::vector<T> numbers(unsigned count = DEFAULT_COUNT(), bool unique = false)
{
    std::vector<T> result(count);

    for (unsigned i = 0; i < result.size(); ++i)
    {
        T generated;

        do
        {
            generated = number<T>();
        }
        while (unique && std::find(result.begin(), result.end(), generated) != result.end()); // NOLINT(whitespace/empty_loop_body)

        result[i] = generated;
    }

    return result;
}

template <typename T = unsigned>
std::vector<T> sequence(T start, T end, int trim = -1)
{
    // create the vector with
    auto sequence = std::vector<T>(end - start);

    // fill it with the sequence [start,end)
    std::iota(sequence.begin(), sequence.end(), start);

    // shuffe the sequence
    std::random_shuffle(sequence.begin(), sequence.end());

    if (trim >= 0)
    {
        // trim the sequence if requested
        sequence = std::vector<T>(sequence.begin(), sequence.begin() + trim);
    }

    return sequence;
}

template <typename T = unsigned>
std::vector<T> sequence(T count)
{
    return sequence(static_cast<T>(0), count);
}

// float

float flot();                       // [0.0, 1.0]
float flot(float max);              // [0.0, max]
float flot(float min, float max);   // [min, max]
std::vector<float> flots(unsigned count = DEFAULT_COUNT()); // [ [0.0, 1.0], ... ]

// string

std::string string(unsigned length = number(15, 20));

template <template <typename> typename T = std::vector>
T<std::string> strings(unsigned count = DEFAULT_COUNT())
{
    T<std::string> result;

    while (result.size() != count)
    {
        result.insert(result.end(), string());
    }

    return result;
}

// ip
std::string ip();

// buffer

std::vector<uint8_t> buffer(size_t length = number(100, 1000));

// very similar to `buffer` but mostly designed for large amounts of data.
// using '/dev/urandom' for randomizing `chunk` bytes (1m by default) and
// then aggregating it to a buffer of size `length` bytes.
std::vector<uint8_t> data(size_t length, size_t chunk = 1e+6);

// boolean

bool boolean();
std::vector<bool> booleans(unsigned count = DEFAULT_COUNT(), bool ensure_both = false);

inline bool chance(float f)
{
    return number(0, 100) < (f * 100);
}

// choice

template <typename T>
T choice(const std::vector<T>& options)
{
    return options[number(options.size())];
}

template <typename T>
std::vector<T> choices(const std::vector<T>& options, unsigned count = DEFAULT_COUNT())
{
    std::vector<T> result(count);

    for (unsigned i = 0; i < result.size(); ++i)
    {
        result[i] = choice(options);
    }

    return result;
}

// subset

template <typename T>
std::set<T> subset(const std::set<T> & s)
{
    std::set<T> result;

    {
        // this is done to make sure the result will always hold at least one element

        typename std::set<T>::iterator it = s.begin();
        std::advance(it, random::number(s.size()));
        result.insert(*it);
    }

    for (const auto & o : s)
    {
        if (boolean())
        {
            result.insert(o);
        }
    }

    return result;
}

template <typename T>
std::vector<T> subset(const std::vector<T> & v)
{
    std::vector<T> result;

    const auto mask = booleans(v.size(), true);

    for (unsigned i = 0; i < v.size(); ++i)
    {
        if (mask.at(i))
        {
            result.push_back(v.at(i));
        }
    }

    return result;
}

// chunks
std::vector<size_t> chunks(size_t total, unsigned count);

template <typename T>
T gen();

template <> inline int          gen<int>()          { return number<int>(); }
template <> inline unsigned     gen<unsigned>()     { return number<unsigned>(); }
template <> inline float        gen<float>()        { return flot(); }
template <> inline bool         gen<bool>()         { return boolean(); }
template <> inline std::string  gen<std::string>()  { return string(); }

template <typename T>
std::vector<T> gen_v(unsigned count = DEFAULT_COUNT());

template <> inline std::vector<int>          gen_v<int>         (unsigned count)    { return numbers<int>(count);       }
template <> inline std::vector<unsigned>     gen_v<unsigned>    (unsigned count)    { return numbers<unsigned>(count);  }
template <> inline std::vector<float>        gen_v<float>       (unsigned count)    { return flots(count);              }
template <> inline std::vector<bool>         gen_v<bool>        (unsigned count)    { return booleans(count);           }
template <> inline std::vector<std::string>  gen_v<std::string> (unsigned count)    { return strings(count);            }

} // namespace runai::llm::streamer::utils::random

#undef DEFAULT_COUNT
