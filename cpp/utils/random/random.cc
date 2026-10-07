/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/random/random.h"

#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "utils/logging/logging.h"
#include "utils/scope_guard/scope_guard.h"

namespace runai::llm::streamer::utils::random
{

void seed(unsigned s)
{
    ::srand(s);
}

unsigned number()
{
    return number(1, 1000);
}

unsigned number(unsigned max)
{
    return number(0, max);
}

unsigned number(unsigned min, unsigned max)
{
    if (min == max)
    {
        return min;
    }
    else if (min > max)
    {
        throw std::exception();
    }

    return (::rand() % (max - min)) + min;
}

float flot()
{
    return flot(1.0);
}

float flot(float max)
{
    return flot(0.0, max);
}

float flot(float min, float max)
{
    return min + (static_cast<float>(::rand()) / static_cast<float>(RAND_MAX / (max - min)));
}

std::vector<float> flots(unsigned count)
{
    std::vector<float> result(count);

    for (unsigned i = 0; i < result.size(); ++i)
    {
        result[i] = flot();
    }

    return result;
}

std::string string(unsigned length)
{
    const auto randchar = []() -> char
    {
        const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";

        return charset[number(sizeof(charset) - 1)];
    };

    std::string result(length, 0);
    std::generate_n(result.begin(), length, randchar);

    return result;
}

std::string ip()
{
    return \
        std::to_string(number(256)) + "." + \
        std::to_string(number(256)) + "." + \
        std::to_string(number(256)) + "." + \
        std::to_string(number(256));
}

std::vector<uint8_t> buffer(size_t length)
{
    const auto randbyte = []() -> char
    {
        return static_cast<uint8_t>(number(UINT8_MAX));
    };

    std::vector<uint8_t> result(length);
    std::generate_n(result.begin(), length, randbyte);

    return result;
}

std::vector<uint8_t> data(size_t length, size_t chunk)
{
    // read a random block of `chunk` bytes
    std::vector<uint8_t> block(std::min(length, chunk));

    {
        // Open /dev/urandom in read-only and binary mode
        std::ifstream urandom("/dev/urandom", std::ios::in | std::ios::binary);
        PASSERT(urandom) << "Failed to read from /dev/urandom";

        ScopeGuard guard([&](){urandom.close();});

        PASSERT(urandom.is_open()) << "Failed to open /dev/urandom in read-only mode";

        urandom.read(reinterpret_cast<char*>(block.data()), block.size());

        PASSERT(urandom) << "Failed to read from /dev/urandom";
    }

    // create a vector for the result
    auto result = std::vector<uint8_t>();

    // reserve enough space for efficiency
    result.reserve(length);

    // append the random block as many times as needed
    while (result.size() < length)
    {
        result.insert(result.end(), block.begin(), block.begin() + std::min(block.size(), length - result.size()));
    }

    // verify ourselves
    ASSERT(result.size() == length);

    // and we're done
    return result;
}

bool boolean()
{
    return number(2) == 0 ? true : false;
}

std::vector<bool> booleans(unsigned count, bool ensure_both)
{
    std::vector<bool> result(count);

    for (unsigned i = 0; i < result.size(); ++i)
    {
        result[i] = boolean();
    }

    if (ensure_both)
    {
        const unsigned t = number(count);
        unsigned f;

        do
        {
            f = number(count);
        } while (f == t);

        result[t] = true;
        result[f] = false;
    }

    return result;
}

std::vector<size_t> chunks(size_t total, unsigned count)
{
    ASSERT(total >= count) << "Invalid arguments; total = " << total << "; count = " << count;

    // for efficiency in this particular scenario
    if (count == 1)
    {
        return { total };
    }

    auto result = std::vector<size_t>{};
    result.reserve(count);

    auto remaining = total - count; // this ensures that the last chunk will have at least 1 byte
    auto used = 0u;
    for (size_t i = 0; i < count - 1; ++i)
    {
        // We want the chunks to be somewhat of equal size to avoid exausting the total
        // too soon into the iteration, so we limit each allocation depending how much chunks and size
        // is left.
        const auto addition = number<size_t>(1, std::max(1lu, (remaining + 1) / (count-i)));
        used += addition;

        result.push_back(addition);
        remaining -= (remaining >= addition) * addition;
    }

    // We may have exausted `remaining` before finishing the loop, so we take `used` into account when
    // calculating the last chunk
    result.push_back(total - used);

    return result;
}

} // namespace namespace runai::llm::streamer::utils::random
