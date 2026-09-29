/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <iostream>

namespace runai
{

namespace
{

struct Urandom
{
    Urandom() : _fd(open("/dev/urandom", O_RDONLY))
    {
        if (_fd == -1)
        {
            perror("Failed opening /dev/urandom");
            exit(EXIT_FAILURE);
        }
    }

    ~Urandom()
    {
        if (_fd != -1)
        {
            close(_fd);
        }
    }

    template <typename T, class = typename std::enable_if<std::is_pod<T>::value, void>::type>
    T read()
    {
        T value = {};

        size_t read_ = 0;
        while (read_ < sizeof(T))
        {
            const ssize_t count = ::read(_fd, reinterpret_cast<void *>((size_t)(&value) + read_), sizeof(T) - read_);

            if (count == -1 || count == 0)
            {
                if (count == -1 && errno == EINTR)
                {
                    continue;
                }

                perror("Failed reading from /dev/urandom");
                exit(EXIT_FAILURE);
            }

            read_ += count;
        }

        return value;
    }

 private:
    int _fd = -1;
};

} // namespace

extern "C" GTEST_API_ int main(int argc, char **argv)
{
    const auto seed = Urandom().read<unsigned int>();
    std::cout << "Using seed " << seed << std::endl;
    srand(seed);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

} // namespace runai
