/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <gtest/gtest.h>

extern "C" double __wrap_log(double);
extern "C" double __wrap_log2(double);
extern "C" double __wrap_pow(double, double);

TEST(PortableMath, LinkerRedirects)
{
    // Volatile pointers preserve real symbol references instead of compiler builtins.
    double (*volatile log_fn)(double) = std::log;
    double (*volatile log2_fn)(double) = std::log2;
    double (*volatile pow_fn)(double, double) = std::pow;

    EXPECT_EQ(log_fn, &__wrap_log);
    EXPECT_EQ(log2_fn, &__wrap_log2);
    EXPECT_EQ(pow_fn, &__wrap_pow);
}

TEST(PortableMath, ValuesAndDomainErrors)
{
    double (*volatile log_fn)(double) = std::log;
    double (*volatile log2_fn)(double) = std::log2;
    double (*volatile pow_fn)(double, double) = std::pow;

    EXPECT_DOUBLE_EQ(log_fn(1.0), 0.0);
    EXPECT_DOUBLE_EQ(log2_fn(8.0), 3.0);
    EXPECT_DOUBLE_EQ(pow_fn(2.0, 3.0), 8.0);
    EXPECT_TRUE(std::isnan(log_fn(-1.0)));
    EXPECT_TRUE(std::isinf(log_fn(0.0)));
    EXPECT_TRUE(std::isnan(pow_fn(-1.0, 0.5)));
}
