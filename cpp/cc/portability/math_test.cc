/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <gtest/gtest.h>

extern "C" double __wrap_log(double);
extern "C" double __wrap_log2(double);
extern "C" double __wrap_pow(double, double);

TEST(PortableMath, ValuesAndDomainErrors)
{
    EXPECT_DOUBLE_EQ(__wrap_log(1.0), 0.0);
    EXPECT_DOUBLE_EQ(__wrap_log2(8.0), 3.0);
    EXPECT_DOUBLE_EQ(__wrap_pow(2.0, 3.0), 8.0);
    EXPECT_TRUE(std::isnan(__wrap_log(-1.0)));
    EXPECT_TRUE(std::isinf(__wrap_log(0.0)));
    EXPECT_TRUE(std::isnan(__wrap_pow(-1.0, 0.5)));
}
