/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/backend_api/object_storage/object_storage.h"

#include <gtest/gtest.h>

#include "utils/temp/env/env.h"

namespace runai::llm::streamer::common::backend_api
{

namespace
{

constexpr char MarginEnvVar[] = "RUNAI_STREAMER_INFLIGHT_WINDOW_MARGIN";

} // namespace

TEST(InflightWindowMargin, DefaultWhenUnset)
{
    utils::temp::UnsetEnv unset(MarginEnvVar);
    EXPECT_DOUBLE_EQ(inflight_window_margin(), 1.5);
}

TEST(InflightWindowMargin, ValidPositiveFloat)
{
    utils::temp::UnsetEnv unset(MarginEnvVar);
    utils::temp::Env env(MarginEnvVar, "2.0");
    EXPECT_DOUBLE_EQ(inflight_window_margin(), 2.0);
}

TEST(InflightWindowMargin, TrailingGarbageRejected)
{
    // a numeric prefix followed by junk must not be silently accepted as the prefix value
    utils::temp::UnsetEnv unset(MarginEnvVar);
    utils::temp::Env env(MarginEnvVar, "2.0garbage");
    EXPECT_DOUBLE_EQ(inflight_window_margin(), 1.5);
}

TEST(InflightWindowMargin, NonNumericRejected)
{
    utils::temp::UnsetEnv unset(MarginEnvVar);
    utils::temp::Env env(MarginEnvVar, "abc");
    EXPECT_DOUBLE_EQ(inflight_window_margin(), 1.5);
}

TEST(InflightWindowMargin, NonPositiveRejected)
{
    {
        utils::temp::UnsetEnv unset(MarginEnvVar);
        utils::temp::Env env(MarginEnvVar, "0");
        EXPECT_DOUBLE_EQ(inflight_window_margin(), 1.5);
    }
    {
        utils::temp::UnsetEnv unset(MarginEnvVar);
        utils::temp::Env env(MarginEnvVar, "-2.0");
        EXPECT_DOUBLE_EQ(inflight_window_margin(), 1.5);
    }
}

} // namespace runai::llm::streamer::common::backend_api
