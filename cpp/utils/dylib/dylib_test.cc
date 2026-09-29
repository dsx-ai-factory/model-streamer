/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/dylib/dylib.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <utility>

#include "utils/fd/fd.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::utils
{

namespace
{

bool loaded(const std::string & path)
{
    temp::Path temp;

    std::string cmd = "cat /proc/" + std::to_string(getpid()) + "/maps | grep " + path + " | wc -l > " + temp.path;
    int ret = system(cmd.c_str());
    (void) ret;

    auto buffer = Fd::read(temp.path);
    auto raw = std::string(buffer.begin(), buffer.end());
    EXPECT_GT(raw.size(), 0);
    raw.erase(std::remove(raw.begin(), raw.end(), '\n'), raw.end());

    std::string::size_type idx;
    const int count = std::stoi(raw, /* out */ &idx);

    EXPECT_EQ(idx, raw.size());

    return count > 0;
}

namespace Helper
{

const char Path[] = "./utils/dylib/dylib_test_helper";

namespace Symbol
{

using Type = int(*)();
const char Name[] = "foo";
const int Result = 217;

} // namespace Symbol

} // namespace Helper

} // namespace

TEST(Load, Sanity)
{
    ASSERT_FALSE(loaded(Helper::Path));

    auto dylib = Dylib(Helper::Path);
    (void) dylib;

    EXPECT_TRUE(loaded(Helper::Path));
}

TEST(Load, NonExisting)
{
    EXPECT_THROW(Dylib d(random::string()), std::exception);
}

TEST(Creation, Empty)
{
    auto dylib = Dylib();
    (void) dylib;
}

TEST(Operator_Bool, Const____Invalid)
{
    const auto dylib = Dylib();

    EXPECT_FALSE(dylib);
    EXPECT_FALSE(dylib.operator bool());
}

TEST(Operator_Bool, Const____Valid)
{
    const auto dylib = Dylib(Helper::Path);

    EXPECT_TRUE(dylib);
    EXPECT_TRUE(dylib.operator bool());
}

TEST(Operator_Bool, Non_Const____Invalid)
{
    auto dylib = Dylib();

    EXPECT_FALSE(dylib);
    EXPECT_FALSE(dylib.operator bool());
}

TEST(Operator_Bool, Non_Const____Valid)
{
    auto dylib = Dylib(Helper::Path);

    EXPECT_TRUE(dylib);
    EXPECT_TRUE(dylib.operator bool());
}

TEST(Dlsym, Sanity)
{
    Dylib dylib(Helper::Path);

    auto foo = dylib.dlsym<Helper::Symbol::Type>(Helper::Symbol::Name);
    EXPECT_EQ(Helper::Symbol::Result, foo());
}

TEST(Dlsym, NonExisting)
{
    Dylib dylib(Helper::Path);

    EXPECT_THROW(dylib.dlsym(random::string()), std::exception);
}

TEST(StaticDlsym, Sanity)
{
    for (auto flag : { RTLD_DEFAULT, RTLD_NEXT })
    {
        EXPECT_EQ(
            Dylib::dlsym(flag, "dlopen"),
            ::dlopen);
    }
}

TEST(StaticDlsym, FunctionType)
{
    for (auto flag : { RTLD_DEFAULT, RTLD_NEXT })
    {
        EXPECT_EQ(
            Dylib::dlsym<decltype(dlopen)>(flag, "dlopen"),
            ::dlopen);
    }
}

TEST(StaticDlsym, NonExisting)
{
    for (auto handle : { RTLD_DEFAULT, RTLD_NEXT })
    {
        EXPECT_THROW(
            Dylib::dlsym(handle, random::string()),
            std::exception);
    }
}

} // namespace runai::llm::streamer::utils
