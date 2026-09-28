#include "utils/env/env.h"

#include <unistd.h>
#include <gtest/gtest.h>

#include <climits>

#include "utils/temp/env/env.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::utils
{

TEST(EnvExists, No)
{
    EXPECT_FALSE(env_exists(random::string()));
}

TEST(EnvExists, Yes)
{
    temp::Env env;
    EXPECT_TRUE(env_exists(env.name));
}

TEST(try_getenv__Default, Non_Existing)
{
    std::string s;
    EXPECT_FALSE(try_getenv(random::string(), /* out */ s));
}

TEST(try_getenv__Default, Existing)
{
    temp::Env e;

    std::string s;
    EXPECT_TRUE(try_getenv(e.name, /* out */ s));
    EXPECT_EQ(s, e.value);
}

TEST(try_getenv__String, Non_Existing)
{
    std::string s;
    EXPECT_FALSE(try_getenv<std::string>(random::string(), /* out */ s));
}

TEST(try_getenv__String, Existing)
{
    temp::Env e;

    std::string s;
    EXPECT_TRUE(try_getenv<std::string>(e.name, /* out */ s));
    EXPECT_EQ(s, e.value);
}

TEST(try_getenv__Int, Non_Existing)
{
    int i;
    EXPECT_FALSE(try_getenv<int>(random::string(), /* out */ i));
}

TEST(try_getenv__Int, Existing)
{
    const auto name = random::string();
    const auto value = random::number<int>();

    temp::Env e(name, value);

    int i;
    EXPECT_TRUE(try_getenv<int>(name, /* out */ i));
    EXPECT_EQ(i, value);
}

TEST(try_getenv__Bool, Non_Existing)
{
    bool b;
    EXPECT_FALSE(try_getenv<bool>(random::string(), /* out */ b));
}

TEST(try_getenv__Bool, Existing)
{
    const auto name = random::string();
    const auto value = random::boolean();

    temp::Env e(name, value);

    bool b;
    EXPECT_TRUE(try_getenv<bool>(name, /* out */ b));
    EXPECT_EQ(b, value);
}

TEST(Getenv, Sanity)
{
    temp::Env env;
    EXPECT_EQ(getenv(env.name), env.value);
}

TEST(Getenv, NonExisting)
{
    EXPECT_THROW(getenv(random::string()), std::exception);
}

TEST(Getenv_String, Default)
{
    const auto def = random::string();

    EXPECT_EQ(getenv(random::string(), def), def);
    EXPECT_EQ(getenv<std::string>(random::string(), def), def);
}

TEST(Getenv_Int, Sanity)
{
    const int value = random::number();

    temp::Env env(std::to_string(value));

    EXPECT_EQ(getenv<int>(env.name), value);
    EXPECT_EQ(getenv<int>(env.name, random::number()), value);
}

TEST(Getenv_Int, Negative)
{
    temp::Env env("-1");

    EXPECT_EQ(getenv<int>(env.name), -1);
}

TEST(Getenv_Int, NotInt)
{
    temp::Env env("a");

    EXPECT_THROW(getenv<int>(env.name), std::exception);
    EXPECT_THROW(getenv<int>(env.name, random::number()), std::exception);
}

TEST(Getenv_Unsigned_long, Sanity)
{
    const unsigned long value = random::number<unsigned long>();

    temp::Env env(std::to_string(value));

    EXPECT_EQ(getenv<unsigned long>(env.name), value);
    EXPECT_EQ(getenv<unsigned long>(env.name, random::number<unsigned long>()), value);
}

// A negative value is refused, not wrapped.
//
// This test used to assert the opposite - that "-1" gives ULONG_MAX - which is what std::stoul does
// and what the code did. It reads as a deliberate huge number all the way down: nothing after the
// parse can tell the two apart, because stoul also reports the whole string consumed.
//
// The damage is at the callers. RUNAI_STREAMER_PROCESS_GROUP_SIZE=-1 is a divisor, so Azure
// concurrency falls to one and the async queue depth falls to its minimum - slow, with no error
// anywhere. RUNAI_STREAMER_CONCURRENCY=-1 would ask for a huge thread count; Config::to_concurrency
// caps that one now, but a refusal here names the variable instead of quietly serving the ceiling.
//
// So it throws, exactly as "a" does, and runai_file_streamer_start turns that into InvalidParameterError.
TEST(Getenv_Unsigned_long, Negative)
{
    for (const auto text : { "-1", "-4", " -1" })
    {
        temp::Env env(text);

        EXPECT_THROW(getenv<unsigned long>(env.name), std::exception) << text;
        EXPECT_THROW(getenv<unsigned long>(env.name, random::number()), std::exception) << text;

        unsigned long value = 0;
        EXPECT_THROW(try_getenv<unsigned long>(env.name, /* out */ value), std::exception) << text;
    }
}

// A negative value must not survive as a huge count either - the cap in getenv_positive would turn
// ULONG_MAX into the maximum of T, which is as wrong as the value it came from.
TEST(Getenv_Positive, Negative)
{
    temp::Env env("-1");

    EXPECT_THROW(getenv_positive<unsigned>(env.name, 8U), std::exception);
}

// A plus sign is a sign, not a minus. It must still parse.
TEST(Getenv_Unsigned_long, Explicitly_Positive)
{
    temp::Env env("+8");

    EXPECT_EQ(getenv<unsigned long>(env.name), 8UL);
}

TEST(Getenv_Unsigned_long, NotUnsignedLong)
{
    temp::Env env("a");

    EXPECT_THROW(getenv<unsigned long>(env.name), std::exception);
    EXPECT_THROW(getenv<unsigned long>(env.name, random::number()), std::exception);
}

TEST(Getenv_Bool, True)
{
    temp::Env env("1");

    EXPECT_TRUE(getenv<bool>(env.name));

    for (auto def : { true, false })
    {
        EXPECT_TRUE(getenv<bool>(env.name, def));
    }
}

TEST(Getenv_Bool, False)
{
    temp::Env env("0");

    EXPECT_FALSE(getenv<bool>(env.name));

    for (auto def : { true, false })
    {
        EXPECT_FALSE(getenv<bool>(env.name, def));
    }
}

TEST(Getenv_Bool, Default)
{
    for (auto def : { true, false })
    {
        EXPECT_EQ(getenv<bool>(random::string(), def), def);
    }
}

TEST(Getenv_Bool, NotBool)
{
    temp::Env env;

    EXPECT_THROW(getenv<bool>(env.name), std::exception);

    for (auto def : { true, false })
    {
        EXPECT_THROW(getenv<bool>(env.name, def), std::exception);
    }
}

// A value too large for T must not wrap to ZERO, which is the one answer callers cannot use.
//
// Every caller of this used to floor its own value with std::max(1UL, ...) BEFORE storing it in an
// `unsigned`, and that floor does not survive the narrowing: any non-zero multiple of 2^32 truncates
// to 0. Zero was an integer division by zero in AsyncIoSettings and in the Azure client
// configuration, and a fatal ASSERT in BackendPools.
TEST(getenv_positive, Oversized_Values_Are_Capped_Not_Wrapped)
{
    const std::string name = "RUNAI_STREAMER_TEST_" + random::string();

    // 2^32 exactly, the smallest value that truncates to zero rather than to something merely wrong.
    {
        temp::Env env(name, 4294967296UL);
        EXPECT_EQ(getenv_positive<unsigned>(name, 1U), std::numeric_limits<unsigned>::max());
    }

    // And a multiple of it, so the test is not passing on one special case.
    {
        temp::Env env(name, 8589934592UL);
        EXPECT_EQ(getenv_positive<unsigned>(name, 1U), std::numeric_limits<unsigned>::max());
    }
}

// The floor still applies, and it applies AFTER the cap.
TEST(getenv_positive, Zero_And_Unset_Take_The_Minimum)
{
    const std::string name = "RUNAI_STREAMER_TEST_" + random::string();

    EXPECT_EQ(getenv_positive<unsigned>(name, 8U), 8U) << "unset must give the default";

    temp::Env env(name, 0UL);
    EXPECT_EQ(getenv_positive<unsigned>(name, 8U), 1U) << "zero must be floored, not passed through";
    EXPECT_EQ(getenv_positive<unsigned>(name, 8U, 4U), 4U) << "an explicit minimum must win";
}

// An ordinary value passes through untouched - the guard must not change the normal case.
TEST(getenv_positive, Ordinary_Values_Are_Unchanged)
{
    const std::string name = "RUNAI_STREAMER_TEST_" + random::string();

    temp::Env env(name, 512UL);
    EXPECT_EQ(getenv_positive<unsigned>(name, 1U), 512U);
}

} // namespace runai::llm::streamer::utils
