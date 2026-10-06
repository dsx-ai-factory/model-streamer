#include "utils/logging/logging.h"

#include <gtest/gtest.h>

#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "utils/random/random.h"
#include "utils/scope_guard/scope_guard.h"
#include "utils/fd/fd.h"
// #include "utils/env/env.h"
#include "utils/temp/env/env.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::utils
{

namespace Helper
{

struct Scope
{
    Scope(std::string& result) :
        _buffer(nullptr),
        _size(0),
        _stream(::open_memstream(&_buffer, &_size)),
        _result(result)
    {
        EXPECT_NE(_stream, nullptr);
        logging::__file = _stream;
    }

    ~Scope()
    {
        EXPECT_EQ(::fflush(_stream), 0);
        _result = std::string{_buffer};

        std::cerr << "result is: " << _result;

        logging::__file = nullptr;

        ::fclose(_stream);
        ::free(_buffer);
    }

 private:
    char * _buffer;
    size_t _size;
    FILE * _stream;
    std::string & _result;
};

bool modify_state(int &state)
{
    ++state;
    return true;
}

struct StreamCallHelper
{
    bool streamed = false;

    friend std::ostream & operator<<(std::ostream & os, StreamCallHelper &helper)
    {
        helper.streamed = true;
        return os;
    }
};

struct print_scope
{
    print_scope(bool value)
    {
        logging::__print = value;
    }

    ~print_scope()
    {
        logging::__print = _prev;
    }

 private:
    bool _prev = logging::__print;
};

namespace A::B::C
{

struct D
{
    static void foo(void * a = nullptr, int b = 217, char c = ' ')
    {
        LOG(ERROR) << random::string();
    }
};

} // namespace A::B::C

struct Printable
{
    Printable(const std::string_view str_):
        str{str_}
    {}

    std::string str;
};

std::ostream & operator<<(std::ostream & os, const Printable& printable)
{
    os << printable.str;
    return os;
}

} // namespace Helper

TEST(Prefix, Function)
{
    std::string log;

    {
        Helper::Scope scope(log);
        Helper::A::B::C::D::foo();
    }

    EXPECT_TRUE(log.find("foo") != std::string::npos);
}

TEST(Prefix, File)
{
    std::string log;

    {
        Helper::Scope scope(log);
        LOG(ERROR) << random::string();
    }

    EXPECT_TRUE(log.find(__FILE__) != std::string::npos);
}

TEST(Prefix, PID)
{
    std::string log;

    {
        Helper::Scope scope(log);
        LOG(ERROR) << random::string();
    }

    EXPECT_TRUE(log.find(std::to_string(::getpid())) != std::string::npos);
}

TEST(Prefix, TID)
{
    std::string log;

    {
        Helper::Scope scope(log);
        LOG(ERROR) << random::string();
    }

    // Use the kernel syscall so this test also builds with glibc older than 2.30.
    EXPECT_TRUE(log.find(std::to_string(::syscall(SYS_gettid))) != std::string::npos);
}

TEST(LOG, Sanity)
{
    const auto message = random::string();
    std::string log;

    {
        Helper::Scope scope(log);
        LOG(ERROR) << message;
    }

    std::string expected(message + "\n");

    EXPECT_GE(log.length(), expected.length());

    EXPECT_EQ(log.compare(log.length() - expected.length(), expected.length(), expected), 0);
}

TEST(LOG, STREAM_CALLED)
{
    Helper::StreamCallHelper helper;
    Helper::print_scope scope(true);

    ASSERT_TRUE(logging::__print);
    ASSERT_TRUE(logging::__minimum <= logging::Level::ERROR);

    ASSERT_FALSE(helper.streamed);
    LOG(ERROR) << helper;
    EXPECT_TRUE(helper.streamed);
}

TEST(LOG, STREAM_NOT_CALLED)
{
    Helper::StreamCallHelper helper;
    Helper::print_scope scope(false);

    ASSERT_FALSE(logging::__print);
    ASSERT_TRUE(logging::__minimum <= logging::Level::ERROR);

    ASSERT_FALSE(helper.streamed);
    LOG(ERROR) << helper;
    EXPECT_FALSE(helper.streamed);
}

TEST(LOG_BASE_IF, True)
{
    const auto message = random::string();

    std::string log;
    {
        Helper::Scope scope(log);
        LOG_BASE_IF(ERROR, false, false, true) << message;
    }

    EXPECT_TRUE(log.find(message) != std::string::npos);
}

TEST(LOG_BASE_IF, Fatal)
{
    const auto message = random::string();

    std::string log;
    {
        Helper::Scope scope(log);
        EXPECT_THROW(LOG_BASE_IF(ERROR, true, false, true) << message, std::exception);
    }
    EXPECT_TRUE(log.find(message) != std::string::npos);
}

TEST(LOG_BASE_IF, False)
{
    std::string log;

    {
        Helper::Scope scope(log);
        LOG_BASE_IF(ERROR, true, false, false) << random::string();
    }

    EXPECT_TRUE(log.empty());
}

TEST(Log_Errno, Sanity)
{
    const auto message = random::string();
    const auto value = random::number(1, 34);

    std::string log;

    {
        Helper::Scope scope(log);
        const auto errno_ = errno;
        ScopeGuard guard([errno_](){ errno = errno_;});

        errno = value;
        LOG_BASE_IF(ERROR, false, true, true) << message;
    }

    EXPECT_TRUE(log.find(message) != std::string::npos);
    EXPECT_TRUE(log.find(std::string(::strerror(value)) + " [" + std::to_string(value) + "]") != std::string::npos);
}

TEST(CHECK, True)
{
    std::string log;

    {
        Helper::Scope scope(log);
        CHECK(true) << random::string();
    }

    EXPECT_TRUE(log.empty());
}

TEST(CHECK, Stream_Not_Called)
{
    Helper::StreamCallHelper helper;

    ASSERT_FALSE(helper.streamed);
    CHECK(true) << helper;
    EXPECT_FALSE(helper.streamed);
}

TEST(CHECK, False)
{
    std::string log;

    {
        Helper::Scope scope(log);
        CHECK(false) << random::string();
    }

    EXPECT_FALSE(log.empty());
}

TEST(ASSERT, True)
{
    std::string log;

    {
        Helper::Scope scope(log);
        ASSERT(true) << random::string();
    }

    EXPECT_TRUE(log.empty());
}

TEST(ASSERT, Throws_Standard_Exception)
{
    try
    {
        ASSERT(true) << random::string();
    }
    catch(const std::system_error& e)
    {
        // Shouldn't happen...
        FAIL();
    }
}

TEST(ASSERT, Stream_Not_Called)
{
    Helper::StreamCallHelper helper;

    ASSERT_FALSE(helper.streamed);
    ASSERT(true) << helper;
    EXPECT_FALSE(helper.streamed);
}

TEST(ASSERT, False)
{
    const auto message = random::string();
    std::string log;

    {
        Helper::Scope scope(log);
        EXPECT_THROW(ASSERT(false) << message, std::exception);
    }

    EXPECT_TRUE(log.find(message) != std::string::npos);
}

TEST(ASSERT, HAS_SIDE_EFFECTS)
{
    int counter = 0;

    // We want to make sure the condition is executed exactly once, regardless if the log is shown or not.
    ASSERT_GT(logging::__minimum, LOGGING_LEVEL(SPAM));

    LOG_BASE_IF(SPAM, false, false, Helper::modify_state(counter));
    EXPECT_EQ(counter, 1);

    LOG_BASE_IF(WARNING, false, false, Helper::modify_state(counter));
    EXPECT_EQ(counter, 2);

    ASSERT(Helper::modify_state(counter));
    EXPECT_EQ(counter, 3);

    PASSERT(Helper::modify_state(counter));
    EXPECT_EQ(counter, 4);

    CHECK(Helper::modify_state(counter));
    EXPECT_EQ(counter, 5);
}

TEST(PASSERT, Throws_SystemError)
{
    try
    {
        const auto errno_ = errno;
        ScopeGuard guard([errno_](){ errno = errno_;});
        errno = 42;

        PASSERT(true) << random::string();
    }
    catch(const std::system_error& e)
    {
        EXPECT_EQ(e.code().value(), 42);
    }
}

TEST(PASSERT, True)
{
    std::string log;

    {
        Helper::Scope scope(log);
        PASSERT(true) << random::string();
    }

    EXPECT_TRUE(log.empty());
}

TEST(PASSERT, Stream_Not_Called)
{
    Helper::StreamCallHelper helper;

    ASSERT_FALSE(helper.streamed);
    PASSERT(true) << helper;
    EXPECT_FALSE(helper.streamed);
}

TEST(PASSERT, False)
{
    const auto message = random::string();
    const auto value = random::number(1, 34);

    std::string log;

    {
        Helper::Scope scope(log);

        errno = value;
        EXPECT_THROW(PASSERT(false) << message, std::exception);
    }

    EXPECT_TRUE(log.find(message) != std::string::npos);
    EXPECT_TRUE(log.find(std::string(::strerror(value)) + " [" + std::to_string(value) + "]") != std::string::npos);
}

namespace
{

std::string helper(
    const std::string & message,
    const std::string & level = "WARNING")
{
    temp::Path temp;

    const auto cmd = std::string("./utils/logging/logging_test_helper ") + level + " " + message + " 2> " + temp.path;
    int ret = ::system(cmd.c_str());
    (void) ret;
    auto buffer = Fd::read(temp.path);
    auto str = std::string(buffer.begin(), buffer.end());

    return str;
}

} // namespace

TEST(RUNAI_STREAMER_LOG_TO_STDERR, Default)
{
    EXPECT_TRUE(helper(random::string()).empty());
}

TEST(RUNAI_STREAMER_LOG_TO_STDERR, True)
{
    const temp::Env RUNAI_STREAMER_LOG_TO_STDERR("RUNAI_STREAMER_LOG_TO_STDERR", true);

    const auto message = random::string();

    EXPECT_TRUE(helper(message).find(message) != std::string::npos);
}

TEST(RUNAI_LOG_TO_STDERR, False)
{
    const temp::Env RUNAI_LOG_TO_STDERR("RUNAI_STREAMER_LOG_TO_STDERR", false);

    EXPECT_TRUE(helper(random::string()).empty());
}

TEST(RUNAI_LOG_FILE, Created)
{
    temp::Path temp;
    const temp::Env RUNAI_STREAMER_LOG_FILE("RUNAI_STREAMER_LOG_FILE", temp.path);

    const auto message = random::string();

    helper(message, "WARNING");
    const auto buffer = Fd::read(temp.path);
    const auto from_file = std::string(buffer.begin(), buffer.end());

    EXPECT_TRUE(from_file.find(message) != std::string::npos);
}

TEST(RUNAI_LOG_FILE, Non_Existing)
{
    for (auto print : { true, false })
    {
        const auto path = random::string() + "/" + random::string();
        const temp::Env RUNAI_STREAMER_LOG_FILE("RUNAI_STREAMER_LOG_FILE", path);
        const temp::Env RUNAI_STREAMER_LOG_TO_STDERR(print ? "RUNAI_STREAMER_LOG_TO_STDERR" : random::string(), true);

        const auto message = random::string();

        const auto from_stderr = helper(message, "WARNING");

        if (print)
        {
            EXPECT_TRUE(from_stderr.find(message) != std::string::npos);
        }
        else
        {
            EXPECT_TRUE(from_stderr.empty());
        }

        // verify file does not exist
        EXPECT_EQ(::access(path.c_str(), F_OK), -1);
    }
}

TEST(Configuration, Independency)
{
    for (auto logtostderr : { true, false })
    {
        for (auto logtofile : { true, false })
        {
            const temp::Env RUNAI_STREAMER_LOG_TO_STDERR("RUNAI_STREAMER_LOG_TO_STDERR", logtostderr);

            temp::Path logfile;
            const temp::Env RUNAI_STREAMER_LOG_FILE(logtofile ? "RUNAI_STREAMER_LOG_FILE" : random::string(), logfile.path);

            const auto message = random::string();

            const auto from_stderr = helper(message, "WARNING");

            if (logtostderr)
            {
                EXPECT_TRUE(from_stderr.find(message) != std::string::npos);
            }
            else
            {
                EXPECT_TRUE(from_stderr.empty());
            }

            if (logtofile)
            {
                const auto buffer = Fd::read(logfile.path);
                const auto from_file = std::string(buffer.begin(), buffer.end());
                EXPECT_TRUE(from_file.find(message) != std::string::npos);
            }
            else
            {
                EXPECT_EQ(::access(logfile.path.c_str(), F_OK), -1);
            }
        }
    }
}

const auto Levels = std::vector<std::string>
{
    "SPAM",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
};

TEST(Levels, File)
{
    for (unsigned configured = 0; configured < Levels.size(); ++configured)
    {
        const temp::Env RUNAI_STREAMER_LOG_LEVEL("RUNAI_STREAMER_LOG_LEVEL", Levels[configured]);

        for (unsigned logged = 0; logged < Levels.size(); ++logged)
        {
            temp::Path temp;
            const temp::Env RUNAI_LOG_FILE("RUNAI_STREAMER_LOG_FILE", temp.path);

            const auto message = random::string();

            const auto from_stderr = helper(message, Levels[logged]);
            EXPECT_TRUE(from_stderr.empty());

            const auto buffer = Fd::read(temp.path);
            const auto from_file = std::string(buffer.begin(), buffer.end());

            if (logged >= configured)
            {
                EXPECT_TRUE(from_file.find(message) != std::string::npos);
            }
            else
            {
                EXPECT_TRUE(from_file.empty());
            }
        }
    }
}

TEST(Levels, All_Levels_Not_Logged_To_Stderr_By_Default)
{
    for (auto env : { true, false })
    {
        const temp::Env RUNAI_STREAMER_LOG_TO_STDERR(env ? "RUNAI_STREAMER_LOG_TO_STDERR" : random::string(), false);

        for (auto level : Levels)
        {
            const auto message      = random::string();
            const auto from_stderr  = helper(message, level);

            ASSERT_TRUE(from_stderr.empty());
        }
    }
}

} // namespace runai::llm::streamer::utils
