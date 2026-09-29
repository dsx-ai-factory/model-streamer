/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/logging/logging.h"

#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <system_error>
#include <vector>

namespace runai::llm::streamer::utils::logging
{

namespace
{

static Level __init_minimum()
{
    char const * const level = getenv("RUNAI_STREAMER_LOG_LEVEL");

    if (level != nullptr)
    {
        if      (!strcmp(level, "SPAM"))    { return Level::SPAM;    } // NOLINT(readability/braces)
        else if (!strcmp(level, "DEBUG"))   { return Level::DEBUG;   } // NOLINT(readability/braces)
        else if (!strcmp(level, "INFO"))    { return Level::INFO;    } // NOLINT(readability/braces)
        else if (!strcmp(level, "WARNING")) { return Level::WARNING; } // NOLINT(readability/braces)
        else if (!strcmp(level, "ERROR"))   { return Level::ERROR;   } // NOLINT(readability/braces)
    }

    return Level::WARNING;
}

static bool __init_print()
{
    char const * const value = getenv("RUNAI_STREAMER_LOG_TO_STDERR");

    return value == nullptr ? false : strcmp(value, "1") == 0;
}

static FILE * __init_file()
{
    struct File
    {
        File(char const * const path) : _f(fopen(path, "a"))
        {
            if (_f != nullptr)
            {
                // set file to be line bufferred
                setlinebuf(_f);
            }
        }

        ~File() noexcept
        {
            if (_f != nullptr)
            {
                fclose(_f);
            }
        }

        operator FILE * () { return _f; }

     private:
        FILE * _f = nullptr;
    };

    char const * const path = getenv("RUNAI_STREAMER_LOG_FILE");

    if (path == nullptr)
    {
        return nullptr;
    }
    else
    {
        static File __f(path);
        return __f;
    }
}

} // namespace

thread_local Level __minimum = __init_minimum();
thread_local bool __print = __init_print();
thread_local FILE * __file = __init_file();

Color color(Level level)
{
    switch (level)
    {
        case Level::SPAM:       return Color::BLUE;
        case Level::DEBUG:      return Color::MAGENTA;
        case Level::INFO:       return Color::GREEN;
        case Level::WARNING:    return Color::YELLOW;
        case Level::ERROR:      return Color::RED;
        default: break;
    }

    return Color::None;
}

Message::Message(
            Level level,
            Color color,
            bool fatal,
            bool log_errno,
            const char * level_str,
            const char * function,
            const char * file,
            int line) :
    _level(level),
    _color(color),
    _errno(errno),
    _fatal(fatal),
    _log_errno(log_errno),
    _raw(),
    _stream(_raw, sizeof(_raw))
{
    // Log time
    _stream << std::left << "[" << current_time() << "] [" << std::setw(7) << level_str << "] " <<
        "[" << ::getpid() << " " << ::syscall(SYS_gettid) << "] ";

    static constexpr auto MINIMUM_WIDTH = 100l;
    _stream << "[" << file << ":" << std::right << std::setw(3) << line << " @ " <<
        function << std::setw(std::max<std::streamoff>(0, MINIMUM_WIDTH - static_cast<int64_t>(_stream.pcount()))) << "] ";
}

Message::~Message() noexcept(false)
{
    if (_log_errno)
    {
        _stream << ": " << ::strerror(_errno) << " [" << _errno << "]";
    }

    _stream << "\n";

    if (_level >= __minimum)
    {
        const auto __fwrite = [&](auto fileobj, auto colored){
            if (colored && _color != Color::None)
            {
                // Set the color according to code
                fprintf(fileobj, "\033[0;3%dm", (int)_color); // NOLINT(readability/casting)
            }

            fwrite(_raw, _stream.pcount(), 1, fileobj);

            if (colored && _color != Color::None)
            {
                // resets the color to default
                fprintf(fileobj, "\033[m");
            }
        };

        if (__print)
        {
            __fwrite(stderr, true);
        }

        if (__file)
        {
            __fwrite(__file, false);
        }
    }

    // the above code may change errno so we want to restore it
    errno = _errno;

    if (_fatal)
    {
        if (_log_errno)
        {
            throw std::system_error(_errno, std::generic_category());
        }
        else
        {
            throw std::exception();
        }
    }
}

std::string Message::current_time()
{
    std::ostringstream s;

    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm result{};

    // Convert time to local time
    const auto tm = ::localtime_r(&t, &result);
    if (tm == nullptr)
    {
        s << "[ERROR - Invalid time]";
    }
    else
    {
        s << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        s << "." << std::setfill('0') << std::setw(3) << ms.count();
    }
    return s.str();
}

std::string human_readable_size(size_t bytes, bool raw)
{
    const size_t bytes_ = bytes;
    std::vector<std::string> suffix = { "B", "KB", "MB", "GB", "TB", "PB", "EB" };
    int length = suffix.size();

    int i = 0;
    double dblBytes = bytes;

    for (i = 0; (bytes / 1000) > 0 && i < length - 1; i++, bytes /= 1000)
        dblBytes = bytes / 1000.0;

    std::stringstream fmtStr;
    fmtStr << std::fixed << std::setprecision(2) << dblBytes << " " << suffix[i];

    if (raw)
    {
        fmtStr << " (" << bytes_ << " bytes)";
    }

    return fmtStr.str();
}

} // namespace runai::llm::streamer::utils::logging
