/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <string>

// this module is heavily based on Google's glog library core implementation

namespace runai::llm::streamer::utils::logging
{

enum class Level
{
    // keep synced with logging.cc
    SPAM = 0,
    DEBUG,
    INFO,
    WARNING, // default
    ERROR,
};

thread_local extern Level __minimum; // minimum level to log.
thread_local extern bool __print; // whether or not to print to stderr.
thread_local extern FILE * __file; // a file to log to; this is optional.

inline bool should_process_log(Level level, bool fatal)
{
    return ((level >= __minimum) && (__print || __file)) || fatal;
}

enum class Color
{
    None = -1,

    // matching ANSI color codes
    BLACK   = 0,
    RED     = 1,
    GREEN   = 2,
    YELLOW  = 3,
    BLUE    = 4,
    MAGENTA = 5,
};

Color color(Level level);

struct Message final
{
    struct Voidify final
    {
        void operator& (std::ostream&) {}
    };

    Message(
        Level level,
        Color color,
        bool fatal,
        bool log_errno,
        const char * level_str,
        const char * function,
        const char * file,
        int line);

    ~Message() noexcept(false);

    std::ostream& stream() { return _stream; }

 private:
    const Level _level;
    const Color _color; // relevant for stderr only
    const int _errno;
    bool _fatal;
    bool _log_errno;

    char _raw[10000];

    struct Stream final : std::ostream
    {
        Stream(char * raw, int length) : std::ostream(nullptr),
            _buffer(raw, length)
        {
            // setting stream buffer
            rdbuf(&_buffer);
        }

        size_t pcount() const
        {
            return _buffer.pcount();
        }

     private:
        struct Buffer final : std::streambuf
        {
            Buffer(char * raw, int length)
            {
                // setting output sequence pointers
                setp(raw, raw + length);
            }

            int_type overflow(int_type ch) override
            {
                // this effectively ignores overflow
                return ch;
            }

            size_t pcount() const
            {
                return pptr() - pbase();
            }
        };

        Buffer _buffer;
    };

    std::string current_time();

    Stream _stream;
};

std::string human_readable_size(size_t bytes, bool raw = false);

#define LOGGING_LEVEL(level)        (::runai::llm::streamer::utils::logging::Level:: level)
#define LOGGING_COLOR(color)        (::runai::llm::streamer::utils::logging::Color:: color)
#define LOGGING_LEVEL_COLOR(level)  (::runai::llm::streamer::utils::logging::color(LOGGING_LEVEL(level)))

#define LOG_BASE(level, color, fatal, log_errno) (!::runai::llm::streamer::utils::logging::should_process_log(LOGGING_LEVEL(level), fatal)) ? \
                                                         (void) 0 : \
                                                         ::runai::llm::streamer::utils::logging::Message::Voidify() & ::runai::llm::streamer::utils::logging::Message(LOGGING_LEVEL(level), (color), fatal, log_errno, # level, __func__, __FILE__, __LINE__).stream()
#define LOG_COLOR(color)                            LOG_BASE(INFO,  LOGGING_COLOR(color),       false, false )
#define LOG(level)                                  LOG_BASE(level, LOGGING_LEVEL_COLOR(level), false, false )

#define LOG_BASE_IF(level, fatal, log_errno, condition) \
  !(condition) ? (void) 0 : LOG_BASE(level, LOGGING_LEVEL_COLOR(level), fatal, log_errno)

#define LOG_IF(level, condition) \
  !(condition) ? (void) 0 : LOG_BASE(level, LOGGING_LEVEL_COLOR(level), false, false)

#define CHECK(condition) LOG_BASE_IF(WARNING, false, false, !(condition))
#define ASSERT(condition) LOG_BASE_IF(ERROR, true, false, !(condition))
#define PASSERT(condition) LOG_BASE_IF(ERROR, true, true, !(condition))

} // namespace runai::llm::streamer::utils::logging
