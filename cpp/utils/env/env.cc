#include "utils/env/env.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <stdexcept>

#include "utils/logging/logging.h"


namespace runai::llm::streamer::utils
{

namespace
{

// std::stoul accepts a NEGATIVE string and wraps it.
//
// Measured: "-1" gives 18446744073709551615, and it reports the whole string consumed - so the length
// check below cannot tell it from a deliberate huge number. Every caller of these overloads wants a
// count, and no count is negative.
//
// Rejected, not defaulted. A wrong value that is quietly replaced is a typo nobody finds.
// RUNAI_STREAMER_CONCURRENCY=-1 would ask for UINT_MAX threads, and a negative process group size
// divides Azure concurrency down to one. This throws the same exception "abc" already throws, so
// runai_file_streamer_start answers InvalidParameterError and the message names the variable.
unsigned long parse_unsigned(const std::string & variable, const std::string & s)
{
    const auto first = s.find_first_not_of(" \t\n\v\f\r");

    if (first != std::string::npos && s[first] == '-')
    {
        throw std::invalid_argument(variable + "=" + s + " is negative, and must be a count");
    }

    std::string::size_type idx;
    const unsigned long value = std::stoul(s, /* out */ &idx);

    ASSERT(idx == s.size()) << "Failed parsing string '" << s << "' as an integer";

    return value;
}

} // namespace

template <>
bool try_getenv<std::string>(const std::string & variable, /* out */ std::string & s)
{
    char const * const raw = ::getenv(variable.c_str());

    if (raw == nullptr)
    {
        return false;
    }

    /* out */ s = std::string(raw);
    return true;
}

template <>
bool try_getenv<int>(const std::string & variable, /* out */ int & i)
{
    std::string s;
    if (!try_getenv<std::string>(variable, /* out */ s))
    {
        return false;
    }

    std::string::size_type idx;
    i = std::stoi(s, /* out */ &idx);

    ASSERT(idx == s.size()) << "Failed parsing string '" << s << "' as an integer";

    return true;
}

template <>
bool try_getenv<unsigned long>(const std::string & variable, /* out */ unsigned long & i)
{
    std::string s;
    if (!try_getenv<std::string>(variable, /* out */ s))
    {
        return false;
    }

    i = parse_unsigned(variable, s);

    return true;
}

template <>
bool try_getenv<bool>(const std::string & variable, /* out */ bool & b)
{
    int i;
    if (!try_getenv<int>(variable, /* out */ i))
    {
        return false;
    }

    /* out */ b = static_cast<bool>(i);
    return true;
}

bool env_exists(const std::string & variable)
{
    std::string value;
    return try_getenv(variable, /* out */ value);
}

template <>
std::string getenv<std::string>(const std::string & variable)
{
    std::string value;

    if (!try_getenv(variable, /* out */ value))
    {
        LOG(ERROR) << "Failed getting environment variable '" << variable << "'";
        throw std::exception();
    }

    return value;
}

template <>
int getenv<int>(const std::string & variable)
{
    std::string::size_type idx;
    const auto s = getenv<std::string>(variable);
    int i = std::stoi(s, /* out */ &idx);

    ASSERT(idx == s.size()) << "Failed parsing string '" << s << "' as an integer";

    return i;
}

template <>
unsigned long getenv<unsigned long>(const std::string & variable)
{
    return parse_unsigned(variable, getenv<std::string>(variable));
}

template <>
bool getenv<bool>(const std::string & variable)
{
    return static_cast<bool>(getenv<int>(variable));
}

template <>
int getenv<int>(const std::string & variable, int def)
{
    std::string raw;

    if (!try_getenv(variable, /* out */ raw))
    {
        return def;
    }

    std::string::size_type idx;
    int i = std::stoi(raw, /* out */ &idx);

    ASSERT(idx == raw.size()) << "Failed parsing string '" << raw << "' as an integer";

    return i;
}

template <>
unsigned long getenv<unsigned long>(const std::string & variable, unsigned long def)
{
    std::string raw;

    if (!try_getenv(variable, /* out */ raw))
    {
        return def;
    }

    return parse_unsigned(variable, raw);
}

template <>
bool getenv<bool>(const std::string & variable, bool def)
{
    return static_cast<bool>(getenv<int>(variable, static_cast<int>(def)));
}

template <>
std::string getenv<std::string>(const std::string & variable, std::string def)
{
    std::string value;

    if (!try_getenv(variable, /* out */ value))
    {
        return def;
    }

    return value;
}

} // namespace runai::llm::streamer::utils
