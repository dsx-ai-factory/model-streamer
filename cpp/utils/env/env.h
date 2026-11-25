/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>

namespace runai::llm::streamer::utils
{

template <typename T = std::string>
bool try_getenv(const std::string & variable, /* out */ T & value);

template <>
bool try_getenv(const std::string & variable, /* out */ std::string & value);

template <>
bool try_getenv(const std::string & variable, /* out */ int & value);

template <>
bool try_getenv(const std::string & variable, /* out */ bool & value);

template <>
bool try_getenv(const std::string & variable, /* out */ unsigned long & value);

bool env_exists(const std::string & variable);

void chmod(const std::string & path, mode_t mode);

template <typename T = std::string>
T getenv(const std::string & variable);

template <typename T = std::string>
T getenv(const std::string & variable, T def);

template <>
std::string getenv<std::string>(const std::string & variable);

template <>
int getenv<int>(const std::string & variable);

template <>
unsigned long getenv<unsigned long>(const std::string & variable);

template <>
bool getenv<bool>(const std::string & variable);

template <>
int getenv<int>(const std::string & variable, int def);

template <>
unsigned long getenv<unsigned long>(const std::string & variable, unsigned long def);

template <>
bool getenv<bool>(const std::string & variable, bool def);

template <>
std::string getenv<std::string>(const std::string & variable, std::string def);

// A count from the environment that is safe to divide by and safe to store in T.
//
// getenv<unsigned long> parses the whole 64-bit range, but callers keep these in narrower members,
// usually `unsigned`. A floor applied BEFORE the narrowing does not survive it: std::max(1UL, x)
// promises at least one as an unsigned long, and any non-zero multiple of 2^32 then truncates to 0.
//
// Zero was not a small number in any of the places this replaced. It was an integer division by zero
// in AsyncIoSettings and in the Azure client configuration, and a fatal ASSERT in BackendPools -
// three ways for one mistyped environment variable to kill the host process.
//
// So the order here is: read wide, CAP into T, then floor. Capping first is what makes the floor
// mean anything.
template <typename T>
T getenv_positive(const std::string & variable, T def, T minimum = 1)
{
    static_assert(std::is_unsigned<T>::value, "getenv_positive is for unsigned counts");

    const auto configured = getenv<unsigned long>(variable, static_cast<unsigned long>(def));
    const auto capped = std::min<unsigned long>(configured, std::numeric_limits<T>::max());

    return std::max(static_cast<T>(capped), minimum);
}

} // namespace runai::llm::streamer::utils
