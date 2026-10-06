/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "utils/random/random.h"

namespace runai::llm::streamer::utils::temp
{

// Sets an environment variable for the lifetime of the object and restores its previous value (or
// leaves it unset) on destruction.
//
// It OVERWRITES a variable that is already set. An earlier version did not, and a test whose variable
// was already in the environment then ran under a value it did not choose. setenv reports success
// when it declines to overwrite, so nothing failed and nothing was logged - the test simply checked
// something else. That is reachable whenever a variable is exported in the shell or passed with
// bazel's --test_env.
//
// THE CONTRACT: on one variable, these must be destroyed in STACK ORDER, on one thread.
//
// Nesting is fine, and is used deliberately. This pair appears in several tests:
//
//     utils::temp::UnsetEnv unset(SomeVar);      // clear whatever the environment had
//     utils::temp::Env      env(SomeVar, "2.0"); // then set exactly what this test wants
//
// It works because each object saves the state it finds: Env sees "unset", so ~Env unsets, and
// ~UnsetEnv then puts the original back. That is what makes a test independent of a variable
// exported in the shell or passed with bazel's --test_env.
//
// What breaks it is destruction OUT of stack order, which needs a heap-allocated instance or a second
// thread. Then the first destructor restores the value it saved while the other object is still
// active, and the second restores something stale or unsets a variable it never set. The variable
// ends up holding whatever the later destructor happened to see - silently, in a helper whose entire
// job is to leave the environment as it found it.
//
// NOT enforced, and neither half of the obvious enforcement works:
//
//   a mutex   would be worse than nothing. setenv and getenv are not thread-safe - setenv may
//             reallocate `environ` while another thread is inside getenv, which is the race
//             PRE_SETENV_DELAY papers over in the .cc. Locking our own writers leaves every reader
//             thread exposed while making the helper look safe across threads, which it is not.
//
//   a check   can only run in the destructor, since nesting is legal and only the ORDER is wrong.
//             ASSERT throws (logging.cc), and throwing from a destructor during unwinding calls
//             std::terminate; LOG(ERROR) would be silent, because no log sink is enabled under bazel.
//
// So: single-threaded, stack-scoped. Do not put one of these on the heap.
struct Env
{
    Env(const std::string & value = random::string());

    Env(
        const std::string & name,
        char const * const value);

    Env(
        const std::string & name,
        const std::string & value);

    Env(
        const std::string & name,
        int value);

    Env(
        const std::string & name,
        unsigned long value);

    Env(
        const std::string & name,
        bool value);

    Env(
        const std::string & name,
        float value);

    ~Env();

    Env(Env &&) = delete;
    Env(const Env &) = delete;

    Env & operator=(Env &&) = delete;
    Env & operator=(const Env &) = delete;

    std::string name;
    std::string value;

    // What was there before, so the destructor can put it back. Same pair as UnsetEnv holds, and for
    // the same reason.
    std::string previous_value;
    bool had_value = false;
};

// Unsets an environment variable for the lifetime of the object and restores its
// previous value (or leaves it unset) on destruction.
//
// Same save-and-restore structure as Env, so the same contract applies, and it applies ACROSS the
// two types: an UnsetEnv and an Env on one variable must unwind in stack order just as two of either
// must. Nesting them is the intended way to pin a variable regardless of the ambient environment -
// see Env above.
struct UnsetEnv
{
    explicit UnsetEnv(const std::string & name);
    ~UnsetEnv();

    UnsetEnv(UnsetEnv &&) = delete;
    UnsetEnv(const UnsetEnv &) = delete;

    UnsetEnv & operator=(UnsetEnv &&) = delete;
    UnsetEnv & operator=(const UnsetEnv &) = delete;

    std::string name;
    std::string previous_value;
    bool had_value = false;
};

} // namespace runai::llm::streamer::utils::temp
