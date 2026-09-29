/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/thread/thread.h"

#include <utility>

#include "utils/logging/logging.h"


namespace runai::llm::streamer::utils
{

Thread::Thread(const std::function<void()> & f) :
    _t([=](){
        try
        {
            f();
        }
        catch (const std::system_error & e)
        {
            LOG(ERROR) << "Caught system error in entrypoint: " << typeid(e).name() << " " << e.what();
        }
        catch (const std::exception & e)
        {
            LOG(ERROR) << "Caught exception in entrypoint: " << typeid(e).name() << " " << e.what();
        }
        catch (...)
        {
            LOG(ERROR) << "Caught unknown exception in entrypoint";
        }
    })
{}

Thread::~Thread()
{
    try
    {
        join();
    }
    catch (...) {}
}

Thread & Thread::operator=(Thread && other)
{
    join();

    _t = std::move(other._t);

    return *this;
}

void Thread::join()
{
    if (_t.joinable())
    {
        _t.join();
    }
}

bool Thread::joinable()
{
    return _t.joinable();
}

} // namespace runai::llm::streamer::utils
