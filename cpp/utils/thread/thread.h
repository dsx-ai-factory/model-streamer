/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <thread>
#include <string>

namespace runai::llm::streamer::utils
{

struct Thread
{
    Thread() = default;
    Thread(const std::function<void()> & f); // an exception must not be thrown from inside 'f'
    ~Thread(); // joins the thread

    Thread(Thread &&) = default;
    Thread & operator=(Thread &&);

    // join the thread
    // this may be called many times (as opposed to std::thread::join())
    void join();

    bool joinable();

 private:
    std::thread _t;
};

} // namespace runai::llm::streamer::utils
