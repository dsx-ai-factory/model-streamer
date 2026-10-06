/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/scope_guard/scope_guard.h"

#include <utility>

namespace runai::llm::streamer::utils
{

ScopeGuard::ScopeGuard(std::function<void()> && lambda) :
    _lambda(lambda)
{}

ScopeGuard::ScopeGuard(ScopeGuard && other) :
    _lambda()
{
    *this = std::move(other);
}

ScopeGuard& ScopeGuard::operator=(ScopeGuard && other)
{
    // best effort lambda call
    _release();

    _lambda = std::move(other._lambda);
    other._lambda = nullptr;

    return *this;
}

ScopeGuard::~ScopeGuard()
{
    try
    {
        _release();
    }
    catch (...)
    {
    }
}

void ScopeGuard::_release()
{
    try
    {
        if (_lambda)
        {
            _lambda();
        }
    }
    catch (...)
    {}
}

void ScopeGuard::cancel()
{
    _lambda = nullptr;
}

} // namespace runai::llm::streamer::utils
