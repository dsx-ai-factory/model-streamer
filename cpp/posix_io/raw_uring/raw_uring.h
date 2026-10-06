/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace runai::llm::streamer::posix_io
{

// Whether this kernel will register a buffer of this size, asked with the raw syscalls.
//
// Shared by the probe's tests and the engine's. Both need the same answer, and neither may compute it
// from the thing it is checking - reading IoUringProbe or the engine's own count would make the
// assertion vacuous. One copy, so the two suites cannot drift into asking different questions.
//
// The answer is not a kernel version: registration charges RLIMIT_MEMLOCK, so the same kernel answers
// differently under a different limit.
//
// Registers the way IoUringEngine does - a sparse table, then one slot filled. The classic
// IORING_REGISTER_BUFFERS is older than both opcodes, so asking that way would answer a question the
// engine never puts to the kernel.
bool raw_can_register_buffer(size_t bytesize = 4096);

} // namespace runai::llm::streamer::posix_io
