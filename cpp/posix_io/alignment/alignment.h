/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>

#include "posix_io/io_engine/io_engine.h"

namespace runai::llm::streamer::posix_io
{

// What a direct read was asked to do. Enough to say why the kernel refused it.
struct Requested
{
    size_t offset = 0;
    size_t bytesize = 0;
    const char * buffer = nullptr;
};

// The one number that matters for a direct read.
//
// The kernel has separate limits for the memory address and for the file offset, and they can differ.
// The fully general rule is harder than it is worth. We use the larger of the two for everything, so
// one number covers all three checks. Over-aligning is always correct. It only wastes a little.
size_t block_size(const Limits & limits);

// The block size we assume a direct read needs, and the default for direct_block_size() below.
//
// 65536, assumed rather than measured. statx reports the real numbers through STATX_DIOALIGN, which
// arrived in kernel 6.1 while our floor is 5.15 - and it is missing from the build headers here even
// on a 6.8 kernel.
//
// Raised from 4096. That covered every block size in use at the time (512 and 4096) and nothing more.
// 16K-logical-block NVMe is arriving, and on such a device an O_DIRECT open SUCCEEDS and every read
// then fails with EINVAL - so the assumption fails on exactly the storage a fast loader is bought for.
// 64 KB covers 512, 4K, 16K, 32K and 64K at once.
//
// Over-aligning cannot fail, and here it is close to free. The streamer places its own destination
// buffers (see the note on is_congruent below), so congruence is arranged rather than hoped for, and
// the cost is bounded by the head and tail blocks of each chunk - under 2% of an 8 MB chunk at this
// size. See design_measured_alignment.md.
//
// Read by two places that must agree:
//
//   the engines   IoUringEngine and LibaioEngine both report it as their Limits alignment
//   routing       FsAsyncRouter::groups decides congruence before any engine exists, so it cannot
//                 ask an engine and needs the number early
//
// They share the constant rather than each writing 4096, because a disagreement would be invisible.
// Routing and the worker would differ about which files can be read directly, nothing would fail, and
// the reads would quietly take a path nobody intended.
//
// If an engine ever measures the real value with statx, it will report something equal or NARROWER
// than this, and routing stays correct: it would refuse a few files that could have been read
// directly, sending them to the synchronous reader. Slower for those files, never wrong.
constexpr size_t DirectBlockSize = 65536;

// The block actually used, which is DirectBlockSize unless RUNAI_STREAMER_DIRECT_BLOCK overrides it.
//
// A knob rather than a constant for two reasons. It lets a benchmark compare block sizes without a
// rebuild, and it lets an operator drop back if a mount turns out to prefer a smaller one.
//
// Read ONCE and cached: it is consulted per read, and an environment lookup on that path would be
// visible. It also must not change under a running streamer - routing decided congruence against one
// value and the worker must plan passes against the same one.
//
// Rejected values fall back to DirectBlockSize with a warning rather than failing: a mistyped block
// should not stop a model loading, and every rejected value would break posix_memalign in ScratchPool
// or the congruence maths.
//
// PYTHON READS THE SAME VARIABLE (requests_iterator.py, DIRECT_IO_BLOCK). The two must agree: Python
// pads the ring to make destinations congruent, and if it padded to a different block than this one
// tests against, congruence would fail and direct reads would quietly stop happening.
size_t direct_block_size();

// The largest block a mount probe will ever return, and therefore the value a worker runs at while it
// has no measurement.
//
// Provisional-at-the-maximum is what makes a later measurement safe to adopt: it can only be smaller,
// so buffers sized for this one are big enough for it and nothing has to be reallocated.
constexpr size_t MaxProbeBlock = 65536;

// The rules direct_block_size() applies to a configured value, separated so they can be tested.
//
// Returns `configured` when it is usable - a power of two, at least 512 - and DirectBlockSize
// otherwise, with a warning. Exposed because direct_block_size() caches its answer for the life of
// the process, so a test cannot reach these rules through the environment.
size_t usable_direct_block(unsigned long configured);

// RUNAI_STREAMER_DIRECT_BLOCK as a number, or DirectBlockSize when it is absent or is not a number.
//
// Separate from usable_direct_block(), which judges a value that already parsed. Reading THROWS on a
// non-numeric value - std::stoul on "abc" or "", a fatal ASSERT on "4096x" - and that exception used
// to escape the mount probe and end as a ZeroDivisionError in Python. Exposed so the fallback can be
// tested; direct_block_size() and direct_block_override() both cache, so neither can be.
unsigned long configured_direct_block();

// The operator's explicit block, or 0 when RUNAI_STREAMER_DIRECT_BLOCK is unset.
//
// An OVERRIDE, not a default: when set it replaces the per-mount measurement entirely. That is what
// makes it an escape hatch - a way to pin the block when a measurement is wrong or when comparing
// values - and it is how the benchmark holds one arm at 4096 while another measures.
//
// Distinct from direct_block_size(), which answers "what do we use when nothing was measured" and so
// cannot tell an explicit 65536 from the default.
size_t direct_block_override();

// Can ANY part of this read be done directly?
//
// This is the rule that decides everything, and it is not "is the address aligned". A direct read
// puts file byte F+k at address B+k: the two move together. So it needs one k where both
// (F+k) and (B+k) are multiples of the block. That is possible only when:
//
//     (B - F) % block == 0
//
// If it fails, there is no such k anywhere in the range - not at the start, not in the middle. The
// range has no aligned part at all, so 100% of it must be copied. If it holds, only the first and
// last blocks need care.
//
// For an address and a file offset chosen without thought, this holds about once in `block` tries.
// That is why the streamer places its own buffers on purpose - see o_direct_alignment.md section 7.
bool is_congruent(size_t file_offset, const void * buffer, size_t block);

// A direct read that covers [offset, offset + bytesize), with everything aligned.
//
// The start moves DOWN to the block below, and the length moves UP. So the read covers more of the
// file than was asked for. The extra bytes are real file data, and they have to go somewhere:
//
//     head bytes  land at   buffer - head
//     tail bytes  land after the wanted data
//
// THE CALLER MUST OWN THAT SPACE. This function only does arithmetic. It cannot know whether there is
// room before `buffer`, and writing there is what an overrun looks like. The streamer's own pool
// reserves that space on purpose; memory from a caller does not have it, which is why a read into
// caller memory does not take this path.
//
// Only meaningful when is_congruent() holds. Without congruence there is no aligned read that lands
// the wanted bytes in the right place, whatever the rounding.
struct AlignedRead
{
    size_t offset = 0;     // rounded down to a block
    size_t bytesize = 0;   // rounded up to a block
    char * buffer = nullptr;   // moved down by the same amount as the offset
    size_t head = 0;       // unwanted bytes before the data, i.e. offset - original offset
};

AlignedRead expand(size_t offset, size_t bytesize, char * buffer, size_t block);

// Why a direct read was refused, naming the rule that failed.
//
// The kernel says only EINVAL, which is the same answer for four different mistakes. Reading that in
// a log tells nobody anything. This names each rule and says which one broke:
//
//   direct read rejected: offset=8000 (align 512 BAD), buffer=0x... (align 512 ok),
//                         length=100 (align 512 BAD), congruent=no
std::string alignment_diagnosis(const Requested & requested, const Limits & limits);

}; // namespace runai::llm::streamer::posix_io
