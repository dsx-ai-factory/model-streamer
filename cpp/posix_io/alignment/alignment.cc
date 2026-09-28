#include "posix_io/alignment/alignment.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#include "common/exception/exception.h"
#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// A block of 0 or 1 means "no rule to follow". Treating it as 1 keeps every test below true without
// a special case, because everything is a multiple of 1.
size_t usable(size_t block)
{
    return block == 0 ? 1 : block;
}

} // namespace

// RUNAI_STREAMER_DIRECT_BLOCK as a number, or DirectBlockSize when it is not set.
//
// A value that is not a number THROWS, deliberately, and is not caught: std::stoul raises on "abc"
// and on an empty string, and a fatal ASSERT fires on trailing garbage like "4096x".
//
// That is what every other numeric variable already does - RUNAI_STREAMER_CONCURRENCY=abc makes
// runai_file_streamer_start return InvalidParameterError - and a typo must not be answered by quietly serving a
// different value. An operator who mistypes this is told at start, rather than getting a block they
// did not ask for and a load that is slower for reasons nothing reports.
unsigned long configured_direct_block()
{
    return utils::getenv<unsigned long>("RUNAI_STREAMER_DIRECT_BLOCK", DirectBlockSize);
}

size_t usable_direct_block(unsigned long configured)
{
    // A power of two, and at least 512 - the smallest logical block any device reports. Anything else
    // breaks posix_memalign in ScratchPool, and the congruence maths assumes powers of two when it
    // takes a maximum of the two alignments.
    const bool power_of_two = configured != 0 && (configured & (configured - 1)) == 0;

    if (!power_of_two || configured < 512)
    {
        // REJECTED, not replaced. An earlier version fell back to DirectBlockSize with a warning, and
        // that hides the typo it exists to catch: the operator asked for 3000, got 65536, and nothing
        // they would read says so. Failing here surfaces as InvalidParameterError from runai_file_streamer_start,
        // which is what every other malformed variable already does.
        LOG(ERROR) << "RUNAI_STREAMER_DIRECT_BLOCK=" << configured << " is not usable: it must be a"
                   << " power of two and at least 512";
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }

    if (configured != DirectBlockSize)
    {
        LOG(INFO) << "Direct-read block set to " << configured << " by RUNAI_STREAMER_DIRECT_BLOCK."
                  << " Python must read the same value, or destinations stop being congruent and"
                  << " direct reads quietly stop";
    }

    return static_cast<size_t>(configured);
}

size_t direct_block_override()
{
    static const size_t resolved = []() -> size_t
    {
        if (!utils::env_exists("RUNAI_STREAMER_DIRECT_BLOCK"))
        {
            return 0;   // no override; mounts are measured
        }

        const auto block = usable_direct_block(configured_direct_block());

        LOG(INFO) << "RUNAI_STREAMER_DIRECT_BLOCK=" << block << " overrides the per-mount measurement"
                  << " for every mount in this process";
        return block;
    }();

    return resolved;
}

size_t direct_block_size()
{
    // Resolved once. A static local is initialised on first use and is thread-safe since C++11, which
    // matters because several workers can reach this at the same time.
    //
    // That caching is also why the RULES live in usable_direct_block() above rather than in this
    // lambda: the first call in a process wins, so a test cannot drive them through the environment.
    // A pure function can be tested directly, and this wrapper is then too small to be wrong.
    static const size_t resolved = usable_direct_block(configured_direct_block());

    return resolved;
}

size_t block_size(const Limits & limits)
{
    return std::max<size_t>({ usable(limits.offset_alignment), usable(limits.buffer_alignment), 1 });
}

bool is_congruent(size_t file_offset, const void * buffer, size_t block)
{
    const auto address = reinterpret_cast<uintptr_t>(buffer);
    const auto step = usable(block);

    // Written as two remainders rather than as (address - file_offset) % step.
    //
    // For our block sizes the two are the same. Both numbers are unsigned, so the subtraction wraps
    // when the file offset is larger - but 2^64 is itself a multiple of any power of two, so the wrap
    // does not change the remainder. Block sizes are always 512 or 4096.
    //
    // The remainder form is used anyway, because it needs no such argument: it is correct for any
    // block, and a reader does not have to work out whether the wrap is safe.
    return (address % step) == (file_offset % step);
}

AlignedRead expand(size_t offset, size_t bytesize, char * buffer, size_t block)
{
    const auto step = usable(block);

    AlignedRead out;
    out.offset = offset - (offset % step);
    out.head = offset - out.offset;

    // The end moves up to the next block. The length therefore covers the head bytes as well as the
    // wanted bytes, which is why head is added before rounding.
    const size_t wanted_end = out.head + bytesize;
    out.bytesize = wanted_end + ((step - (wanted_end % step)) % step);

    // The address moves down by exactly what the offset moved down by. That is what keeps the file
    // and the memory in step, and it is why this is only correct when they were congruent already.
    out.buffer = buffer - out.head;

    return out;
}

std::string alignment_diagnosis(const Requested & requested, const Limits & limits)
{
    const auto address = reinterpret_cast<uintptr_t>(requested.buffer);
    const auto block = block_size(limits);

    const bool offset_ok = limits.offset_alignment == 0 || requested.offset % limits.offset_alignment == 0;
    const bool length_ok = limits.offset_alignment == 0 || requested.bytesize % limits.offset_alignment == 0;
    const bool buffer_ok = limits.buffer_alignment == 0 || address % limits.buffer_alignment == 0;
    const bool congruent = is_congruent(requested.offset, requested.buffer, block);

    std::stringstream ss;
    ss << "direct read rejected:"
       << " offset=" << requested.offset << " (align " << limits.offset_alignment << (offset_ok ? " ok" : " BAD") << ")"
       << ", buffer=" << static_cast<const void *>(requested.buffer)
       << " (align " << limits.buffer_alignment << (buffer_ok ? " ok" : " BAD") << ")"
       << ", length=" << requested.bytesize
       << " (align " << limits.offset_alignment << (length_ok ? " ok" : " BAD") << ")"
       << ", congruent=" << (congruent ? "yes" : "no");

    // The length is checked against offset_alignment, not against buffer_alignment. The kernel's
    // offset rule covers the length as well. Checking it against the other number gives a message
    // where every field says "ok" and nothing explains the failure.
    //
    // congruence is last because it is the one that matters most: the three above can all say "ok"
    // while the read is still impossible.
    return ss.str();
}

}; // namespace runai::llm::streamer::posix_io
