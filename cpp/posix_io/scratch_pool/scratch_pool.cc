/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/scratch_pool/scratch_pool.h"

#include <cstdlib>
#include <algorithm>

#include "common/exception/exception.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// posix_memalign accepts an alignment only if it is a power of two AND a multiple of sizeof(void *).
// Measured on this platform: 1, 2 and 4 all return EINVAL, and so does any non-power-of-two such as
// 3000. Only 8 and above, at a power of two, are accepted.
//
// This used to be `block == 0 ? 1 : block`, and 1 is the one value that cannot allocate. block_size()
// hands out 1 as its "no rule to follow" fallback (alignment.cc), so the fallback meant to keep
// things working was exactly the input that made this constructor throw UnknownError - blaming the
// allocation, when nothing was wrong with the request.
//
// Unreachable today: the pool is built only for a direct strategy (async_io_worker.cc), and both
// direct engines report DirectBlockSize. But the fallback exists for the case where an engine
// reports no rule, so it must not be the value that fails.
//
// Rounded UP, never down. _block is also the SIZE of each buffer, not only its alignment, and a
// bounced pass needs a whole block - so a short buffer would be a worse bug than a misaligned one.
size_t allocatable(size_t block)
{
    size_t alignment = sizeof(void *);

    // The `!= 0` stops a shift overflow from looping forever on an absurd block. An allocation that
    // large fails in posix_memalign below, which reports it properly.
    while (alignment < block && alignment != 0)
    {
        alignment <<= 1;
    }

    return alignment == 0 ? sizeof(void *) : alignment;
}

} // namespace

ScratchPool::ScratchPool(size_t count, size_t block) :
    _block(allocatable(block)),
    _memory(nullptr, &std::free)
{
    if (count == 0)
    {
        return;
    }

    // posix_memalign, not new[]: the kernel checks the ADDRESS of a direct read, and the default
    // allocator promises about 16 bytes. One allocation for all of them, so the base is aligned once
    // and every slice at a block multiple is aligned too.
    void * raw = nullptr;
    if (::posix_memalign(&raw, _block, count * _block) != 0)
    {
        LOG(ERROR) << "Cannot allocate " << count << " scratch buffers of " << _block << " bytes";
        throw common::Exception(common::ResponseCode::UnknownError);
    }

    _memory.reset(static_cast<char *>(raw));

    _free.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        _free.push_back(_memory.get() + i * _block);
    }
}

char * ScratchPool::take()
{
    if (_free.empty())
    {
        return nullptr;
    }

    char * buffer = _free.back();
    _free.pop_back();
    return buffer;
}

void ScratchPool::give(char * buffer)
{
    ASSERT(buffer != nullptr) << "returning a null scratch buffer";

    // Taken from the back and returned to the back, so a buffer is reused while it is still warm in
    // the cache rather than after every other one has been used.
    _free.push_back(buffer);
}

size_t ScratchPool::block() const
{
    return _block;
}

size_t ScratchPool::free_count() const
{
    return _free.size();
}

}; // namespace runai::llm::streamer::posix_io
