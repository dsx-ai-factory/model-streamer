/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace runai::llm::streamer::posix_io
{

// Aligned buffers for reading the partial blocks at the edges of a region.
//
// A direct read needs its offset, its length and its address to be block multiples. A region rarely
// starts or ends on a block boundary, so its first and last blocks are read into one of these and the
// wanted bytes are copied out. Everything between is read straight into the caller's memory.
//
// ONE BLOCK per buffer, not one chunk. That is the whole reason this is cheap. An earlier design gave
// every in-flight slot a chunk-sized buffer, which is depth x 8 MiB - about 2 GiB at depth 256 - and
// then had to limit how many reads could bounce at once. A block-sized buffer makes the pool
// depth x 4096, a few megabytes, so there can be one per in-flight read and no limit at all.
//
// NOT THREAD SAFE. One worker owns it, like everything else on this path.
class ScratchPool
{
 public:
    // `count` buffers of `block` bytes each. One per in-flight read means take() can never fail.
    ScratchPool(size_t count, size_t block);

    // A free buffer, aligned to the block size, or nullptr if none is left.
    //
    // nullptr is not expected: the pool is sized to the in-flight window, and a read holds at most one
    // buffer. It is returned rather than asserted so that a caller which somehow asks for more can
    // read buffered instead of dying.
    char * take();

    // Give one back. Must be a buffer this pool handed out.
    void give(char * buffer);

    size_t block() const;
    size_t free_count() const;

 private:
    const size_t _block;

    // One allocation, sliced. Aligned once at the base, so every slice at a block multiple is aligned
    // too - which is what the kernel checks.
    std::unique_ptr<char, void (*)(void *)> _memory;

    std::vector<char *> _free;
};

}; // namespace runai::llm::streamer::posix_io
