/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace runai::llm::streamer::impl
{

// Represents a single contiguous read operation from a specific file - one worker's slice of a
// ContiguousTransfer.
//
// The file is identified by original_file_index alone; the path is deliberately NOT held here. Batches
// already receives the path once per transfer, and nothing ever read this field - keeping it copied a
// std::string into every task, which is one heap allocation per task (thousands per submission) for a
// value nobody used.

struct FileReadTask
{
    FileReadTask(unsigned workload_index,
                 unsigned file_idx,
                 size_t offset,
                 size_t sz,
                 char * dst);

    FileReadTask(FileReadTask&&) = default;
    FileReadTask& operator=(FileReadTask&&) = default;

    unsigned workload_index;

    unsigned original_file_index; // Index of the file within the submitted request
    size_t offset_in_file;        // Starting byte offset within this file
    size_t size;                  // Number of bytes to read
    char * destination;           // Pointer to the destination buffer for this task's data
};

} // namespace runai::llm::streamer::impl
