/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/assigner/file_read_task/file_read_task.h"

namespace runai::llm::streamer::impl
{

// Constructor
FileReadTask::FileReadTask(unsigned workload_index, unsigned file_idx, size_t offset, size_t sz, char * dst) :
    workload_index(workload_index),
    original_file_index(file_idx),
    offset_in_file(offset),
    size(sz),
    destination(dst)
{}

} // namespace runai::llm::streamer::impl
