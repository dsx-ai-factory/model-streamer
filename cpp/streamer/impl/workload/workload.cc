/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "streamer/impl/workload/workload.h"

#include <utility>

#include "common/response_code/response_code.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

size_t Workload::size() const
{
    return _batches.size();
}

std::vector<Batch> & Workload::batches()
{
    return _batches;
}

const std::vector<Batch> & Workload::batches() const
{
    return _batches;
}

common::ResponseCode Workload::add_batch(Batch && batch)
{
    // Appended, not keyed by batch.file_index: one file can contribute several batches to the same workload
    // (one per ContiguousTransfer) as soon as its ranges are not all contiguous.
    if (size() == 0)
    {
        _is_object_storage = batch.is_object_storage();
    }
    else if  (auto res = verify_batch(batch); res != common::ResponseCode::Success)
    {
        return res;
    }

    _batches.push_back(std::move(batch));

    return common::ResponseCode::Success;
}

bool Workload::is_object_storage() const
{
    return _is_object_storage;
}

void Workload::fail(common::ResponseCode code)
{
    for (auto & batch : _batches)
    {
        batch.handle_error(code);
    }
}

common::ResponseCode Workload::verify_batch(const Batch & batch)
{
    if (batch.is_object_storage() != is_object_storage())
    {
         LOG(ERROR) << "Workload contains paths of different storage backends";

        return common::ResponseCode::InvalidParameterError;
    }

    return common::ResponseCode::Success;
}

void Workload::execute(std::atomic<bool> & stopped)
{
    if (size() == 0)
    {
        return;
    }

    // Object-storage workloads are read asynchronously by the ObjectStorageWorker pool, not here.
    ASSERT(!is_object_storage()) << "object-storage workload must be executed by ObjectStorageWorker";

    for (auto & batch : _batches)
    {
        batch.execute(stopped);
        LOG(DEBUG) << "Finished batch " << batch;
    }
}

}; // namespace runai::llm::streamer::impl
