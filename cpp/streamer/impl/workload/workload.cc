/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        _device = batch.device;
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

common::Device Workload::device() const
{
    return _device;
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

    // A worker reads the destination off whichever batch a completed chunk belongs to, so a mixed
    // workload would copy to the wrong device rather than fail. One submission names one device, so
    // it cannot happen today - which was also true of the backend mix above, until it was not.
    if (batch.device != device())
    {
        LOG(ERROR) << "Workload contains batches for different devices: " << device()
                   << " and " << batch.device;

        return common::ResponseCode::InvalidParameterError;
    }

    return common::ResponseCode::Success;
}

void Workload::execute(std::atomic<bool> & stopped, const DeviceStaging * staging)
{
    if (size() == 0)
    {
        return;
    }

    // Object-storage workloads are read asynchronously by the ObjectStorageWorker pool, not here.
    ASSERT(!is_object_storage()) << "object-storage workload must be executed by ObjectStorageWorker";

    for (auto & batch : _batches)
    {
        batch.execute(stopped, staging);
        LOG(DEBUG) << "Finished batch " << batch;
    }
}

}; // namespace runai::llm::streamer::impl
