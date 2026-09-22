
#pragma once

#include <atomic>
#include <vector>
#include "common/device/device.h"
#include "streamer/impl/batch/batch.h"
#include "common/response_code/response_code.h"

namespace runai::llm::streamer::impl
{

// Workload - a validated container of batches that share a backend kind. NOT one batch per file: a file
// whose ranges are not all contiguous yields one ContiguousTransfer per contiguous group, and several of
// those can land in the same workload. It is otherwise
// stateless: filesystem workloads are read synchronously by execute(); object-storage workloads are handed
// to the ObjectStorageWorker pool, which reads the batches via batches() and owns the async scheduling.
struct Workload
{
    // The mount's measured direct-I/O block, or 0 when it could not be measured for this submission.
    //
    // Carried per WORKLOAD, not fixed when the engine is built. A submission whose files were all
    // unreadable yields 0, and the engine then runs provisionally; the next submission measures again
    // and the worker adopts the answer. Without this a long-lived streamer - checkpoint restore, many
    // submissions on one streamer - would keep a provisional block for the life of the process.
    size_t direct_block = 0;

    Workload() = default;
    Workload(Workload &&) = default;
    Workload & operator=(Workload &&) = default;

    // Read a filesystem workload synchronously. NOT for object storage (asserts): those are dispatched to
    // the ObjectStorageWorker pool.
    void execute(std::atomic<bool> & stopped);

    common::ResponseCode add_batch(Batch && batch);

    // Fail all of this workload's batches with the given code, pushing an error response for each
    // of their sub-ranges (whichever task of a sub-range finishes last pushes the single response,
    // so this composes correctly with sub-ranges partly covered by other, dispatched workloads).
    // Used to drain a workload that could not be dispatched (see Streamer::async_request).
    void fail(common::ResponseCode code);

    size_t size() const;

    bool is_object_storage() const;

    // Where this workload's destinations live. Taken from its first batch, and every later one must
    // agree - see verify_batch. Host for an empty workload, which has nowhere to copy to anyway.
    common::Device device() const;

    // The batches, in insertion order. There is no index to look one up by: a file whose ranges are not all
    // contiguous yields several ContiguousTransfers, hence several Batches, and two of them can be assigned
    // to the same workload - so position is NOT a file index. Every consumer iterates, and each batch
    // carries its own file_index, which is what completions and errors are attributed by (see
    // ObjectStorageWorker::report_workload). Responses are routed by Batch pointer (TaskState::batch), not
    // by position; those pointers are taken after the workload has been moved into its Inflight entry and no
    // batch is added afterwards, so vector growth cannot invalidate them.
    // Exposed (rather than befriending the worker) so ObjectStorageWorker can chunk the tasks and route
    // completions; non-const because completion handling mutates the batches (handle_response /
    // handle_error). add_batch stays the validated way to add one, and is always called (populating the
    // workload) before batches() is read - the workload is fully built at dispatch time.
    std::vector<Batch> & batches();
    const std::vector<Batch> & batches() const;

 private:
    common::ResponseCode verify_batch(const Batch & batch);

    std::vector<Batch> _batches;
    bool _is_object_storage = false;
    common::Device _device;
};

}; // namespace runai::llm::streamer::impl
