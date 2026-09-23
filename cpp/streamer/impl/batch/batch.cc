#include "streamer/impl/batch/batch.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <set>
#include <map>
#include <vector>

#include "utils/logging/logging.h"

#include "common/exception/exception.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "common/range/range.h"
#include "streamer/impl/reader/reader.h"
#include "streamer/impl/file/file.h"
#include "streamer/impl/s3/s3.h"
#include "utils/scope_guard/scope_guard.h"
#include "utils/semaphore/semaphore.h"

namespace runai::llm::streamer::impl
{

Batch::Batch(SubmissionId submission_id, unsigned workload_index, unsigned file_index, const std::string & path, const common::s3::S3ClientWrapper::Params & params, const Tasks && tasks, std::shared_ptr<common::Responder> responder, std::shared_ptr<const Config> config, size_t chunk_bytesize, common::Device device) :
    submission_id(submission_id),
    device(device),
    workload_index(workload_index),
    file_index(file_index),
    path(path),
    object_storage_params(params),
    tasks(tasks),
    chunks(split_into_chunks(this->tasks, chunk_bytesize)),
    range(tasks),
    responder(responder),
    config(config)
{
    LOG(DEBUG) << "Batch " << path << " range " << range << " ; " << this->tasks.size() << " tasks in "
               << chunks.size() << " chunks to " << device;
}

size_t Batch::total_bytes() const
{
    return range.size;
}

size_t Batch::end_offset() const
{
    return range.end;
}

void Batch::execute(std::atomic<bool> & stopped, const DeviceStaging * staging)
{
    LOG(DEBUG) << "Start reading from file " << path;

    auto response_code = common::ResponseCode::Success;
    try
    {
        ASSERT(!is_object_storage()) << "Unsupported reader mode for object storage backends";

        _reader = std::make_unique<File>(path, *config);

        if (device.is_host())
        {
            read(*config, stopped);
        }
        else if (staging != nullptr && staging->valid())
        {
            read_to_device(*staging, stopped);
        }
        else
        {
            // The caller routed a device batch to a reader that cannot stage it. pread into device
            // memory is a segmentation fault, not an error, so this refuses rather than tries.
            LOG(ERROR) << "No staging for a batch destined for " << device;
            throw common::Exception(common::ResponseCode::UnsupportedDeviceType);
        }
    }
    catch(const common::Exception & e)
    {
        response_code = e.error();
    }
    catch (...)
    {
        response_code = common::ResponseCode::UnknownError;
    }

    // in case of an error all of the batch's unfinished tasks are failed with the same error code
    // in case of success the finished tasks were already notified
    handle_error(response_code);
}

void Batch::handle_error(common::ResponseCode response_code)
{
    // in case of an error all of the batch's unfinished tasks are failed with the same error code
    // in case of success the finished tasks were already notified

    if (response_code != common::ResponseCode::Success)
    {
        if (response_code != common::ResponseCode::FinishedError)
        {
            LOG(ERROR) << "Failed to read from file " << path << " ; error: " << response_code;
        }
        else
        {
            LOG(SPAM) << "Finished reading from file " << path;
        }

        // Note:
        // At this point no more tasks are expected to finish, since synchronous reading has ended and for asyncronous reading the thread stopped waiting for finished tasks
        for (auto & task : tasks)
        {
            if (task.finished_request(response_code))
            {
                common::Response response(submission_id, file_index, task.request->index, task.request->ret());
                responder->push(std::move(response), task.request->bytesize);
            }
        }
    }
}

// read the entire range and send notifications for each sub range
void Batch::read(const Config & config, std::atomic<bool> & stopped)
{
    if (tasks.empty())
    {
        LOG(DEBUG) << "Empty batch";
        return;
    }

    auto file_offset = range.start;
    // A batch covers one ContiguousTransfer, whose ranges are adjacent in both the file and the
    // destination, so the whole batch writes into one contiguous buffer starting at the first task's
    // destination. Both cursors advance in lockstep below.
    char * buffer = tasks[0].destination();

    size_t num_chunks = range.size / config.fs_sync_read_block_bytesize;

    // seek just once because tasks are consecutive within the range
    _reader->seek(file_offset);

    // read task's range in chunks
    size_t i = 0;
    for (; i < num_chunks && !stopped; ++i)
    {
        _reader->read(config.fs_sync_read_block_bytesize, buffer);

        file_offset += config.fs_sync_read_block_bytesize;
        buffer += config.fs_sync_read_block_bytesize;

        finished_until(file_offset, common::ResponseCode::Success);
    }

    if (file_offset < range.end && !stopped)
    {
        num_chunks++;
        i = 1;
        _reader->read(range.end - file_offset, buffer);
        finished_until(range.end, common::ResponseCode::Success);
    }

    // An empty batch range (range.start == range.end) enters neither branch above, so without this its
    // tasks would never be notified and the submission would wait for responses that never come. That
    // is reachable whenever a transfer carries only zero sized ranges - which the streamer accepts and
    // must still answer, one response per range whatever its size.
    // finished_until only ever advances _unfinished, so this is a no-op in every other case.
    if (!stopped)
    {
        finished_until(range.end, common::ResponseCode::Success);
    }

    LOG(DEBUG) << "Finished reading " << i << "/" << num_chunks << " chunks from file " << path << (stopped ? " - terminated" : " successfully");

    if (stopped)
    {
        throw common::Exception(common::ResponseCode::FinishedError);
    }
}

void Batch::read_to_device(const DeviceStaging & staging, std::atomic<bool> & stopped)
{
    if (tasks.empty())
    {
        LOG(DEBUG) << "Empty batch";
        return;
    }

    // FROM THE POOL, not from the config. A read must never be larger than the buffer it lands in,
    // and two values kept in step by hand is how that stops being true - it overflowed the buffer the
    // first time they disagreed.
    const size_t block = staging.pool->buffer_bytesize();

    // As in read(): a batch covers one ContiguousTransfer, so the whole range lands in one contiguous
    // destination starting at the first task's. That destination is DEVICE memory here, so nothing
    // below may write to it - only the copy does.
    char * const destination = tasks[0].destination();

    _reader->seek(range.start);

    // Written by the issuer's completions, read by this thread. `landed` counts copies that have
    // retired, IN ORDER: one stream per device is FIFO and this thread submits in file order, so the
    // n-th completion is the n-th block.
    std::atomic<unsigned> landed{ 0 };
    std::atomic<int> failure{ static_cast<int>(common::ResponseCode::Success) };
    utils::Semaphore retired(0);
    unsigned submitted = 0;

    size_t offset = range.start;

    {
        // Every completion holds a reference to the three above, so this thread may not leave the
        // block until they have all fired - including when a read throws. The guard is what makes
        // that true on every path, and its scope is what puts the wait before the final answer.
        utils::ScopeGuard drain([&]()
            {
                for (unsigned i = 0; i < submitted; ++i)
                {
                    retired.wait();
                }
            });

        while (offset < range.end && !stopped)
        {
            // Blocking, deliberately: this thread has nothing else to do, and a full pool IS the
            // backpressure that stops it reading faster than the device can absorb.
            StagingBuffer buffer;
            const auto code = staging.pool->acquire(buffer);
            if (code != common::ResponseCode::Success)
            {
                throw common::Exception(code);
            }

            if (!buffer.valid())
            {
                // The pool was stopped, which only happens on teardown. The rest of the range will
                // never be read, so it must not be answered as read - the same mistake as counting a
                // failed copy. FinishedError is what the stopped path below reports.
                throw common::Exception(common::ResponseCode::FinishedError);
            }

            const size_t bytesize = std::min(block, range.end - offset);

            // Until the issuer has it, this buffer is ours to give back: a read that throws leaves
            // nobody else to do it, and the pool is three deep - the third failed batch would leave
            // the fourth waiting for a buffer forever. From submit() on, the issuer returns it on
            // every path, so the guard is cancelled rather than released twice.
            utils::ScopeGuard give_back([&]() { staging.pool->release(buffer); });

            _reader->read(bytesize, buffer.data);

            staging.issuer->submit(device.id, staging.pool, buffer, bytesize,
                                   destination + (offset - range.start),
                                   [&landed, &failure, &retired](common::ResponseCode ret)
                                   {
                                       if (ret != common::ResponseCode::Success)
                                       {
                                           int expected = static_cast<int>(common::ResponseCode::Success);
                                           failure.compare_exchange_strong(expected, static_cast<int>(ret));
                                       }
                                       else if (failure.load() == static_cast<int>(common::ResponseCode::Success))
                                       {
                                           // ONLY for a copy that landed, and only while none before it
                                           // failed. This counter is what the reader answers ranges
                                           // from, so counting a failed copy would report bytes that
                                           // never reached the device as successfully read.
                                           //
                                           // Checking `failure` is enough because completions arrive in
                                           // order: an earlier failure has already been recorded.
                                           landed.fetch_add(1, std::memory_order_release);
                                       }

                                       retired.post();
                                   });

            give_back.cancel();
            ++submitted;
            offset += bytesize;

            // Answer what has already reached the device, without waiting. Called only from this thread,
            // so _unfinished needs no lock - the completions touch none of the batch.
            // The clamp is load-bearing, not defensive: every landed block is a full `block` except
            // possibly the last, so once that one lands the sum overshoots range.end - and answering
            // the whole range is exactly right at that point.
            finished_until(std::min(range.start + static_cast<size_t>(landed.load(std::memory_order_acquire)) * block,
                                    range.end));
        }
    }   // every copy has retired here, so nothing is still writing to the destination

    const auto worst = static_cast<common::ResponseCode>(failure.load());
    if (worst != common::ResponseCode::Success)
    {
        throw common::Exception(worst);
    }

    if (!stopped)
    {
        // Also answers a range of size zero, which reads nothing and still owes a response.
        finished_until(range.end, common::ResponseCode::Success);
    }

    LOG(DEBUG) << "Finished " << submitted << " blocks to " << device << " from file " << path
               << (stopped ? " - terminated" : " successfully");

    if (stopped)
    {
        throw common::Exception(common::ResponseCode::FinishedError);
    }
}

void Batch::handle_response(const common::backend_api::Response & response, const Task * task_ptr)
{
    // Aborting if a single task failed, we should replace this by a retry mechanism
    if (response.ret != common::ResponseCode::Success)
    {
        LOG(ERROR) << "Error " << response.ret << " while waiting for responses";
        throw common::Exception(response.ret);
    }

    ASSERT(task_ptr != nullptr) << "Received response from a null task";

    handle_task_response(response.ret, task_ptr);
}

void Batch::handle_task_response(const common::ResponseCode response_code, const Task * task_ptr)
{
    // Aborting if a single task failed, we should replace this by a retry mechanism

    ASSERT(task_ptr->request->file_index == file_index) << "Received response from a different file " << task_ptr->request->file_index << " expected " << file_index;

    LOG(SPAM) << "Received object storage response: File index " << file_index << " request index " << task_ptr->request->index << " ret " << response_code;
    if (task_ptr->finished_request(response_code))
    {
        common::Response request_response(submission_id, file_index, task_ptr->request->index, task_ptr->request->ret());
        responder->push(std::move(request_response), task_ptr->request->bytesize);
    }
}

// notify unfinished tasks up to but not including offset end
void Batch::finished_until(size_t file_offset, common::ResponseCode ret /*= common::ResponseCode::Success */)
{
    unsigned i = _unfinished;
    for (; i < tasks.size(); ++i)
    {
        if (file_offset < tasks[i].info.end)
        {
            break;
        }
        if (tasks[i].finished_request(ret))
        {
            const auto & r = tasks[i].request;
            common::Response response(submission_id, file_index, r->index, r->ret());
            LOG(SPAM) << "Sending response " << response;
            responder->push(std::move(response), tasks[i].request->bytesize);
        }
    }
    _unfinished = i;
}

unsigned Batch::finished_until() const
{
    return _unfinished;
}

bool Batch::is_object_storage() const
{
    return object_storage_params.valid();
}

std::ostream & operator<<(std::ostream & os, const Batch & r)
{
    return os << r.path << " range " << r.range << " ; " << r.tasks.size() << " tasks";
}

Batch::Range::Range(size_t start_offset, size_t end_offset) :
    common::Range(start_offset, end_offset - start_offset),
    end(end_offset)
{
    if (end < start)
    {
        LOG(ERROR) << "Invalid range " << start << " - " << end;
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }
}

Batch::Range::Range(const Tasks & tasks) :
    Range(calculate_start(tasks), calculate_end(tasks))
{}

size_t Batch::Range::calculate_start(const Tasks & tasks)
{
    if (tasks.empty())
    {
        return 0;
    }
    return tasks[0].info.offset;
}

size_t Batch::Range::calculate_end(const Tasks & tasks)
{
    if (tasks.empty())
    {
        return 0;
    }

    return tasks[tasks.size() - 1].info.end;
}

std::ostream & operator<<(std::ostream & os, const Batch::Range & r)
{
    return os << "Range from " << r.start << " to " << r.end;
}


}; // namespace runai::llm::streamer::impl
