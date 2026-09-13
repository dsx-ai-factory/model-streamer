#include "streamer/streamer.h"

#include <unistd.h>

#include <memory>
#include <string>
#include <vector>

#include "common/exception/exception.h"
#include "common/response_code/response_code.h"
#include "common/s3_credentials/s3_credentials.h"
#include "streamer/impl/streamer/streamer.h"
#include "posix_io/alignment/alignment.h"

// The C API is defined at file scope, where streamer.h declares it. The helper below is the only part
// that is ours to name, so it keeps the namespace.
namespace runai::llm::streamer
{
namespace
{

// Marshal the C request arrays and submit, forwarding out_submission_id. Credentials are NOT passed here:
// they are streamer-scoped, set once via runai_set_credentials.
//
// The C arrays are flat and grouped by file in the order of paths; they are transposed here into one
// FileRanges per file, each holding its ranges as (offset, size, dst) triples. Validation is ordered so
// that no array is dereferenced before it has been checked - in particular paths[i] is checked before
// being used to construct a std::string.
int submit_request(impl::Streamer * s,
                   NvFileStreamerSubmissionId * out_submission_id,
                   unsigned num_files,
                   const char ** paths, unsigned * num_ranges,
                   size_t * range_offsets, size_t * range_sizes, void ** range_dsts,
                   NvFileStreamerDevice device)
{
    // Rejected before anything is committed, so a submission this build cannot serve owes no responses.
    if (device.type != NV_FILE_STREAMER_DEVICE_CPU)
    {
        return static_cast<int>(common::ResponseCode::UnsupportedDeviceType);
    }

    if (num_files > 0 && (paths == nullptr || num_ranges == nullptr))
    {
        return static_cast<int>(common::ResponseCode::InvalidParameterError);
    }

    size_t total_ranges = 0;
    for (unsigned i = 0; i < num_files; ++i)
    {
        total_ranges += num_ranges[i];
    }

    // the range arrays are dereferenced only if the submission actually carries ranges
    if (total_ranges > 0 && (range_offsets == nullptr || range_sizes == nullptr || range_dsts == nullptr))
    {
        return static_cast<int>(common::ResponseCode::InvalidParameterError);
    }

    std::vector<impl::FileRanges> request(num_files);

    size_t base = 0;
    for (unsigned i = 0; i < num_files; ++i)
    {
        if (paths[i] == nullptr)
        {
            return static_cast<int>(common::ResponseCode::InvalidParameterError);
        }
        request[i].path = paths[i];

        const unsigned n = num_ranges[i];
        request[i].ranges.reserve(n);
        for (unsigned j = 0; j < n; ++j)
        {
            request[i].ranges.push_back(impl::ReadRange{ range_offsets[base + j], range_sizes[base + j], range_dsts[base + j] });
        }
        base += n;
    }

    return static_cast<int>(s->async_request(request, out_submission_id));
}

} // namespace
}  // namespace runai::llm::streamer

using namespace runai::llm::streamer;

// Library for reading files concurrently into host memory buffers
// A single submission (runai_request) may cover many files, and many submissions may be in flight at
// once - each response carries the id of the submission it belongs to
// NOT THREAD SAFE - caller must not send requests and responses in parallel

// Creates a streamer object; see streamer.h for the configuration environment variables.

int runai_start(void ** streamer)
{
    // verify configuration
    std::unique_ptr<impl::Config> config;
    try
    {
        config = std::make_unique<impl::Config>();
    }
    catch(...)
    {
        return static_cast<int>(common::ResponseCode::InvalidParameterError);
    }

    try
    {
        *streamer = new impl::Streamer(*config);
    }
    catch(...)
    {
        return static_cast<int>(common::ResponseCode::UnknownError);
    }
    return static_cast<int>(common::ResponseCode::Success);
}

// destroys streamer object

void runai_end(void * streamer)
{
    try
    {
        auto s = static_cast<impl::Streamer *>(streamer);
        if (s != nullptr)
        {
            delete s;
        }
    }
    catch(...)
    {
    }
}


// Set the streamer's object-storage credentials as a general key/value dictionary (canonical config-param
// keys; see common::s3::Credentials). Set-once and thread-safe: the same credentials may be set repeatedly
// (Success); a different set after the first returns CredentialsAlreadySet. Credentials are streamer-scoped -
// the read/list entry points use whatever was set here.
int runai_set_credentials(
    void * streamer,
    const char ** param_keys,
    const char ** param_values,
    unsigned num_params)
{
    try
    {
        auto s = static_cast<impl::Streamer *>(streamer);
        if (s == nullptr)
        {
            return static_cast<int>(common::ResponseCode::InvalidParameterError);
        }

        return static_cast<int>(s->set_credentials(common::s3::Credentials(param_keys, param_values, num_params)));
    }
    catch (const common::Exception & e)
    {
        // report the specific code (see runai_request for why this matters)
        return static_cast<int>(e.error());
    }
    catch(...)
    {
    }
    return static_cast<int>(common::ResponseCode::UnknownError);
}

int runai_set_fs_strategy(
    void * streamer,
    const char * candidates)
{
    try
    {
        auto s = static_cast<impl::Streamer *>(streamer);
        if (s == nullptr || candidates == nullptr)
        {
            return static_cast<int>(common::ResponseCode::InvalidParameterError);
        }

        return static_cast<int>(s->set_fs_strategy(candidates));
    }
    catch (const common::Exception & e)
    {
        // report the specific code (see runai_request for why this matters)
        return static_cast<int>(e.error());
    }
    catch(...)
    {
    }
    return static_cast<int>(common::ResponseCode::UnknownError);
}

int runai_request(
    void * streamer,
    NvFileStreamerSubmissionId * out_submission_id,
    unsigned num_files,
    const char ** paths,
    unsigned * num_ranges,
    size_t * range_offsets,
    size_t * range_sizes,
    void ** range_dsts,
    NvFileStreamerDevice device
)
{
    // default the id to 0 ("none") so every return path - including early failures and a throw
    // during argument marshalling - leaves a defined value the caller can rely on
    if (out_submission_id != nullptr)
    {
        *out_submission_id = 0;
    }

    try
    {
        auto s = static_cast<impl::Streamer *>(streamer);
        if (s == nullptr)
        {
            return static_cast<int>(common::ResponseCode::InvalidParameterError);
        }

        // credentials are streamer-scoped (runai_set_credentials), not per request
        return submit_request(s, out_submission_id, num_files, paths, num_ranges, range_offsets, range_sizes, range_dsts, device);
    }
    catch (const common::Exception & e)
    {
        // Report the specific code rather than collapsing it to UnknownError. async_request lets typed
        // exceptions escape - a null destination or a byte-total overflow both throw InvalidParameterError -
        // and UnknownError is the code that tells a caller to abort everything, whereas an argument error is
        // attributable to this submission and recoverable. Nothing is committed when these throw, so
        // returning the real code leaves no responses owed.
        return static_cast<int>(e.error());
    }
    catch(...)
    {
    }
    return static_cast<int>(common::ResponseCode::UnknownError);
}

int runai_response(
    void * streamer,
    NvFileStreamerSubmissionId * out_submission_id,
    unsigned * file_index,
    unsigned * index,
    int * submission_done,
    unsigned timeout_ms)
{
    try
    {
        if (streamer == nullptr || file_index == nullptr || index == nullptr)
        {
            return static_cast<int>(common::ResponseCode::InvalidParameterError);
        }

        auto * s = static_cast<impl::Streamer *>(streamer);
        bool done = false;
        auto r = s->response(timeout_ms, done);

        *index = r.index;
        *file_index = r.file_index;
        if (out_submission_id != nullptr) *out_submission_id = r.submission_id;
        if (submission_done != nullptr) *submission_done = done ? 1 : 0;
        return static_cast<int>(r.ret);
    }
    catch (const common::Exception & e)
    {
        // report the specific code (see runai_request for why this matters)
        return static_cast<int>(e.error());
    }
    catch(...)
    {
    }
    return static_cast<int>(common::ResponseCode::UnknownError);
}

const char * unexpected_error = "Unexpected error occured";

int runai_probe_direct_block_size(
    void *        streamer,
    const char ** paths,
    unsigned      num_paths,
    size_t *      out_block)
{
    if (out_block == nullptr)
    {
        return static_cast<int>(common::ResponseCode::InvalidParameterError);
    }

    // Written BEFORE anything that can fail, so EVERY exit leaves a usable block.
    //
    // The caller lays out its ring from this number and divides by it. The Python binding treats a
    // zero as a broken promise and refuses to stream, so an exit that returns an error code without
    // writing here turns an inconclusive probe into a failed model load.
    //
    // That is the opposite of what this API is for. It reports a code AND a block: a mount that could
    // not be measured costs a page-aligned layout now and is asked again on the next submission.
    // Streamer::direct_block_for keeps that promise on all of its own exits; this wrapper has to keep
    // it on the ones before and around the call.
    //
    // The host page size, as the same fallback direct_block_for uses when nothing could be measured.
    *out_block = static_cast<size_t>(::sysconf(_SC_PAGESIZE));

    try
    {
        auto s = static_cast<impl::Streamer *>(streamer);
        if (s == nullptr || (num_paths != 0 && paths == nullptr))
        {
            return static_cast<int>(common::ResponseCode::InvalidParameterError);
        }

        std::vector<std::string> list;
        list.reserve(num_paths);
        for (unsigned i = 0; i < num_paths; ++i)
        {
            if (paths[i] != nullptr)
            {
                list.emplace_back(paths[i]);
            }
        }

        return static_cast<int>(s->direct_block_for(list, *out_block));
    }
    catch (const common::Exception & e)
    {
        return static_cast<int>(e.error());
    }
    catch (...)
    {
    }
    return static_cast<int>(common::ResponseCode::UnknownError);
}

const char * runai_response_str(int response_code)
{
    try
    {
        return common::description(response_code);
    }
    catch(const std::exception& e)
    {
    }

    return unexpected_error;
}

int runai_list_files(
    void *                streamer,
    const char *          prefix,
    int                   is_recursive,
    const char **         allow_patterns,
    unsigned              num_allow_patterns,
    const char **         ignore_patterns,
    unsigned              num_ignore_patterns,
    RunaiFileListCallback callback,
    void *                user_data)
{
    try
    {
        if (!streamer || !prefix || !callback)
            return static_cast<int>(common::ResponseCode::InvalidParameterError);

        std::vector<std::string> allow, ignore;
        for (unsigned i = 0; allow_patterns && i < num_allow_patterns; ++i) allow.emplace_back(allow_patterns[i]);
        for (unsigned i = 0; ignore_patterns && i < num_ignore_patterns; ++i) ignore.emplace_back(ignore_patterns[i]);

        auto * s = static_cast<impl::Streamer *>(streamer);
        // credentials are streamer-scoped (runai_set_credentials), applied when the listing client is built
        const auto files = s->list_files(prefix, is_recursive != 0, allow, ignore);
        for (const auto & entry : files)
        {
            callback(entry.first.c_str(), entry.second, user_data);
        }

        return static_cast<int>(common::ResponseCode::Success);
    }
    catch (const common::Exception & e)
    {
        return static_cast<int>(e.error());
    }
    catch (...)
    {
    }
    return static_cast<int>(common::ResponseCode::UnknownError);
}
