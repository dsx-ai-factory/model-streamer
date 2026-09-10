#include "posix_io/alignment/alignment.h"

#include <fcntl.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>
#include "utils/fd/fd.h"
#include "utils/logging/logging.h"
#include "streamer/device.h"
#include "common/response_code/response_code.h"
#include "common/submission/submission_id.h"


namespace runai::llm::streamer
{
// One range of a submission, exactly as the caller described it: its own source offset, size and
// destination. The mock keeps all three per range rather than a file offset plus a running destination
// cursor, so a submission whose ranges are non-contiguous in the file or in memory is served correctly.
struct MockRange {
    size_t offset;
    size_t size;
    char * dst;
};

struct State {
    utils::Fd file;
    std::vector<MockRange> ranges;
    unsigned total_items = 0;
    unsigned current_item = 0;

    // Set when the file could not be opened. Every range still gets a response, carrying this code -
    // the real streamer fails a file's ranges individually rather than dropping them, and a dropped
    // response would hang the caller (runai_response blocks).
    int error = 0;
};

// One submission in flight. The caller may have several at once, so each owns its files, its cursor and
// its response counters - sharing any of them attributes one submission's progress to another, and a
// shared file list means a second request destroys the first one's pending work.
struct Submission {
    std::vector<State> files;
    unsigned current_file = 0;
    unsigned response_total = 0;
    unsigned response_given = 0;

    bool drained() const { return response_given >= response_total; }
};

// ONE STREAMER's state, owned by the handle. The real library makes runai_start `new impl::Streamer`
// and runai_end `delete` it (streamer/streamer.cc:22-56), so submissions and ids belong to a handle,
// not to the process. Holding them in globals instead let a second runai_start wipe the submissions a
// live streamer was still draining - and that is reachable, not hypothetical: FileStreamer.list_files
// starts a temporary streamer when used outside its context manager, so listing during a stream would
// corrupt it here while working in production. A mock that disagrees with the product about this
// certifies nothing either way.
struct StreamerState {
    // Live submissions, keyed by the id responses are attributed to and erased once drained. A map
    // (rather than a list walked per response) because the id is the lookup key and ids are ordered.
    std::map<SubmissionId, Submission> submissions;
    SubmissionId next_id = 1;              // 0 is reserved, matching impl SubmissionsMgr::_next_id
    SubmissionId round_robin_cursor = 0;   // id served last, so the next response comes from another submission
};

// The handle IS the state. Every entry point casts it back, so a stale handle faults here exactly as it
// would in the real library, rather than quietly working against shared globals.
StreamerState & state_of(void * streamer)
{
    return *static_cast<StreamerState *>(streamer);
}


// Record one file's ranges. The seek is deliberately NOT done here: each range seeks to its own offset
// when it is served, since ranges need not be ordered or adjacent.
int request(void * streamer, const char * path, unsigned num_ranges, const size_t * range_offsets, const size_t * range_sizes, void ** range_dsts, State * state)
{
    // Record the ranges BEFORE touching the file. The submission owes exactly one response per range
    // whatever happens below, and runai_request does not consult this function's result - so returning
    // early would leave total_items at 0 while the response counter had already been raised by num_ranges,
    // and those responses would never be produced. The caller would then block forever in runai_response.
    state->ranges.reserve(num_ranges);
    for (unsigned j = 0; j < num_ranges; ++j) {
        state->ranges.push_back(MockRange{ range_offsets[j], range_sizes[j], reinterpret_cast<char*>(range_dsts[j]) });
    }
    state->total_items = num_ranges;

    // A file with NO ranges yields no transfer and no batch in the real streamer, so it is never opened.
    // All-zero-sized ranges are different: they do yield a batch, and Batch::execute opens the file before
    // looking at any size - so an unopenable file fails them (verified: it returns FileAccessError).
    if (num_ranges == 0) {
        return 0;
    }

    state->file = utils::Fd(::open(path, O_RDONLY));
    if (state->file.fd() == -1) {
        LOG(ERROR) << "Error opening file: " << path;
        state->error = 2;   // common::ResponseCode::FileAccessError, reported once per range
        return -1;
    }

    return 0;
}

// Read one sub-range. Returns 0 on success or a per-sub-range error code (matching common::ResponseCode:
// FileAccessError=2 on a read failure, EofError=3 on a short read). Even on an error this is a COMPLETED
// sub-range response: *index is set and the item is consumed, so runai_response still reports the owning
// submission id and submission_done - matching the real C API (which sets all out-params on an error response).
int response(void * streamer, unsigned * index, State * state)
{
    size_t result = 0;
    const auto & range = state->ranges[state->current_item];

    // Consume the item up front, so every exit below produces exactly one response for this range.
    *index = state->current_item;
    state->current_item++;

    // The file could not be opened: report the error rather than dropping the response. Checked BEFORE the
    // zero-sized shortcut, because the real streamer opens the file regardless of range size.
    if (state->error != 0)
    {
        return state->error;
    }

    // A zero-sized range reads nothing, so it is completed without touching the (successfully opened) file.
    if (range.size == 0)
    {
        return 0;
    }

    int ret = 0;
    try
    {
        // seek per range: the previous range may have been elsewhere in the file, or later in it
        state->file.seek(range.offset);
        result = state->file.read(range.size, range.dst, utils::Fd::Read::Eof);
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << "Failed to read from file at offset " << range.offset;
        ret = 2;   // common::ResponseCode::FileAccessError
    }

    if (ret == 0 && result != range.size)
    {
        LOG(ERROR) << "Reached EOF";
        ret = 3;   // common::ResponseCode::EofError
    }

    return ret;
}

extern "C" int runai_start(void ** streamer)
{
    // A fresh state per streamer, so a new one cannot disturb another that is still draining. No reset of
    // anything shared is needed (or possible) any more - there is nothing shared.
    try
    {
        *streamer = new StreamerState;
    }
    catch (...)
    {
        return static_cast<int>(common::ResponseCode::UnknownError);
    }
    return 0;
}

extern "C" void runai_end(void * streamer)
{
    // Really free it, like the real runai_end. A leaked state would be harmless in a test process, but
    // then a handle used after runai_end would keep working here and fault in production - which is the
    // bug FileStreamer.__exit__ clears self.streamer to avoid.
    delete static_cast<StreamerState *>(streamer);
}

// Serve the next range of ONE submission. Returns -1 when that submission has no range left.
static int submission_next_response(void * streamer, Submission & submission, unsigned * file_index, unsigned * index)
{
    while (submission.current_file < submission.files.size())
    {
        State & state = submission.files[submission.current_file];
        if (state.current_item >= state.total_items)
        {
            ++submission.current_file;
            continue;
        }
        *file_index = submission.current_file;
        return response(streamer, index, &state);
    }
    return -1;
}

extern "C" int runai_set_credentials(
    void * streamer,
    const char ** param_keys,
    const char ** param_values,
    unsigned num_params)
{
    return 0;
}

// The Python suite loads this mock and ctypes binds EVERY symbol at import, so a new runai_* entry
// point must appear here as well as in streamer.ldscript or the whole suite fails to import.
//
// Accepts anything: the mock reads no files, so it has no strategy to choose between.
extern "C" int runai_set_fs_strategy(
    void * streamer,
    const char * candidates)
{
    return 0;
}

extern "C" int runai_request(
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
    // Zeroed here, before anything that can fail, exactly as the real runai_request does: the contract
    // says the id is left 0 when the call fails before one is assigned, and a caller reusing the
    // variable would otherwise read the previous submission's id as if this one were live.
    if (out_submission_id != nullptr) {
        *out_submission_id = 0;
    }

    // Refused like the real C API, so a Python test that asks for a device gets the same answer here.
    if (device.type != NV_FILE_STREAMER_DEVICE_CPU) {
        return static_cast<int>(common::ResponseCode::UnsupportedDeviceType);
    }

    Submission submission;

    // The range arrays are flat and grouped by file in the order of paths; base walks that grouping.
    // Each file's ranges are handed over as they were submitted - every range keeps its own offset,
    // size and destination, so scattered submissions are served correctly rather than collapsed into
    // one contiguous span per file.
    size_t base = 0;
    for (unsigned i = 0; i < num_files; ++i) {
        const unsigned n = num_ranges[i];

        State state;
        request(streamer,
                paths[i],
                n,
                range_offsets + base,
                range_sizes + base,
                range_dsts + base,
                &state);

        submission.response_total += n;
        submission.files.push_back(std::move(state));
        base += n;
    }

    StreamerState & state = state_of(streamer);
    const SubmissionId id = state.next_id++;
    if (out_submission_id != nullptr) {
        *out_submission_id = id;
    }

    // A submission owing no responses is never made live: it can never be drained, so it would stall the
    // round robin forever. It is simply complete on arrival, which is what no responses means.
    if (submission.response_total > 0) {
        state.submissions.emplace(id, std::move(submission));
    }
    return 0;
}

extern "C" int runai_response(
    void * streamer,
    NvFileStreamerSubmissionId * out_submission_id,
    unsigned * file_index,
    unsigned * index,
    int * submission_done,
    unsigned timeout_ms
)
{
    StreamerState & state = state_of(streamer);
    if (state.submissions.empty()) {
        // Unreachable: the caller only asks while it has a live request, and every request owes at least
        // one response. Deliberately NOT FinishedError - that maps to None in libstreamer.py, so an
        // accounting bug would look like a clean teardown and silently truncate the stream. -1 is no
        // ResponseCode, and the out-params are left zeroed, so submission id 0 - never issued, since
        // SubmissionsMgr::_next_id starts at 1 - makes get_chunks raise on the first iteration.
        return -1;
    }

    // Round robin across the live submissions so responses INTERLEAVE. The real streamer makes no ordering
    // promise across submissions - its responder is demuxed by submission id for exactly that reason - so a
    // mock that drained them in order would certify a caller that wrongly assumes ordering. Deterministic,
    // and with one submission in flight it is the previous file-by-file order.
    auto it = state.submissions.upper_bound(state.round_robin_cursor);
    if (it == state.submissions.end()) {
        it = state.submissions.begin();
    }

    const SubmissionId id = it->first;
    Submission & submission = it->second;

    int r = submission_next_response(streamer, submission, file_index, index);
    if (r < 0) {
        return r;   // a live submission always has a range left; defensive
    }

    state.round_robin_cursor = id;
    submission.response_given += 1;

    // r is a real sub-range result: 0 (Success) or a per-range error code. Set the out-params in both cases
    // (like the real C API), so a per-range error still carries its submission id and submission_done.
    if (out_submission_id != nullptr) {
        *out_submission_id = id;
    }
    const bool done = submission.drained();
    if (submission_done != nullptr) {
        *submission_done = done ? 1 : 0;
    }
    if (done) {
        state.submissions.erase(it);
    }
    return r;
}

// The mock reads nothing, so it probes nothing. It reports the process-wide default, which is what
// the Python ring pads with in tests - a measured answer would need a real mount and a real read.
extern "C" int runai_probe_direct_block_size(void * streamer, const char ** paths, unsigned num_paths,
                                             size_t * out_block)
{
    (void)streamer; (void)paths; (void)num_paths;

    if (out_block == nullptr)
    {
        return static_cast<int>(runai::llm::streamer::common::ResponseCode::InvalidParameterError);
    }

    *out_block = runai::llm::streamer::posix_io::direct_block_size();
    return static_cast<int>(runai::llm::streamer::common::ResponseCode::Success);
}

extern "C" const char * runai_response_str(int response_code)
{
    return 0;
}

extern "C" int runai_list_files(
    void *        streamer,
    const char *  prefix,
    int           is_recursive,
    const char ** allow_patterns,
    unsigned      num_allow_patterns,
    const char ** ignore_patterns,
    unsigned      num_ignore_patterns,
    void (*callback)(const char*, size_t, void*),
    void *        user_data)
{
    namespace fs = std::filesystem;

    if (streamer == nullptr || prefix == nullptr || callback == nullptr)
    {
        return -1;
    }

    // Object storage paths are not supported in the mock
    if (::strncmp(prefix, "s3://", 5) == 0 ||
        ::strncmp(prefix, "gs://", 5) == 0 ||
        ::strncmp(prefix, "az://", 5) == 0)
    {
        return 0;
    }

    const fs::path root(prefix);
    if (!fs::exists(root))
    {
        return 1;
    }

    auto fire = [&](const fs::path& p, size_t size)
    {
        const char* cpath = p.c_str();
        if (allow_patterns != nullptr && num_allow_patterns > 0)
        {
            bool matched = false;
            for (unsigned j = 0; j < num_allow_patterns; ++j)
            {
                if (::fnmatch(allow_patterns[j], cpath, 0) == 0) { matched = true; break; }
            }
            if (!matched) return;
        }
        for (unsigned j = 0; ignore_patterns != nullptr && j < num_ignore_patterns; ++j)
        {
            if (::fnmatch(ignore_patterns[j], cpath, 0) == 0) return;
        }
        callback(cpath, size, user_data);
    };

    auto visit = [&](const fs::directory_entry& entry)
    {
        if (entry.is_regular_file())
        {
            fire(entry.path(), static_cast<size_t>(entry.file_size()));
        }
    };

    if (is_recursive)
    {
        for (const auto& e : fs::recursive_directory_iterator(root)) visit(e);
    }
    else
    {
        for (const auto& e : fs::directory_iterator(root)) visit(e);
    }

    return 0;
}

} // namespace runai::llm::streamer
