#ifndef RUNAI_FILE_STREAMER_STREAMER_H
#define RUNAI_FILE_STREAMER_STREAMER_H

// Plain C that also compiles as C++, because this header ships in the SDK tarball and a C program
// must be able to include it.

#include <stddef.h>

#include "streamer/device.h"
#include "streamer/response_code.h"
#include "streamer/submission_id.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*RunaiFileStreamerFileListCallback)(const char* path, size_t file_size, void* user_data);

// Library for reading files concurrently into host memory buffers
// A single submission (runai_file_streamer_request) may cover many files, and many submissions may be in flight at
// once - each response carries the id of the submission it belongs to
// NOT THREAD SAFE - caller must not send requests and responses in parallel

// Creates a streamer object; returns Success or an error code.
// Takes no configuration arguments - the streamer configures itself from the environment. Each variable
// below sets BOTH backends; when a variable is unset the two backends fall back to different defaults:
//   RUNAI_STREAMER_CONCURRENCY    : number of worker threads. Unset: 16 filesystem, 8 object storage
//   RUNAI_STREAMER_CHUNK_BYTESIZE : bytes per read call. Unset: 2 MiB filesystem, the S3 client default
//                                   for object storage. Minimums are enforced - 2 MiB and 5 MiB respectively
// Worker pools are created lazily, one per backend actually used.

int runai_file_streamer_start(void ** streamer /* return parameter */);

// destroys streamer object

void runai_file_streamer_end(void * streamer);

// Set the streamer's object-storage credentials as a general key/value dictionary (param_keys /
// param_values / num_params). Keys are the plugin's canonical config-parameter names (e.g.
// "access_key_id", "secret_access_key", "session_token", "region", "endpoint"); arbitrary keys are
// carried through to the backend. Credentials are streamer-scoped and set once: setting the same
// credentials again returns Success; a different set after the first returns CredentialsAlreadySet (create
// a new streamer for a different identity). Call this before submitting object-storage reads / listing.
int runai_file_streamer_set_credentials(
    void * streamer,
    const char ** param_keys,
    const char ** param_values,
    unsigned num_params
);

// Choose how filesystem reads are served: an ORDERED PREFERENCE LIST of strategy names, best first,
// e.g. "io_uring_buffered,sync_buffered". The first the host can provide wins; the rest are rejected
// with a logged reason.
//
// Names: io_uring_direct, io_uring_buffered, libaio_direct, sync_buffered. An unknown name, a
// duplicate or an empty entry returns InvalidParameterError - a typo must not silently become a
// fallback nobody asked for. A list the host cannot serve is an error too, not a quiet fall-through
// to the synchronous reader; include sync_buffered to allow that explicitly.
//
// Streamer-scoped and SET ONCE, like runai_file_streamer_set_credentials: the same value again returns Success, a
// different value returns FsStrategyConflict. A list this host cannot serve returns
// FsStrategyUnavailable. The list is resolved on the first filesystem
// submission, and any different value after that is rejected too - by then an engine has been built
// for the resolved answer. Create a new streamer to use a different strategy.
//
// Optional. Without it the streamer reads RUNAI_STREAMER_FS_STRATEGY, defaulting to the synchronous
// reader. Object-storage reads are unaffected: the strategy names a filesystem engine, and a
// submission that reads object storage never consults it.
int runai_file_streamer_set_fs_strategy(
    void * streamer,
    const char * candidates
);

// Multi-request submit: read multiple files concurrently.
//
// A submission is a list of files; each file carries a list of RANGES to read. A range is an
// arbitrary (source offset, size) within its file with its own destination - ranges need not be
// contiguous in the file, need not be contiguous in memory, and need not be ordered.
//
// num_files     : number of files to read
// paths         : list of file paths - one entry per file, however many ranges that file has
// num_ranges    : number of ranges for each file
// range_offsets : flat array of sum(num_ranges) source offsets, each within its owning file
// range_sizes   : flat array of sum(num_ranges) range sizes in bytes
// range_dsts    : flat array of sum(num_ranges) destination pointers
// device        : where those destinations live - one device for the whole submission
//
// The three flat arrays are indexed identically and grouped by file in the order of paths: file f's
// ranges occupy [sum(num_ranges[0..f)), sum(num_ranges[0..f])). Destinations must not overlap.
//
// RESPONSE COUNT - a submission owes exactly sum(num_ranges) responses, one per range:
//   - a ZERO-SIZED range still gets its own response (it is completed immediately, without reaching
//     storage), so it must be counted like any other;
//   - a file with num_ranges[f] == 0 contributes no responses, and is otherwise accepted.
// Size the response loop by that sum. runai_file_streamer_response blocks indefinitely at timeout_ms = 0, so a
// caller that skips zero-sized ranges when counting waits for a response that has already been
// delivered. A submission with sum(num_ranges) == 0 owes nothing and completes immediately.
//
// DEVICE - one per submission, so every destination in it lives on the same device. A load that
// scatters across several GPUs is sent as several submissions, which run together and are drained by
// their own ids. Only RUNAI_FILE_STREAMER_DEVICE_CPU is served today; anything else returns UnsupportedDeviceType and
// commits nothing, so no responses are owed for it.
//
// Credentials are NOT passed here - set them once via runai_file_streamer_set_credentials.
//  out_submission_id : always set to this submission's id once one is assigned, and left 0 only
//                      if the call fails before that (e.g. invalid parameters). On Success it
//                      identifies the submission; use it to demux responses from
//                      runai_file_streamer_response. If the call fails after the submission was committed,
//                      its responses are still delivered and can be drained by this id.
int runai_file_streamer_request(
    void * streamer,
    RunaiFileStreamerSubmissionId * out_submission_id /* return parameter */,
    unsigned num_files,
    const char ** paths,
    unsigned * num_ranges,
    size_t * range_offsets,
    size_t * range_sizes,
    void ** range_dsts,
    RunaiFileStreamerDevice device
);

// Multi-request response. Returns the next ready range from any in-flight submission.
//  out_submission_id : set to the owning submission's id (which submission this response is for).
//  file_index, index : the file, and the index of the range within that file, as submitted.
//  submission_done   : set to 1 iff this was the submission's last response (it is now complete).
//  timeout_ms        : max time to wait for a response; 0 blocks indefinitely.
// ret is the truthful per-range code (Success or a specific error), TimedOut on timeout, or
// FinishedError on teardown.
int runai_file_streamer_response(
    void * streamer,
    RunaiFileStreamerSubmissionId * out_submission_id /* return parameter */,
    unsigned * file_index /* return parameter */,
    unsigned * index /* return parameter */,
    int * submission_done /* return parameter */,
    unsigned timeout_ms
);

const char * runai_file_streamer_response_str(int response_code);

// The block a caller must lay destinations out at for THESE paths, so reads can be served with
// O_DIRECT.
//
// NOT A GETTER. It opens a file and READS on each distinct mount, walking 512, 4096, 16384, 65536
// until one is accepted - because the requirement belongs to the mount and no kernel below 6.1 will
// report it. Cached per mount on this streamer, so a 200-shard model costs one probe.
//
// FILESYSTEM paths only - object-storage URIs are skipped, since they name no mount and never reach
// O_DIRECT. A submission cannot legally mix the two (runai_file_streamer_request rejects that with
// UnsupportedBackendMix), but this is called before any submission exists, so URIs are skipped rather
// than treated as an error.
//
// Path-aware because the answer is per mount, and a request can span several. It returns the LARGEST
// any of them requires: congruence at a power of two implies congruence at every smaller one, so one
// number satisfies them all, while each mount may still use a smaller block internally.
//
// Never use a larger block than this reports. Over-padding is not free - measured on NFS under the
// `chunks` partition policy, a 64 KiB layout against a 4 KiB mount cost 2.4x the load time.
//
//  out_block  always set. On Success it is measured. On UnknownError nothing could be probed - every
//             path missing or unreadable - and it holds the host page size so the caller can still
//             lay out its buffers. Ask again on the next submission rather than caching that value:
//             a long-lived streamer would otherwise keep it for the life of the process.
//
// ret is Success, or UnknownError when nothing could be measured. It does not fail a submission.
int runai_file_streamer_probe_direct_block_size(
    void *        streamer,
    const char ** paths,
    unsigned      num_paths,
    size_t *      out_block
);

// List files at the given object storage prefix.
//
// streamer is a handle from runai_file_streamer_start; listing reuses its object-storage clients, backend handle and
// credentials (set once via runai_file_streamer_set_credentials) rather than creating throwaway ones.
//
// For each matching entry the callback is invoked as:
//   callback(path, file_size, user_data)
// where path is the full object URI and file_size is the size in bytes.
// user_data is passed through to every callback invocation unchanged.
//
// Example:
//   struct Result { std::vector<std::pair<std::string,size_t>> files; };
//   Result result;
//   runai_file_streamer_list_files(streamer, "s3://my-bucket/models/", 1,
//       nullptr, 0, nullptr, 0,
//       [](const char* p, size_t sz, void* ud) {
//           static_cast<Result*>(ud)->files.emplace_back(p, sz);
//       }, &result);
//
// allow_patterns / ignore_patterns are fnmatch(3) patterns; NULL means no filter.
int runai_file_streamer_list_files(
    void *                   streamer,
    const char *             prefix,
    int                      is_recursive,
    const char **            allow_patterns,
    unsigned                 num_allow_patterns,
    const char **            ignore_patterns,
    unsigned                 num_ignore_patterns,
    RunaiFileStreamerFileListCallback    callback,
    void *                   user_data
);

#ifdef __cplusplus
}   // extern "C"
#endif

#endif // RUNAI_FILE_STREAMER_STREAMER_H
