#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <map>
#include <mutex>

#include "common/submission/submission_id.h"

namespace runai::llm::streamer::impl
{

// SubmissionsMgr - per-submission bookkeeping for the multi-request streamer.
//
// Assigns submission ids and tracks, per submission, how many responses remain to be consumed,
// so the single consumer can decide when a submission is complete (submission_done) and report
// its throughput. One submission == one runai_file_streamer_request call.
//
// Self-contained and thread-safe: its mutex is a STRICT LEAF - it is never held while touching
// the responder or the thread pool (and this class never calls into them), so it cannot be
// nested with those locks in either order. That keeps the streamer's lock ordering trivially
// consistent. The clock is injectable so throughput is deterministically testable.
class SubmissionsMgr
{
 public:
    using Clock = std::function<std::chrono::steady_clock::time_point()>;

    // Result of consuming one response. Consuming a response for an unknown submission is an
    // accounting bug (over-delivery past `expected`) and ASSERTs rather than returning here.
    struct Result
    {
        enum class Outcome
        {
            Pending,    // not the submission's last response yet
            Completed,  // the submission's last response - throughput fields below are valid
        };

        Outcome outcome = Outcome::Pending;

        // valid only when outcome == Completed
        size_t total_bytes = 0;
        long elapsed_ms = 0;
        size_t throughput_bps = 0;
    };

    explicit SubmissionsMgr(Clock now = default_now);

    // Mint a fresh submission id: rotating counter, skipping 0 (reserved as the "none" value /
    // Response default) and any id still live in the registry (only relevant after a 2^32 wrap).
    SubmissionId generate();

    // Register an accepted submission expecting `expected` responses totalling `total_bytes`.
    void add(SubmissionId submission_id, unsigned expected, size_t total_bytes);

    // Account for one consumed response of submission_id (see Result).
    Result consume(SubmissionId submission_id);

    // Number of live (registered, not yet completed) submissions.
    size_t size() const;

 private:
    static std::chrono::steady_clock::time_point default_now();

    struct Submission
    {
        unsigned remaining;
        size_t total_bytes;
        std::chrono::steady_clock::time_point submit_time;
    };

    Clock _now;
    mutable std::mutex _mutex;
    SubmissionId _next_id = 1;   // 0 is reserved
    std::map<SubmissionId, Submission> _submissions;
};

} // namespace runai::llm::streamer::impl
