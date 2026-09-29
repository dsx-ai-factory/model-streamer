/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

#include "posix_io/strategy/strategy.h"
#include "common/submission/submission_id.h"

namespace runai::llm::streamer::impl
{

// What one submission did, and which reader served it.
//
// A small struct with an accessor, NOT a metrics framework. The point is that a TEST can read these
// numbers. Checking them by reading log lines would tie the test to the wording of a message, and
// the wording changes.
struct SubmissionStats
{
    SubmissionId submission_id = 0;

    // Which reader served each file. A submission can now use two of them at once - the synchronous
    // pool for one file and an engine for another - so "why was this slow?" cannot be answered
    // without knowing what read what.
    //
    // One entry per file that produced a transfer. A file with no ranges reads nothing and appears
    // here, so that "this file was skipped" and "this file is missing" stay different answers.
    struct FileStrategy
    {
        std::string path;
        posix_io::Strategy strategy = posix_io::Strategy::SyncBuffered;
    };
    std::vector<FileStrategy> files;

    // Mounts that had to share an engine because the limit was reached. Zero unless
    // RUNAI_STREAMER_FS_MAX_ENGINES is too low for the number of mounts read.
    unsigned shared_engine_mounts = 0;

    // NOT HERE, on purpose. Achieved depth, short-read re-stages and bounced bytes all belong to a
    // WORKER rather than to a submission, because one worker serves many submissions and interleaves
    // their chunks. They live in AsyncIoCounters above, summed across workers.
    //
    // There is a second reason they could not live here even if the scope fitted: this struct is
    // recorded when a submission is DISPATCHED, before any read has happened, so nothing measured
    // during the reads can reach it. Everything here is knowable in advance.
};

// What the async workers have done, summed over all of them.
//
// ENGINE-SCOPED, not per submission, and that is a correction to design 5.11. A worker serves many
// submissions and interleaves their chunks, so "the bytes this submission bounced" would mean
// attributing a shared worker's work after the fact. Achieved depth is worse: submissions share one
// engine's window, so the highest number of reads "during" a submission has no clear meaning at all.
//
// The questions these answer are global anyway - is congruent placement working, are reads being
// split, is the window being filled - so a total is the honest shape.
struct AsyncIoCounters
{
    // Every byte delivered by an async worker. Only useful beside bounced_bytes.
    uint64_t bytes_read = 0;

    // Bytes copied out of a scratch buffer.
    //
    // AN ASSERTION, and the reason this struct exists. Reading into buffers the streamer placed itself
    // should bounce NOTHING: each region starts at an offset that makes it congruent with its file
    // offset, so no edge needs a partial block. A number that grows with bytes_read means the
    // placement has stopped working and every byte is being copied - which costs throughput and
    // reports no error anywhere.
    //
    // Non-zero is expected only for destinations the CALLER placed, which the range API allows.
    uint64_t bounced_bytes = 0;

    // Reads the kernel answered with fewer bytes than asked for, and which were re-staged for the
    // rest. Routine under io_uring rather than a fault; a large number against the number of chunks
    // says reads are being split and each one is costing more than one round trip.
    uint64_t short_read_restages = 0;

    // The most reads any single engine had outstanding at once.
    //
    // A MAXIMUM ACROSS ENGINES, not a sum: each engine has its own window, and adding them would
    // describe a queue depth no device ever saw. Well below the configured depth means the window is
    // not being filled, which is what says whether io_submit time should be read as a real loss.
    unsigned achieved_depth = 0;

    // The average number of reads outstanding, weighted by how long each level lasted.
    //
    // achieved_depth is a HIGH-WATER MARK: it says the window was full ONCE, which is a different
    // claim and a much weaker one. A run that touches 64 outstanding for an instant and then averages
    // three reports the same maximum as one that sits at 64 throughout, and the two have completely
    // different explanations for a slow read.
    //
    // Carried as the numerator and denominator rather than a ratio, so summing across workers is
    // meaningful: a mean cannot be added, and each worker runs for its own length of time.
    uint64_t inflight_nanos = 0;      // sum over time of (reads outstanding x nanoseconds at that level)
    uint64_t observed_nanos = 0;      // how long the worker was observed, whatever was outstanding

    // Reads outstanding on average, or 0 before anything has been observed.
    double average_inflight() const;

    // Adds another worker's numbers into these.
    AsyncIoCounters & operator+=(const AsyncIoCounters & other);
};

std::ostream & operator<<(std::ostream &, const AsyncIoCounters &);

std::ostream & operator<<(std::ostream &, const SubmissionStats &);

// Stats for the streamer: one entry per submission, plus a total.
//
// PER SUBMISSION, not only a total. The streamer lives for many submissions, so a total alone cannot
// answer "did THIS submission behave badly?" - which is the question a test asks. A total alone is
// also misleading rather than merely rough: a submission served half by the synchronous pool would
// report zero for counters that only the engine can raise.
//
// Old submissions are dropped once the list is full, so a long-lived streamer cannot grow without
// limit. The total keeps counting.
//
// Thread safe. Submissions can run at the same time, and the numbers arrive from their workers.
class AsyncIoStats
{
 public:
    // How many submissions to remember. Small on purpose: this is for the last few runs and for
    // tests, not a history.
    static constexpr size_t MaxSubmissions = 64;

    void record(const SubmissionStats & stats);

    // The stats for one submission, or nothing if it is unknown or has been dropped.
    bool find(SubmissionId submission_id, SubmissionStats & out) const;

    // Every submission still remembered, oldest first.
    std::vector<SubmissionStats> submissions() const;

    // Totals over every submission, including ones already dropped.
    SubmissionStats total() const;

    // How many submissions were dropped to keep the list small. Tells a reader that `submissions()`
    // is not the whole story.
    size_t dropped() const;

 private:
    mutable std::mutex _mutex;

    std::vector<SubmissionStats> _submissions;   // oldest first
    SubmissionStats _total;
    size_t _dropped = 0;
};

}; // namespace runai::llm::streamer::impl
