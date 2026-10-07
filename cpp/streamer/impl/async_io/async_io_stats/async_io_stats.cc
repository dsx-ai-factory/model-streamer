/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/async_io/async_io_stats/async_io_stats.h"

#include <algorithm>

namespace runai::llm::streamer::impl
{

std::ostream & operator<<(std::ostream & os, const SubmissionStats & stats)
{
    os << "submission " << stats.submission_id;

    if (stats.shared_engine_mounts > 0)
    {
        os << ", mounts sharing an engine " << stats.shared_engine_mounts;
    }

    // Counted, not listed. A model has hundreds of files, and one line per file would bury the
    // numbers above. The list itself is for a test or for a reader who asks for it.
    unsigned async_files = 0;
    for (const auto & file : stats.files)
    {
        if (posix_io::is_async(file.strategy))
        {
            ++async_files;
        }
    }

    if (!stats.files.empty())
    {
        os << ", files " << stats.files.size() << " (" << async_files << " through an engine)";
    }

    return os;
}

void AsyncIoStats::record(const SubmissionStats & stats)
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);

    // The total counts every submission, including ones dropped below. Without that, a long run
    // would appear to have done less work than it did.
    //
    // Shared mounts is kept as the HIGHEST seen, not a sum. Two submissions usually read the same
    // mounts, so adding them would count the same sharing twice.
    _total.shared_engine_mounts = std::max(_total.shared_engine_mounts, stats.shared_engine_mounts);
    _total.files.clear();   // a list of every file ever read is not a total

    _submissions.push_back(stats);

    if (_submissions.size() > MaxSubmissions)
    {
        _submissions.erase(_submissions.begin());
        ++_dropped;
    }
}

bool AsyncIoStats::find(SubmissionId submission_id, SubmissionStats & out) const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);

    for (const auto & stats : _submissions)
    {
        if (stats.submission_id == submission_id)
        {
            out = stats;
            return true;
        }
    }

    return false;
}

std::vector<SubmissionStats> AsyncIoStats::submissions() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _submissions;
}

SubmissionStats AsyncIoStats::total() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _total;
}

size_t AsyncIoStats::dropped() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _dropped;
}

AsyncIoCounters & AsyncIoCounters::operator+=(const AsyncIoCounters & other)
{
    bytes_read += other.bytes_read;
    bounced_bytes += other.bounced_bytes;
    short_read_restages += other.short_read_restages;

    // MAX, not sum - see the field. Each engine has its own window, so adding them would report a
    // queue depth no device ever saw.
    achieved_depth = std::max(achieved_depth, other.achieved_depth);

    // SUMMED, unlike achieved_depth, and that is why the two parts are carried separately: adding
    // means-of-different-durations is meaningless, adding the numerators and denominators is not.
    inflight_nanos += other.inflight_nanos;
    observed_nanos += other.observed_nanos;

    return *this;
}

double AsyncIoCounters::average_inflight() const
{
    return observed_nanos == 0 ? 0.0 : static_cast<double>(inflight_nanos) / observed_nanos;
}

std::ostream & operator<<(std::ostream & os, const AsyncIoCounters & counters)
{
    os << counters.bytes_read << " bytes read";

    if (counters.bytes_read != 0)
    {
        // The ratio, because the raw number says nothing on its own: 4 KB bounced out of 4 KB is a
        // broken placement, and out of 40 GB is two edges of one region.
        os << ", " << counters.bounced_bytes << " bounced ("
           << (counters.bounced_bytes * 100 / counters.bytes_read) << "%)";
    }

    os << ", " << counters.short_read_restages << " short reads re-staged"
       << ", deepest engine reached " << counters.achieved_depth;

    // Beside the peak, not instead of it. The peak says the window CAN fill; this says whether it
    // stayed full - which is what decides whether time inside io_submit was a real loss.
    if (counters.observed_nanos != 0)
    {
        os << ", " << counters.average_inflight() << " outstanding on average";
    }

    return os;
}

}; // namespace runai::llm::streamer::impl
