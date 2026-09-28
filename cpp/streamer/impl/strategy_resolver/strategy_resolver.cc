#include "streamer/impl/strategy_resolver/strategy_resolver.h"

#include <sstream>
#include <utility>
#include <vector>

#include "common/exception/exception.h"
#include "posix_io/io_uring_probe/io_uring_probe.h"
#include "posix_io/libaio_probe/libaio_probe.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

namespace
{

using posix_io::Strategy;

// Why this candidate cannot serve, or Success if it can.
//
// SyncBuffered is always available - it is the existing threadpool, which needs nothing probed. That
// is what makes it usable as the last entry of a list and what makes exhaustion mean something: a
// list that ends in sync_buffered cannot be exhausted, and one that does not was a deliberate choice
// to fail rather than fall back.
common::ResponseCode availability_of(Strategy strategy)
{
    switch (strategy)
    {
    case Strategy::SyncBuffered:
        return common::ResponseCode::Success;

    case Strategy::IoUringBuffered:
    case Strategy::IoUringDirect:
        // Both use the same ring. What differs is how each FILE is opened, and that is decided when it
        // is opened - it depends on the mount, and on whether the destination is congruent with the
        // file offset. So availability here asks only whether this host has io_uring at all.
        return posix_io::IoUringProbe::instance().capability().error;

    case Strategy::LibaioDirect:
        // Almost always available - Docker's default seccomp profile does not block io_setup, unlike
        // io_uring_setup. Probed anyway, because /proc/sys/fs/aio-max-nr is node wide and shared with
        // every other pod, so a busy node can refuse a context however healthy the kernel is.
        return posix_io::LibaioProbe::instance().capability().error;
    }

    return common::ResponseCode::UnknownError;
}

} // namespace

StrategyResolver::StrategyResolver(std::string default_candidates, Availability availability) :
    _availability(availability ? std::move(availability) : Availability(availability_of)),
    _default_candidates(std::move(default_candidates))
{}

common::ResponseCode StrategyResolver::set_candidates(const std::string & candidates)
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);

    // Parse before deciding anything, so a typo is reported as a typo rather than as a conflict.
    try
    {
        (void)posix_io::parse_candidates(candidates);
    }
    catch (const common::Exception & e)
    {
        LOG(ERROR) << "Cannot set filesystem strategy to '" << candidates << "': " << e.error();
        return common::ResponseCode::InvalidParameterError;
    }

    // Already settled. Compare against what was actually walked, which may have come from the
    // environment rather than from an earlier call - otherwise setting a strategy after the first
    // request would report Success and do nothing.
    if (_resolved_from.has_value())
    {
        if (_resolved_from.value() == candidates)
        {
            return common::ResponseCode::Success;
        }

        // _resolved is checked separately from _resolved_from - they are NOT set together. A failed
        // resolve() records the list it walked and leaves _resolved empty, so reading it here threw
        // std::bad_optional_access on exactly that path. The throw then met the catch(...) in
        // runai_file_streamer_set_fs_strategy and turned this specific, recoverable code into UnknownError, which
        // tells the caller to abort everything.
        //
        // It only ever fired with a log sink enabled: should_process_log() requires __print or
        // __file, so with neither set the message is never built and the value never read. That is
        // to say it appeared only when someone turned logging on to investigate - see the `env` on
        // this package's test target, which keeps a sink enabled so the case stays covered.
        if (_resolved.has_value())
        {
            LOG(ERROR) << "Filesystem strategy is already resolved from '" << _resolved_from.value()
                       << "' (using " << _resolved.value() << "); create a new streamer to use '"
                       << candidates << "'";
        }
        else
        {
            LOG(ERROR) << "Filesystem strategy is already resolved from '" << _resolved_from.value()
                       << "', and that resolution FAILED, so no strategy is in use; create a new"
                       << " streamer to use '" << candidates << "'";
        }

        return common::ResponseCode::FsStrategyConflict;
    }

    // Not resolved yet: first set wins, as for credentials.
    if (_candidates.has_value() && _candidates.value() != candidates)
    {
        LOG(ERROR) << "Filesystem strategy was already set to '" << _candidates.value()
                   << "'; create a new streamer to use '" << candidates << "'";
        return common::ResponseCode::FsStrategyConflict;
    }

    _candidates = candidates;
    return common::ResponseCode::Success;
}

common::ResponseCode StrategyResolver::resolve()
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return resolve_locked();
}

common::ResponseCode StrategyResolver::resolve_locked()
{
    if (_resolved.has_value())
    {
        return common::ResponseCode::Success;   // idempotent; every submission after the first
    }

    const auto & candidates = _candidates.has_value() ? _candidates.value() : _default_candidates;

    std::vector<Strategy> list;
    try
    {
        list = posix_io::parse_candidates(candidates);
    }
    catch (const common::Exception & e)
    {
        LOG(ERROR) << "Cannot read filesystem strategy '" << candidates << "': " << e.error();
        return common::ResponseCode::InvalidParameterError;
    }

    // Recorded before the walk, so that a set_candidates arriving afterwards has something to compare
    // against even if no candidate could serve.
    _resolved_from = candidates;

    std::ostringstream rejections;

    for (const auto strategy : list)
    {
        const auto available = _availability(strategy);
        if (available == common::ResponseCode::Success)
        {
            _resolved = strategy;
            LOG(INFO) << "Filesystem strategy: " << strategy << " (from '" << candidates << "')"
                      << rejections.str();
            return common::ResponseCode::Success;
        }

        // Collected rather than logged one by one: on success they explain why the winner won, and on
        // exhaustion they are the whole error.
        rejections << "; " << strategy << " rejected (" << available << ")";
    }

    LOG(ERROR) << "No filesystem strategy in '" << candidates << "' can be served here"
               << rejections.str()
               << ". Add sync_buffered to the list to allow the synchronous reader";
    return common::ResponseCode::FsStrategyUnavailable;
}

posix_io::Strategy StrategyResolver::resolved() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    ASSERT(_resolved.has_value()) << "filesystem strategy read before it was resolved";
    return _resolved.value();
}

bool StrategyResolver::is_resolved() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _resolved.has_value();
}

std::string StrategyResolver::resolved_from() const
{
    const auto guard = std::unique_lock<std::mutex>(_mutex);
    return _resolved_from.value_or(std::string());
}

}; // namespace runai::llm::streamer::impl
