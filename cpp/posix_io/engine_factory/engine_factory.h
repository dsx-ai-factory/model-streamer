#pragma once

#include <memory>

#include "posix_io/io_engine/io_engine.h"
#include "posix_io/strategy/strategy.h"

namespace runai::llm::streamer::posix_io
{

// Build the engine for this strategy, or return nullptr if the host cannot provide it - a blocked
// io_uring_setup, a missing opcode, an aio context that cannot be sized.
//
// Takes one resolved strategy, not the candidate list: walking the list and recording why each was
// rejected is the dispatcher's job. That way nullptr means one thing only - "not available here" -
// rather than also meaning "no engine was wanted". is_async(strategy) must be true.
//
// Called on the FIRST WORKLOAD, not at streamer construction: depth depends on
// RUNAI_STREAMER_PROCESS_GROUP_SIZE, which Python does not set until stream_files(), long after
// runai_file_streamer_start() returned. Building earlier would read the default of 1 and skip the division.
//
// A failure to build a real engine DEMOTES the strategy for the rest of the process
// (IoUringProbe::mark_unavailable), so resolution stops offering something that has already failed.
//
// Its own target rather than part of io_engine, because it constructs the concrete engines and they
// depend on the interface - putting it there would be a dependency cycle.
std::unique_ptr<IoEngine> make_io_engine(Strategy strategy, const AsyncIoConfig & config);

}; // namespace runai::llm::streamer::posix_io
