#pragma once

#include <cstddef>
#include <ostream>

#include "streamer/impl/config/config/config.h"

namespace runai::llm::streamer::impl
{

// What this process will actually use, as opposed to what the operator asked for.
//
// CONSTRUCT ON THE FIRST WORKLOAD, never at streamer construction. Depth is divided by
// RUNAI_STREAMER_PROCESS_GROUP_SIZE, which the Python layer does not write until stream_files() -
// long after runai_file_streamer_start() has returned. Building this early reads the unset default of 1, skips the
// division, and says nothing: the device then sees eight times the intended depth at TP=8.
//
// Every value is clamped against its real ceiling here and logged when clamped. A configured number
// that quietly never materialises is worse than one that is rejected.
class AsyncIoSettings
{
 public:
    // node_wide_depth is resolved for this worker's mount, so the config is not consulted for it.
    //
    // max_read_bytesize defaults to this host's kernel ceiling; it is a parameter so a test can check
    // the clamp without depending on the page size of the machine it runs on.
    AsyncIoSettings(const Config & config, unsigned node_wide_depth, size_t max_read_bytesize);
    AsyncIoSettings(const Config & config, unsigned node_wide_depth);

    // In-flight requests for THIS process: the node-wide figure divided by the number of streamer
    // processes on the node, then bounded at both ends by MinDepth and MaxDepth below.
    unsigned depth() const;

    // Bounds on the RESOLVED depth - what becomes the ring - not on the configured node-wide value.
    //
    // The floor exists because depth 1 is a serial reader paying for the whole asynchronous
    // apparatus, and the division can produce it silently. It bites only above ~170 processes on one
    // node, where the node-wide overshoot is a few requests, so it costs nothing in practice.
    //
    // The ceiling is a SAFETY bound, not a target. What saturates a device is bytes in flight, not
    // requests: at the default 8 MiB chunk even 32 requests is ~268 MB outstanding, past bandwidth x
    // latency for any storage we run on. 1024 is roughly 34x that, and twice the largest depth
    // InstantTensor can produce - so it can only ever catch a mis-set variable, never limit
    // throughput.
    //
    // Both match InstantTensor's max(512 // world_size, 3), which is our formula and our default
    // arrived at independently, with 3 as its floor.
    static constexpr unsigned MinDepth = 3;
    static constexpr unsigned MaxDepth = 1024;

    // Bytes per request, clamped to what the kernel will read in one go. Beyond that the kernel
    // short-reads and the caller re-stages, so the bytes actually in flight would be depth x the cap
    // rather than depth x this - a window smaller than the one reserved against the memory limit,
    // and than the one a sweep measured.
    size_t chunk_bytesize() const;

    // Streamer processes on this node, from RUNAI_STREAMER_PROCESS_GROUP_SIZE. 1 when unset, which is
    // both the single-process case and the "Python has not written it yet" case - the second is a bug
    // and the reason this object is built late.
    unsigned process_group_size() const;

 private:
    unsigned _process_group_size;
    unsigned _depth;
    size_t _chunk_bytesize;
};

std::ostream & operator<<(std::ostream &, const AsyncIoSettings &);

}; // namespace runai::llm::streamer::impl
