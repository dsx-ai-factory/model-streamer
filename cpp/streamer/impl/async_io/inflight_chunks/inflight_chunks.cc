#include "streamer/impl/async_io/inflight_chunks/inflight_chunks.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

posix_io::RequestId InflightChunks::add(const Chunk & chunk, uint64_t workload_id, unsigned batch_index)
{
    const auto id = _next_id++;

    // 64 bits at ~1,280 requests/s takes longer to wrap than the hardware will exist, so this asserts
    // the counter's own integrity rather than guarding a reachable case.
    ASSERT(_chunks.count(id) == 0) << "request id " << id << " collided with a live request";

    _chunks[id] = InflightChunk{ chunk, chunk.offset, chunk.bytesize, workload_id, batch_index };
    return id;
}

const InflightChunk * InflightChunks::find(posix_io::RequestId id) const
{
    const auto it = _chunks.find(id);
    return it == _chunks.end() ? nullptr : &it->second;
}

Progress InflightChunks::record(posix_io::RequestId id, size_t bytes_transferred)
{
    const auto it = _chunks.find(id);
    ASSERT(it != _chunks.end()) << "no in-flight chunk for request " << id << " - check find() first";

    auto & entry = it->second;

    // Compared against what THIS pass asked for, never against the original extent. After a short
    // read those differ, and using the extent would judge every later pass short as well.
    if (bytes_transferred >= entry.remaining)
    {
        entry.cursor += entry.remaining;
        entry.remaining = 0;
        return Progress::Complete;
    }

    if (bytes_transferred == 0)
    {
        // Not an error from the kernel's point of view - it simply had nothing more to give. The
        // caller asked for bytes the file does not have, which is EofError, not success.
        LOG(ERROR) << "Read returned no further bytes at offset " << entry.cursor
                   << " with " << entry.remaining << " still owed";
        return Progress::Eof;
    }

    entry.cursor += bytes_transferred;
    entry.remaining -= bytes_transferred;
    return Progress::Partial;
}

Chunk InflightChunks::pending(posix_io::RequestId id) const
{
    const auto it = _chunks.find(id);
    ASSERT(it != _chunks.end()) << "no in-flight chunk for request " << id;

    const auto & entry = it->second;

    // The destination advances with the cursor: the bytes already read went to the front of the
    // buffer, so the rest goes after them.
    Chunk rest = entry.chunk;
    rest.offset = entry.cursor;
    rest.bytesize = entry.remaining;

    char * const base = entry.staging.valid() ? entry.staging.data : entry.chunk.buffer;
    rest.buffer = base + (entry.cursor - entry.chunk.offset);
    return rest;
}

void InflightChunks::set_staging(posix_io::RequestId id, const StagingBuffer & buffer)
{
    const auto it = _chunks.find(id);
    ASSERT(it != _chunks.end()) << "no in-flight chunk for request " << id;

    it->second.staging = buffer;
}

Chunk InflightChunks::release(posix_io::RequestId id)
{
    const auto it = _chunks.find(id);
    ASSERT(it != _chunks.end()) << "no in-flight chunk for request " << id;

    const Chunk chunk = it->second.chunk;
    _chunks.erase(it);
    return chunk;
}

void InflightChunks::set_bounce(posix_io::RequestId id, char * scratch, size_t skip, size_t wanted)
{
    const auto it = _chunks.find(id);
    ASSERT(it != _chunks.end()) << "no in-flight chunk for request " << id;

    ASSERT(it->second.scratch == nullptr)
        << "request " << id << " already holds a scratch buffer - a pass was staged without landing";

    it->second.scratch = scratch;
    it->second.scratch_skip = skip;
    it->second.scratch_wanted = wanted;
}

char * InflightChunks::clear_bounce(posix_io::RequestId id)
{
    const auto it = _chunks.find(id);
    if (it == _chunks.end())
    {
        return nullptr;   // already released; nothing was held
    }

    char * scratch = it->second.scratch;
    it->second.scratch = nullptr;
    it->second.scratch_skip = 0;
    it->second.scratch_wanted = 0;
    return scratch;
}

InflightChunk * InflightChunks::find_mutable(posix_io::RequestId id)
{
    const auto it = _chunks.find(id);
    return it == _chunks.end() ? nullptr : &it->second;
}

void InflightChunks::release_all_scratch(const std::function<void(char *)> & give)
{
    for (auto & [id, entry] : _chunks)
    {
        (void)id;
        if (entry.scratch != nullptr)
        {
            give(entry.scratch);
            entry.scratch = nullptr;
            entry.scratch_skip = 0;
            entry.scratch_wanted = 0;
        }
    }
}

void InflightChunks::release_all_staging(const std::function<void(const StagingBuffer &)> & give)
{
    for (auto & [id, entry] : _chunks)
    {
        (void)id;
        if (entry.staging.valid())
        {
            give(entry.staging);
            entry.staging = StagingBuffer{};
        }
    }
}

void InflightChunks::clear()
{
    _chunks.clear();   // _next_id is deliberately not reset
}

size_t InflightChunks::size() const
{
    return _chunks.size();
}

}; // namespace runai::llm::streamer::impl
