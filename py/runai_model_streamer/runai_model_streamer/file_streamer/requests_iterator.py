from __future__ import annotations
from typing import List, Tuple, Optional
from collections import deque
from functools import partial
import bisect
import enum
import numpy as np
import os
import humanize

from runai_model_streamer.file_streamer.slot_pool import Slot, SlotPool
from runai_model_streamer.libstreamer.libstreamer import runai_probe_direct_block_size

import logging

logger = logging.getLogger(__name__)

RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME = "RUNAI_STREAMER_MEMORY_LIMIT"
DEFAULT_MEMORY_LIMIT_STRING = "40000000000" # 40 GB (to be set to unlimited for distributed streaming)

RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME = "RUNAI_STREAMER_RING_BUFFERS"
DEFAULT_RING_BUFFERS = 4

RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME = "RUNAI_STREAMER_MAX_PADS_PER_BUFFER"
DEFAULT_MAX_PADS_PER_BUFFER = 1024

class RunaiStreamerMemoryLimitException(Exception):
    pass


class MemoryCapMode(enum.Enum):
    unlimited = 1
    limited = 2
    largest_chunk = 3

class FileChunks:
    """One file plus the ranges to read from it. A range is an arbitrary (offset, size) within the
    file: ranges need not be contiguous with each other and need not be ordered.

    offsets and sizes are parallel lists rather than a list of (offset, size) pairs because they are
    exactly the C layer's range_offsets / range_sizes arrays - they pass through with no per-range
    object, which matters at a few hundred thousand tensors per model."""

    def __init__(self, id: int, path: str, offsets: List[int], sizes: List[int]) -> None:
        if len(offsets) != len(sizes):
            raise ValueError(
                f"offsets ({len(offsets)}) and sizes ({len(sizes)}) must be parallel"
            )
        self.id = id # the id of the file chunk must be unique in the context of a single stream_files request
        self.path = path
        self.offsets = offsets
        self.sizes = sizes

    @staticmethod
    def contiguous(id: int, path: str, offset: int, sizes: List[int]) -> "FileChunks":
        """The common case: sizes laid out back to back in the file, starting at offset."""
        offsets = []
        running = offset
        for size in sizes:
            offsets.append(running)
            running += size
        return FileChunks(id, path, offsets, sizes)

    def total_size(self) -> int:
        return sum(self.sizes)

    def max_chunk_size(self) -> int:
        return max(self.sizes)

    def __repr__(self) -> str:
        """Provides a clear string representation for the object."""
        return (f"FileChunks(id='{self.id}', path='{self.path}', "
                f"num_ranges={len(self.sizes)}, total_size={self.total_size()})")

class FilesRequest:
    """One submission: the files to read, plus a destination for every range.

    range_dsts is flat and in flattened range order (file 0's ranges, then file 1's, ...) - it IS the
    array handed to runai_request, and the same array response-time lookup indexes. file_base holds
    each file's first flat position, so mapping a response's (file_index, range_index) to a
    destination is O(1) rather than a prefix-sum walk per response.

    range_base is the other half of that mapping: each file's first GLOBAL range index, i.e. how many
    of that file's ranges earlier requests already covered. Both bases are frozen when the file is
    appended, which is what makes a request self-describing - everything needed to interpret one of
    its responses is in the request object, not in iterator state that has since moved on. That is the
    prerequisite for keeping several submissions in flight at once."""

    def __init__(self) -> None:
        self.files: List[FileChunks] = []
        self.file_base: List[int] = []
        self.range_base: List[int] = []
        self.num_ranges = 0
        self.range_dsts: List[int] = []
        # Which ring buffer this request was packed into. None once released - a request outlives its
        # buffer, since the caller may still hold the request object after the data has been consumed.
        self.buffer_index: Optional[int] = None

    def append(self, file_chunks: FileChunks, range_base: int) -> None:
        self.file_base.append(self.num_ranges)
        self.range_base.append(range_base)
        self.num_ranges += len(file_chunks.sizes)
        self.files.append(file_chunks)

    def flat_index(self, file_index: int, range_index: int) -> int:
        return self.file_base[file_index] + range_index

    def global_range_index(self, file_index: int, range_index: int) -> int:
        return self.range_base[file_index] + range_index


# The block size a direct read must line up with.
#
# ASKED OF THE LIBRARY, never restated here. This side pads the ring so each range's destination is
# congruent with its file offset; the library tests that congruence against its own number. A second
# copy could drift, and the mismatch would raise nothing - destinations would simply stop being
# congruent and direct reads would quietly become buffered ones.
#
# Not per path. Alignment belongs to the mount, and different mounts are served by different engines -
# but congruence at a power of two implies congruence at every smaller one, so this one value covers
# every mount a request touches. The library treats it as a ceiling it stays under.
#
# Resolved at import, like the library handle it comes from (libstreamer/__init__.py opens the .so at
# module scope). It must not change under a running streamer: the ring is laid out against it once.
# The layout block, resolved PER REQUEST from the mounts the paths actually live on - see
# direct_block_for() in the library. This module-level value is only the fallback used when no
# streamer handle is available to ask with, and equals the host page size.
DIRECT_IO_BLOCK = os.sysconf("SC_PAGESIZE")

def _alloc_slot(slot_size: int) -> Slot:
    """A fresh slot, with room for the pads a direct read needs.

    THE PADS ARE THE POINT. A direct read needs its destination address, its file offset and its
    length each to be a multiple of the block (cpp/posix_io/alignment). The file offset is the file's
    to choose, so _place() moves the ADDRESS to match it - and this extra room is what it moves
    within. Without it no range could be read directly and O_DIRECT would copy every byte instead of
    about 0.1% of it.

    The base is aligned too, but only as a convenience: _place() reaches a congruent address from any
    base, as test_every_slot_base_is_block_aligned records.

    A FREE FUNCTION taking the size, not a method. The pool keeps whatever allocator it is given, so a
    bound method would make the pool reference the iterator that owns it - a cycle, which refcounting
    cannot break. A replaced ring would then hold its buffers until the collector ran.
    """
    raw = np.empty(slot_size + DIRECT_IO_BLOCK, dtype=np.uint8)
    shift = (-raw.ctypes.data) % DIRECT_IO_BLOCK
    array = raw[shift: shift + slot_size]
    return Slot(array, array.ctypes.data)


class FilesRequestsIteratorWithBuffer:
    def __init__(self, buffer_size: int, num_buffers: int, files_chunks: List[FileChunks],
                 direct_block: int = DIRECT_IO_BLOCK, owned: bool = True) -> None:
        # This request's block, measured by the library from the mounts these paths live on. Held per
        # instance, not per module: two requests can touch different mounts, and a value fixed at
        # import could only ever be a guess about the first one.
        self.direct_block = direct_block
        self.files_requests_iterator = FilesRequestsIterator(buffer_size, files_chunks)
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers
        # DEBUG, not INFO: this class cannot know which rank it is on, and every rank builds its own
        # ring, so at INFO a distributed load emits one unattributable copy of this line per rank. It
        # also runs once per stream_files, and two of the three per model load describe the safetensors
        # metadata reads (`1 x 8 Bytes`). SafetensorsStreamer reports the ring at INFO instead - once per
        # load, with the rank - via DistributedStreamer.ring_info(). Keep this line: it is per file list,
        # so it is still the way to see an individual request's ring when debugging.
        logger.debug(
            f"[RunAI Streamer] CPU ring: {num_buffers} x {humanize.naturalsize(buffer_size, binary=True)} "
            f"= {humanize.naturalsize(buffer_size * num_buffers, binary=True)} for files: "
            f"{[file_chunks.path for file_chunks in files_chunks]}"
        )
        self._slot_size = buffer_size + self.direct_block * _max_pads_per_buffer()

        # `owned` is WHO OWNS A YIELDED TENSOR'S MEMORY, and the pool is where that is decided - see
        # SlotPool. ONE ALLOCATION PER SLOT, so a slot that has been given away can die on its own: a
        # view into a shared pool would keep the whole pool alive for as long as the caller holds any
        # one slot.
        self._pool = SlotPool(num_buffers, partial(_alloc_slot, self._slot_size), owned)

    def _place(self, request: FilesRequest, base: int, aligned: bool) -> Optional[List[int]]:
        """Addresses for this request's ranges, or None if an aligned layout does not fit.

        With aligned=True a range may be pushed forward a few bytes so that its address and its file
        offset leave the same remainder. With aligned=False the ranges are packed one after another.
        """
        dsts = []
        cursor = base
        limit = base + self._slot_size

        for file_chunks in request.files:
            for offset, size in zip(file_chunks.offsets, file_chunks.sizes):
                if aligned:
                    # 0 when they already line up, which is the usual case after the first range of a
                    # file.
                    cursor += (offset - cursor) % self.direct_block

                if cursor + size > limit:
                    return None

                dsts.append(cursor)
                cursor += size

        return dsts

    def has_free_buffer(self) -> bool:
        return self._pool.has_free()

    def release(self, request: FilesRequest) -> None:
        """Hand a drained request's slot back to the pool, which decides whether it survives."""
        if request.buffer_index is None:
            raise ValueError("request has no buffer to release (released twice?)")

        self._pool.release(request.buffer_index)
        request.buffer_index = None

    def get_global_file_and_range(
        self, request: FilesRequest, local_file_index: int, local_range_index: int
    ) -> Tuple[int, int, np.ndarray]:
        file_id, global_range_index = self.files_requests_iterator.get_global_file_and_range(
            request, local_file_index, local_range_index
        )
        slot = self._pool.slot(request.buffer_index)
        start = (request.range_dsts[request.flat_index(local_file_index, local_range_index)]
                 - slot.address)
        size = request.files[local_file_index].sizes[local_range_index]
        return file_id, global_range_index, slot.memory[start: start + size]

    def next_request(self) -> Optional[FilesRequest]:
        if not self._pool.has_free():
            raise RuntimeError("no free ring buffer - release a drained request before building another")

        request = self.files_requests_iterator.next_request()
        if request is None or len(request.files) == 0:
            return None

        # Take the buffer only once there is a request to put in it, so end of stream does not consume one.
        request.buffer_index = self._pool.take()

        # Place this request's ranges in its buffer, one absolute address per range. Each range carries
        # its own destination, so placement is free - it is just a running cursor.
        #
        # Where possible a range is placed so that its ADDRESS and its FILE OFFSET leave the same
        # remainder when divided by the block. That is what lets the reader use a direct read. Skipping
        # a few bytes is all it takes, and once the first range of a file lines up, the ranges after it
        # follow by themselves as long as they are laid out one after another in the file.
        base = self._pool.slot(request.buffer_index).address
        dsts = self._place(request, base, aligned=True)

        if dsts is None:
            # Too many pads for the room reserved. Pack tightly instead: correct, and read buffered.
            # This is better than making every buffer big enough for the worst case, which would only
            # ever be reached by a request whose ranges jump around inside the file.
            dsts = self._place(request, base, aligned=False)

        request.range_dsts = dsts

        return request

    @staticmethod
    def with_memory_cap(
        memory_mode: MemoryCapMode,
        files_chunks: List[FileChunks],
        user_memory_limit: Optional[int] = None,
        direct_block: int = DIRECT_IO_BLOCK,
        owned: bool = True,
    ) -> FilesRequestsIteratorWithBuffer:
        buffer_size, num_buffers = _ring_sizing(memory_mode, files_chunks, user_memory_limit)
        return FilesRequestsIteratorWithBuffer(
            buffer_size, num_buffers, files_chunks, direct_block, owned
        )

    @staticmethod
    def with_memory_mode(
        files_chunks: List[FileChunks],
        memory_limit: Optional[int] = None,
        direct_block: int = DIRECT_IO_BLOCK,
        owned: bool = True,
    ) -> FilesRequestsIteratorWithBuffer:
        """memory_limit, when given, overrides the environment.

        The distributed path derives a PER RANK limit from the node total, which cannot be passed via
        the environment without mutating process state that other ranks and later calls also read."""
        if memory_limit is None:
            configured = os.getenv(RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME)
            memory_limit = int(configured if configured is not None else DEFAULT_MEMORY_LIMIT_STRING)
        return FilesRequestsIteratorWithBuffer.with_memory_cap(
            _get_memory_mode(memory_limit), files_chunks, memory_limit, direct_block, owned
        )

class FilesRequestsIterator:
    def __init__(self, memory_limit: int, files_chunks: List[FileChunks]) -> None:
        self.memory_limit = memory_limit
        self.q = deque(FileChunksIterator(file_chunks)
            for file_chunks in files_chunks)
        
        # Per file, how many of its ranges have already been assigned to some request. Updated as each
        # request is BUILT, so nothing reads it at response time - a request carries its own frozen
        # range_base instead. Counting here rather than retroactively on the following call is what lets
        # several requests be in flight: the map belongs to request construction, not to draining.
        self.file_to_assigned_ranges = {}
        for file_chunks in files_chunks:
            self.file_to_assigned_ranges[file_chunks.id] = 0

    def get_global_file_and_range(
        self, request: FilesRequest, local_file_index: int, local_range_index: int
    ) -> Tuple[int, int]:
        # Answered entirely from the request, so a response can be resolved against its own submission
        # however far the iterator has moved on since. The first element is the caller-assigned
        # FileChunks.id, not a path.
        return (
            request.files[local_file_index].id,
            request.global_range_index(local_file_index, local_range_index),
        )

    def next_request(self) -> Optional[FilesRequest]:
        if not self.q:
            return None

        files_request = FilesRequest()
        current_request_memory_size = 0
        while self.q:
            file_chunks_iterator = self.q[0]
            file_chunks = file_chunks_iterator.next_chunks(
                self.memory_limit - current_request_memory_size
            )
            finished = file_chunks_iterator.is_finished()
            if finished:
                self.q.popleft()
            
            if len(file_chunks.sizes) == 0:
                if finished:
                    # The file itself has no range to stream (e.g. an empty
                    # safetensors shard): skip it and keep packing from the
                    # next file. Terminating here would silently drop every
                    # remaining file in the queue.
                    continue
                # The next range does not fit into this request's remaining
                # buffer budget: close this request; the file stays queued
                # for the next request.
                break

            # Note the condition above is on the RANGE COUNT, not the byte total: a file that
            # contributed only zero-sized ranges is a real contribution. The C++ layer answers a
            # zero-sized range like any other, and the caller's tensor indexing counts on exactly
            # one response per range.
            files_request.append(file_chunks, self.file_to_assigned_ranges[file_chunks.id])
            self.file_to_assigned_ranges[file_chunks.id] += len(file_chunks.sizes)
            current_request_memory_size += file_chunks.total_size()

        if len(files_request.files) == 0:
            files_request = None
        return files_request

class FileChunksIterator:
    """Hands out prefixes of a file's ranges, each fitting a caller-supplied byte budget. Ranges carry
    their own offsets now, so this is a slice of both parallel lists - no running offset to track."""

    def __init__(
        self, file_chunks: FileChunks
    ) -> None:
        self.id = file_chunks.id
        self.path = file_chunks.path
        self.offsets = file_chunks.offsets
        self.sizes = file_chunks.sizes
        self.next_index = 0

    def is_finished(self) -> bool:
        return self.next_index >= len(self.sizes)

    def next_chunks(self, size: int) -> FileChunks:
        start = self.next_index
        end = start
        remaining = size
        while end < len(self.sizes) and self.sizes[end] <= remaining:
            remaining -= self.sizes[end]
            end += 1
        self.next_index = end
        return FileChunks(self.id, self.path, self.offsets[start:end], self.sizes[start:end])

def _largest_range(files_chunks: List[FileChunks]) -> int:
    """The largest single range across all the files, or 0 when there is none.

    Both defaults are load bearing: a file may carry no ranges at all (an empty safetensors shard),
    and the file list itself may be empty (stream_files([])). This is the buffer's lower bound - a
    request must be able to hold at least one range, or next_request would return an empty request,
    which the caller reads as end of stream."""
    return max((max(file_chunks.sizes, default=0) for file_chunks in files_chunks), default=0)


def _ring_sizing(
    memory_mode: MemoryCapMode,
    files_chunks: List[FileChunks],
    user_memory_limit: Optional[int],
) -> Tuple[int, int]:
    """(buffer_size, num_buffers) for the ring.

    The DEPTH is the configured quantity and the buffer size is derived - the reverse of what this used
    to do. Depth saturates: measured on a 207 GiB model over object storage at a fixed 10 GiB limit,
    going from 1 buffer to 2 was worth 6.2% and 2 to 4 a further 1.5%, with nothing beyond. A saturating
    quantity has a meaningful default; a good buffer size does not, since it depends on the model and the
    budget.

    The buffer size is made as large as the depth allows (budget // N), floored by the largest single
    range because a range carries one destination address and so cannot span two buffers. That floor is
    what caps the reachable depth at budget // largest_range.

    RAM: N x B never exceeds the limit, but it is NOT pinned to it, so depth is not a free choice. Under
    a binding limit N x B is the budget and depth really is pure shape. When the stream is SMALLER than
    the limit the span search below buys buffers slightly larger than budget // N, and the total then
    varies with the depth target: Llama-3-8B at a 40 GB limit (budget 14.958 GiB) allocates 14.958 GiB at
    depth 1, 15.06 at depth 4 and 15.66 at depth 8. Raising the depth on a small model costs real RAM."""
    largest = _largest_range(files_chunks)
    total = sum(file_chunks.total_size() for file_chunks in files_chunks)

    # Explicit minimal-memory mode: one buffer holding exactly one range. Deriving a ring here would
    # defeat the only reason to ask for this mode.
    if memory_mode == MemoryCapMode.largest_chunk:
        return largest, 1

    if memory_mode == MemoryCapMode.unlimited:
        budget = total
    else:
        if user_memory_limit is None:
            raise RunaiStreamerMemoryLimitException(
                f"MemoryCapMode is Limited, but no limit supplied"
            )
        if user_memory_limit < largest:
            raise RunaiStreamerMemoryLimitException(
                f"Memory limit supplied: {user_memory_limit} cannot be smaller than: {largest}"
            )
        budget = min(user_memory_limit, total)

    target = _ring_buffers()

    # As large as the depth allows, floored by the largest range. Both terms are <= budget (the limit is
    # validated against largest above, and total >= largest always), so THIS buffer size can never take
    # the ring past the limit - which is why there is no longer a warning for that case. The span search
    # below can raise it, and is separately capped at limit // target for the same reason.
    buffer_size = max(largest, budget // target)

    # Nothing to read at all (no ranges, or only zero-sized ones): one empty buffer, as before.
    if buffer_size == 0:
        return 0, 1

    # A stream smaller than the limit is already meant to be spanned entirely by the ring - budget is
    # total in that case, so N x B is the whole model and no buffer is ever recycled. It misses, always:
    # B = total // target leaves not one byte for packing waste, and tensors are indivisible, so every
    # request stops short of B and a remainder request is arithmetic, not bad luck. Llama-3-8B at a 40 GB
    # limit packs into 4 x 3.660 GiB of a 3.739 GiB buffer and then needs a 5th request for the last
    # 336 MiB, which cannot start until the consumer frees one of the four.
    #
    # So spend the headroom the limit already granted on slightly larger buffers. On Llama-3-8B that is
    # 108 MiB more (15.06 GiB against a 37.25 GiB limit) to make the intent actually hold.
    #
    # Only when there IS headroom, which is also what keeps this off the path where it would cost most:
    # a 150k tensor model reaches here only if the limit exceeds its entire size. When the limit binds
    # (every distributed rank, and any model larger than the limit) recycling is the point and there is
    # nothing to spend.
    if memory_mode == MemoryCapMode.limited and total < user_memory_limit:
        spanned = _span_whole_stream(files_chunks, buffer_size, user_memory_limit // target, target)
        if spanned is not None:
            return spanned

    # The depth the budget can actually pay for, which drops below the target whenever the largest range
    # lifted the buffer size above budget // target. There is no separate "no more buffers than there
    # will be requests" term: budget <= total always (it is total, or min(limit, total)), so
    # budget // buffer_size is already <= the request count, and packing waste only pushes the real count
    # higher. The single-request stream falls out of the same fact - total <= B implies budget <= B
    # implies one buffer - which is what the two safetensors metadata reads are on every model load.
    #
    # Note N x B is an ALLOCATION, not a bytes-in-flight figure. On THIS path - a binding limit, so the
    # ring recycles - requests pack to roughly 80% of a buffer because tensors are indivisible, making a
    # 40 GB limit about 32 GB of read-ahead. (The span path above is different: sizing the buffer to fit
    # the stream also happens to pack it well, measured at 97-99% on Llama-3-8B.) That gap is the cost of
    # fixed-size buffers and is not recoverable by changing N: N is already the maximum over every legal
    # buffer size, since min(target, budget // B) is non-increasing in B and B is either the smallest
    # legal size (largest) or large enough that budget // B >= target.
    num_buffers = min(target, budget // buffer_size)

    return buffer_size, num_buffers


def _packing_prefix(files_chunks: List[FileChunks]) -> List[int]:
    """Running byte totals over every range of every file, in the order requests are packed.

    One flat sequence is faithful to FilesRequestsIterator: it takes files off a FIFO queue and carries
    a request across a file boundary, so file boundaries never force a new request."""
    prefix = [0]
    running = 0
    for file_chunks in files_chunks:
        for size in file_chunks.sizes:
            running += size
            prefix.append(running)
    return prefix


def _requests_needed(prefix: List[int], buffer_size: int, cap: int) -> Optional[int]:
    """How many requests greedy packing needs for the whole stream, or None if it needs more than cap.

    Binary searching the prefix sums for each request's extent makes this O(cap x log R) rather than
    O(R): deciding the shape of a 4 buffer ring must not cost a walk over 150k tensors. Bailing out at
    cap is the other half of that - an unpackable buffer size is rejected after cap + 1 probes."""
    index = 0
    requests = 0
    last = len(prefix) - 1
    while index < last:
        # the furthest range whose end still fits in one buffer starting at this range
        nxt = bisect.bisect_right(prefix, prefix[index] + buffer_size) - 1
        if nxt <= index:
            return None    # a single range exceeds the buffer; unreachable while buffer_size >= largest
        index = nxt
        requests += 1
        if requests > cap:
            return None
    return requests


def _span_whole_stream(
    files_chunks: List[FileChunks],
    buffer_size: int,
    max_buffer_size: int,
    target: int,
) -> Optional[Tuple[int, int]]:
    """Smallest buffer in [buffer_size, max_buffer_size] that packs the whole stream into at most
    `target` requests, with the request count it achieves - or None when even the largest cannot.

    The count is returned rather than assuming `target`, so a stream that fits in fewer requests gets
    fewer buffers: a single request stream still gets exactly one buffer, not four empty ones.

    Binary search is valid because request count is non-increasing in buffer size - a bigger buffer
    takes a prefix at least as long at every step - so "fits in target requests" is monotone."""
    if max_buffer_size < buffer_size:
        return None

    prefix = _packing_prefix(files_chunks)
    if _requests_needed(prefix, max_buffer_size, target) is None:
        return None    # the limit does not stretch far enough; recycle as usual

    low, high = buffer_size, max_buffer_size
    while low < high:
        middle = (low + high) // 2
        if _requests_needed(prefix, middle, target) is None:
            low = middle + 1
        else:
            high = middle

    return low, _requests_needed(prefix, low, target)


def _max_pads_per_buffer() -> int:
    """How many pads one ring slot reserves room for.

    A pad is spent each time the write cursor and the file offset fall out of step, because a direct
    read needs them to leave the same remainder when divided by the block. That happens at every file
    boundary in a request, and also at any gap inside a file whose ranges are not laid out back to
    back. Ranges that do follow one another cost nothing: the cursor and the offset advance by the
    same amount and stay in step.

    So the number to beat is the number of offset jumps in one request, which is driven by how many
    files fit in one slot. 1024 pads reserve 4 MB per slot - nothing next to a multi-GB ring.

    A request that needs more pads than this simply packs tightly and reads buffered. That is correct,
    just not direct, so getting this number wrong is slow rather than broken.
    """
    # clamped at 0: no pads means tight packing, which is the pre-direct-io behaviour and always valid
    return max(0, int(os.getenv(RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME, DEFAULT_MAX_PADS_PER_BUFFER)))


def _ring_buffers() -> int:
    # clamped at 1 so a hostile 0 cannot produce a ring with no buffers
    return max(1, int(os.getenv(RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME, DEFAULT_RING_BUFFERS)))


def _get_memory_mode(memory_limit: int) -> MemoryCapMode:
    # An int rather than the raw string, because an explicitly supplied limit never was one - and it
    # makes " -1" or "-01" behave like -1, which the string comparison rejected.
    if memory_limit == -1:
        return MemoryCapMode.unlimited
    elif memory_limit == 0:
        return MemoryCapMode.largest_chunk
    else:
        return MemoryCapMode.limited
