# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import List, Tuple, Optional
from collections import deque
import bisect
import enum
import numpy as np
import os
import re
import humanize

from runai_model_streamer.libstreamer.libstreamer import runai_probe_direct_block_size

import logging

logger = logging.getLogger(__name__)

RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME = "RUNAI_STREAMER_MEMORY_LIMIT"
DEFAULT_MEMORY_LIMIT_STRING = "40000000000" # 40 GB (to be set to unlimited for distributed streaming)

RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME = "RUNAI_STREAMER_RING_BUFFERS"
DEFAULT_RING_BUFFERS = 4

RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME = "RUNAI_STREAMER_MAX_PADS_PER_BUFFER"
DEFAULT_MAX_PADS_PER_BUFFER = 1024

RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME = "RUNAI_STREAMER_HUGE_PAGES"
DEFAULT_HUGE_PAGES = 0

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

# The size of a transparent huge page on x86_64 and on aarch64 with 4 KiB pages.
#
# The pool base is moved to a multiple of this, and then advised with MADV_HUGEPAGE, so the whole
# pool sits on 2 MiB pages instead of 4 KiB ones.
#
# The reason is O_DIRECT, not the copying. Every direct read pins its destination pages before the
# DMA - about 2,048 pages per 8 MiB read, and at 10 GiB/s that is roughly 2.6M pins per second, on
# the one async worker thread. On 2 MiB pages the same read pins 4. Buffered reads pay none of this,
# so it is a cost that O_DIRECT adds rather than one it removes.
#
# Doing it here rather than leaving it to the allocator, because the allocator is not ours. glibc may
# already align a large allocation to a huge page and advise it - measured on one host, a plain
# np.empty comes back with the `hg` flag set and fully huge-page backed - but that depends on the
# glibc version and a tunable, neither of which we control on a customer's node.
#
# A 2 MiB base is also a multiple of DIRECT_IO_BLOCK, so it replaces the old 4096 shift rather than
# adding to it.
#
# OFF unless RUNAI_STREAMER_HUGE_PAGES=1. See _huge_pages_enabled() for the measurements on both
# sides and for why the default is off despite the gain.
#
# THIS SIZE IS FOR THE POOL BASE ONLY. Padding between ranges stays at DIRECT_IO_BLOCK, and must: if
# congruence had to hold modulo 2 MiB, one pad could cost 2 MiB and the 1024-pad budget would want
# 2 GB per buffer instead of 4 MB.
#
# It does not, because the two sizes answer different questions. O_DIRECT alignment comes from the
# STORAGE DEVICE - its logical block, 512 or 4096. A huge page is a property of the MEMORY MAPPING.
# The kernel does not tie them together. Measured on a huge-page-backed pool: a direct read into an
# address one block into a huge page succeeds, as does three blocks in, as does the last block; only a
# non-block-aligned address fails, with EINVAL.
#
# And it costs nothing to sit inside a huge page rather than at its start - io_submit for 64 reads of
# 2 MiB took 1.14-1.42 ms with destinations one block in, against 1.27-1.45 ms with them 2 MiB
# aligned. Pinning still walks one compound page either way.
HUGE_PAGE = 2 * 1024 * 1024

# How many pads one buffer may hold before we give up and pack tightly. See _max_pads_per_buffer().


def _huge_pages_enabled() -> bool:
    """Whether to place the pool on huge pages. OFF by default - opt in with RUNAI_STREAMER_HUGE_PAGES=1.

    Off by default despite a large measured gain, because the risk is on a machine we have not tested
    and the gain is on one we have.

    MEASURED GAIN. Time inside io_submit for 64 direct reads of 2 MiB, on an unfragmented host:

        first pass   50 ms plain   ->  27 ms advised
        every pass   8-10 ms       ->  1.0-1.6 ms

    The second row is the one that counts. Faulting happens once, because the ring is reused for the
    whole stream, but PINNING happens on every direct read - O_DIRECT pins each destination page
    before the DMA. At 4 KiB that is 512 pages per 2 MiB read; at 2 MiB it is one. So this is a 7-8x
    cut in io_submit time on every pass, and io_submit runs on the single async worker, where blocking
    also stops completions being reaped.

    THE RISK, and why the default is still off. With the node's THP defrag set to `madvise` - which is
    common, and is what this host uses - advising a region opts it into DIRECT COMPACTION at fault
    time. On a node whose memory is fragmented the kernel may compact before it can hand out a huge
    page, and that wait lands inside the same io_submit. compact_stall in /proc/vmstat stayed at zero
    throughout our runs, so the bad case is unmeasured rather than absent.

    The two are asymmetric: compaction can only slow the FIRST touch of the pool, while the pinning
    gain applies to every read after it. That is an argument for turning this on once a fragmented
    node has been measured, not for turning it on blind.

    hugetlbfs would remove the risk entirely - pre-reserved pages are never compacted or swapped - but
    it needs node pre-allocation and a pod hugepages-2Mi request, which is a cluster decision rather
    than a code one (design_async_io.md 5.10).

    Anything other than "1" leaves it off, so a typo is safe rather than surprising."""
    return os.getenv(RUNAI_STREAMER_HUGE_PAGES_ENV_VAR_NAME, str(DEFAULT_HUGE_PAGES)).strip() == "1"


def _pool_alignment() -> int:
    """What the pool base is moved to: a huge page when huge pages are on, a block when they are off.

    Never smaller than DIRECT_IO_BLOCK either way, because congruent placement depends on it - a
    misaligned base would silently stop direct reads from working at all."""
    return HUGE_PAGE if _huge_pages_enabled() else DIRECT_IO_BLOCK


def _request_huge_pages(array: np.ndarray) -> None:
    """Ask the kernel to back `array` with huge pages, and say so if it did not.

    Advisory in every sense. madvise cannot fail in a way that matters here - if it is refused, or
    the kernel cannot find contiguous memory, the pool works exactly as before on 4 KiB pages and
    reads are a little slower. So every error is logged and swallowed.

    Whether it WORKED is not checked here. THP is decided when a page is first written, never when
    it is advised, so a check at this point reads zero for an untouched pool whatever the kernel is
    about to do. _report_huge_pages_once() asks later, when the memory has been written."""
    try:
        import ctypes

        MADV_HUGEPAGE = 14
        libc = ctypes.CDLL("libc.so.6", use_errno=True)

        # Only the part that is fully inside a huge page boundary at both ends. madvise takes a
        # page-aligned start, and a partial huge page at either end cannot be backed by one anyway.
        base = array.ctypes.data
        start = _align_up(base, HUGE_PAGE)
        end = (base + array.nbytes) // HUGE_PAGE * HUGE_PAGE
        if end <= start:
            return

        if libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), MADV_HUGEPAGE) != 0:
            logger.debug(
                "[RunAI Streamer] madvise(MADV_HUGEPAGE) was refused (errno %d); the ring buffer "
                "stays on 4 KiB pages", ctypes.get_errno()
            )
    except Exception as exception:  # noqa: BLE001 - never let a hint break a load
        logger.debug("[RunAI Streamer] Could not request huge pages: %s", exception)


# A smaps entry header: "7f0e1c000000-7f0e1c021000 rw-p 00000000 00:00 0 [heap]".
#
# Anchored on the hex range, because that is the only part every header has and no other line in the
# file starts with. Anything else - a field count, a "-" test - can be true of a pathname.
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")


def _mapping_memory(address: int, length: int) -> Optional[Tuple[int, int]]:
    """(resident bytes, huge-page bytes) for the mappings covering [address, address + length).

    EVERY overlapping mapping, not the one holding the first byte. A pool is one allocation but not
    always one VMA: madvise() on a sub-range SPLITS the mapping, and this module madvises the
    huge-page-aligned interior of its own pool - so after one such call the arena carries splits, and a
    later allocation inside it is covered by several VMAs.

    Measured: a 16 MiB pool laid out as a 128 KiB VMA followed by an 18 MiB one. Reading only the first
    reported 131072 bytes resident for a pool that was fully written, which put it below the one-huge-
    page floor the caller uses to decide there is something to judge - so the huge-page report went
    quiet for good, with nothing to show for it.

    A VMA may reach beyond the range, and its Rss then counts memory that is not ours. That is accepted:
    the numbers answer "has anything been written" and "is any of it huge-page backed", and neither is
    made wrong by a neighbour inside the same arena.

    Reads the mapping's own smaps entry rather than the machine-wide counter in /proc/meminfo, which
    everything else on the node also moves.

    Both numbers are needed, not just the huge-page one. Pages are only backed once they are written,
    so a pool nobody has written to reports zero huge pages and is perfectly healthy. Rss says how
    much has been written, which is what makes the other number mean anything.

    Linux only, and /proc may not be mounted, so the caller treats None as "no answer" rather than as
    a failure."""
    end = address + length

    try:
        current = False
        found = False
        resident = 0
        backed = 0

        with open("/proc/self/smaps") as smaps:
            for line in smaps:
                header = _SMAPS_HEADER.match(line)

                # A HEADER, matched on the hex range itself rather than by counting fields. The field
                # count was a heuristic, and a header line that did not match it would leave `current`
                # pointing at the previous mapping - reporting an unrelated mapping's numbers with no
                # sign that anything went wrong.
                if header is not None:
                    low = int(header.group(1), 16)
                    high = int(header.group(2), 16)
                    current = low < end and high > address
                    found = found or current
                    continue

                if not current:
                    continue

                # Accumulated across every overlapping mapping, and never returned early: the last one
                # is only known when the file ends.
                if line.startswith("Rss:"):
                    resident += int(line.split()[1]) * 1024
                elif line.startswith("AnonHugePages:"):
                    backed += int(line.split()[1]) * 1024

        return (resident, backed) if found else None
    except OSError:
        return None


def _align_up(address: int, block: int) -> int:
    """The next multiple of `block` at or after `address`."""
    return (address + block - 1) // block * block


class FilesRequestsIteratorWithBuffer:
    def __init__(self, buffer_size: int, num_buffers: int, files_chunks: List[FileChunks],
                 direct_block: int = DIRECT_IO_BLOCK) -> None:
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
        # ONE allocation sliced into num_buffers views. The pool is fixed at construction and never
        # grows, so a single contiguous region is simpler for the OS to manage than N separate ones.
        #
        # THE BASE IS ALIGNED TO A BLOCK, and each slot carries a little extra room. Both are for
        # direct reads:
        #
        #   A direct read needs the destination address and the file offset to leave the same
        #   remainder when divided by the block - not merely for the address to be aligned. The file
        #   offset comes from the file's own layout and cannot be chosen, so the ADDRESS has to be
        #   moved to match it. That is what the extra room is for: the packing loop below skips a few
        #   bytes before a range so the two line up.
        #
        #   Without this, no part of a region can be read directly, and O_DIRECT would copy every byte
        #   instead of about 0.1% of it.
        #
        # np.empty gives about 16 or 32 bytes of alignment, so the base is moved forward by hand.
        #
        # Moved to a HUGE_PAGE boundary when huge pages are on, and to a DIRECT_IO_BLOCK one when they
        # are off. 2 MiB is a multiple of 4096, so the direct-read rule holds either way - see
        # HUGE_PAGE for what the larger alignment buys and _huge_pages_enabled() for the switch.
        alignment = _pool_alignment()
        self._slot_size = buffer_size + self.direct_block * _max_pads_per_buffer()
        self._raw = np.empty(self._slot_size * num_buffers + alignment, dtype=np.uint8)
        shift = (-self._raw.ctypes.data) % alignment
        self.pool = self._raw[shift: shift + self._slot_size * num_buffers]
        self.buffers = [self.pool[i * self._slot_size: (i + 1) * self._slot_size] for i in range(num_buffers)]
        # Destinations are absolute addresses (that is the C contract), and this class packs them
        # itself, so it can recover a range's slice by subtracting its buffer's base. That is local
        # knowledge of its own allocation, not an assumption the range API makes.
        self.buffer_addresses = [buffer.ctypes.data for buffer in self.buffers]
        self._free_buffers = deque(range(num_buffers))

        # Marked as already reported when huge pages are off, so release() has nothing to do and the
        # switch really means "behave as before" rather than "do the work and stay quiet".
        self._huge_pages_reported = not _huge_pages_enabled()
        if not self._huge_pages_reported:
            _request_huge_pages(self.pool)

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
        return len(self._free_buffers) > 0

    def release(self, request: FilesRequest) -> None:
        """Return a drained request's buffer to the pool. The caller must not touch any view handed
        out for that request afterwards - the next request will overwrite it."""
        if request.buffer_index is None:
            raise ValueError("request has no buffer to release (released twice?)")
        self._report_huge_pages_once(request.buffer_index)
        self._free_buffers.append(request.buffer_index)
        request.buffer_index = None

    def _report_huge_pages_once(self, buffer_index: int) -> None:
        """Say whether the pool really got huge pages. Runs once, on the first buffer returned.

        Timed here, and not next to the madvise, because THP is decided when a page is first written
        - never when it is advised. Checking at allocation always reads zero, since nothing has been
        touched yet. A buffer coming back is the first moment the memory has really been written.

        THP is best effort. Under memory fragmentation the kernel quietly gives 4 KiB pages instead,
        the O_DIRECT pinning cost returns, and there is no error anywhere. This is the only way to
        see it."""
        if self._huge_pages_reported:
            return

        measured = _mapping_memory(self.buffer_addresses[buffer_index], self._slot_size)
        if measured is None:
            self._huge_pages_reported = True    # /proc is not readable here; it never will be
            return

        resident, backed = measured

        # Nothing written yet, so there is nothing to judge - a request of only zero-sized ranges
        # reads no bytes at all, and its buffer comes back untouched. Leave the flag alone and ask
        # again on the next release, when there may be something to see.
        if resident < HUGE_PAGE:
            return

        self._huge_pages_reported = True

        # Only "none at all" is worth a warning.
        #
        # A share would be the more useful number and it cannot be computed. AnonHugePages counts the
        # whole mapping, while only the pages written so far are backed by anything - and at this
        # point that is roughly one buffer of however many. Comparing against the pool size reports a
        # tiny fraction on every healthy load, which is a warning nobody would read twice.
        #
        # Zero is unambiguous: we asked, pages were written, and none of them came back as huge. That
        # is the fragmentation case, and the O_DIRECT pinning cost is back.
        if backed == 0:
            logger.warning(
                "[RunAI Streamer] The ring buffer got no huge pages, out of %s. Direct reads pin "
                "every destination page, so this costs read throughput. It usually means the node's "
                "memory is fragmented.",
                humanize.naturalsize(self.pool.nbytes, binary=True),
            )
        else:
            logger.debug(
                "[RunAI Streamer] Ring buffer huge pages so far: %s of a %s pool",
                humanize.naturalsize(backed, binary=True),
                humanize.naturalsize(self.pool.nbytes, binary=True),
            )

    def get_global_file_and_range(
        self, request: FilesRequest, local_file_index: int, local_range_index: int
    ) -> Tuple[int, int, np.ndarray]:
        file_id, global_range_index = self.files_requests_iterator.get_global_file_and_range(
            request, local_file_index, local_range_index
        )
        buffer = self.buffers[request.buffer_index]
        start = (request.range_dsts[request.flat_index(local_file_index, local_range_index)]
                 - self.buffer_addresses[request.buffer_index])
        size = request.files[local_file_index].sizes[local_range_index]
        return file_id, global_range_index, buffer[start: start + size]

    def next_request(self) -> Optional[FilesRequest]:
        if not self._free_buffers:
            raise RuntimeError("no free ring buffer - release a drained request before building another")

        request = self.files_requests_iterator.next_request()
        if request is None or len(request.files) == 0:
            return None

        # Take the buffer only once there is a request to put in it, so end of stream does not consume one.
        request.buffer_index = self._free_buffers.popleft()

        # Place this request's ranges in its buffer, one absolute address per range. Each range carries
        # its own destination, so placement is free - it is just a running cursor.
        #
        # Where possible a range is placed so that its ADDRESS and its FILE OFFSET leave the same
        # remainder when divided by the block. That is what lets the reader use a direct read. Skipping
        # a few bytes is all it takes, and once the first range of a file lines up, the ranges after it
        # follow by themselves as long as they are laid out one after another in the file.
        base = self.buffer_addresses[request.buffer_index]
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
    ) -> FilesRequestsIteratorWithBuffer:
        buffer_size, num_buffers = _ring_sizing(memory_mode, files_chunks, user_memory_limit)
        return FilesRequestsIteratorWithBuffer(buffer_size, num_buffers, files_chunks, direct_block)

    @staticmethod
    def with_memory_mode(
        files_chunks: List[FileChunks],
        memory_limit: Optional[int] = None,
        direct_block: int = DIRECT_IO_BLOCK,
    ) -> FilesRequestsIteratorWithBuffer:
        """memory_limit, when given, overrides the environment.

        The distributed path derives a PER RANK limit from the node total, which cannot be passed via
        the environment without mutating process state that other ranks and later calls also read."""
        if memory_limit is None:
            configured = os.getenv(RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME)
            memory_limit = int(configured if configured is not None else DEFAULT_MEMORY_LIMIT_STRING)
        return FilesRequestsIteratorWithBuffer.with_memory_cap(
            _get_memory_mode(memory_limit), files_chunks, memory_limit, direct_block
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
