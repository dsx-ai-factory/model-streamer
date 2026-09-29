# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import dataclasses
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import humanize
from runai_model_streamer.file_streamer import FileChunks

import logging

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class _WorkUnit:
    """
    An internal, flattened representation of a single, indivisible chunk of work.
    This simplifies the partitioning logic by breaking down the input into its
    smallest components and tracking their original positions.
    """
    path: str
    offset: int
    size: int
    original_request_index: int # the FileChunks original id
    original_chunk_index: int   # the chunk index in the FileChunks

def partition_by_chunks(
    file_stream_requests: List[FileChunks], n: int
) -> List[List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]]:
    """
    Partitions a list of file read requests into n balanced parts.

    For each returned FileChunks object, it also provides a map from the index
    of a chunk in its new list to a tuple representing its original position
    (original_request_index, original_chunk_index, chunk_size).

    A greedy algorithm is used to assign each chunk to a partition.
    It iterates through the global sorted list of chunks (starting with the largest chunk)
    and assigns each chunk to the partition that is currently the "emptiest"
    (has the smallest total size of assigned work so far).

    Args:
        file_stream_requests: A list of FileChunks objects representing the
                              total work to be done.
        n: The number of partitions to divide the work into.

    Returns:
        A list of n partitions. Each partition is a list of tuples, where each
        tuple contains a new FileChunks object and its corresponding source map.
        The map's key is the new chunk index, and the value is a tuple
        (original request index, original chunk index, chunk size).

    Raises:
        ValueError: If n is not a positive integer.
    """
    if n <= 0:
        raise ValueError("Number of partitions (n) must be a positive integer.")
    
    if not file_stream_requests:
        return [[] for _ in range(n)]

    # 1. Flatten the input `FileChunks` into a single list of `_WorkUnit`s.
    # Zero-sized ranges are kept, not filtered: a zero-element tensor is a real safetensors entry
    # (header shape [0, 3], data_offsets [x, x]) that the reference implementation yields, and
    # create_torch_tensor reconstructs its shape. Dropping them here would make the distributed path
    # yield fewer tensors than the single-process path.
    all_units: List[_WorkUnit] = []
    for request in file_stream_requests:
        for chunk_idx, (offset, size) in enumerate(zip(request.offsets, request.sizes)):
            all_units.append(_WorkUnit(
                path=request.path,
                offset=offset,
                size=size,
                original_request_index=request.id,
                original_chunk_index=chunk_idx
            ))

    # 2. Sort the atomic work units from largest to smallest.
    all_units.sort(key=lambda u: u.size, reverse=True)

    # 3. Distribute the sorted work units into n partitions.
    partitions_of_units: List[List[_WorkUnit]] = [[] for _ in range(n)]
    partition_sizes: List[int] = [0] * n

    for unit in all_units:
        min_size_idx = partition_sizes.index(min(partition_sizes))
        partitions_of_units[min_size_idx].append(unit)
        partition_sizes[min_size_idx] += unit.size

    # 4. Reconstruct the final `FileChunks` objects and their source maps.
    result_partitions: List[List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]] = []
    id_generator = 0
    for partition_of_units in partitions_of_units:
        new_partition: List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]] = []
        units_by_path: Dict[str, List[_WorkUnit]] = defaultdict(list)
        for unit in partition_of_units:
            units_by_path[unit.path].append(unit)
        
        for path, units in units_by_path.items():
            # Sorted by offset not in order to merge anything - ranges carry their own offsets now - but
            # because the C++ assigner only coalesces ranges that arrive in ascending file order. Sorting
            # here is what lets it turn a rank's scattered ranges back into contiguous transfers.
            #
            # Size breaks ties so that a zero sized range sorts BEFORE a range starting at the same
            # offset. Placed after, it would land at the far end of the preceding range and break the
            # transfer twice; placed before, both the file offsets and the destinations stay adjacent.
            units.sort(key=lambda u: (u.offset, u.size))

            # One FileChunks per path per rank. Previously a rank's units for a path were split into one
            # FileChunks per contiguous run, which on a 66-shard model produced 16308 entries (mean run
            # length 1.14) and 16308 duplicated path strings for 66 distinct paths.
            new_partition.append((
                FileChunks(
                    id=id_generator,
                    path=path,
                    offsets=[unit.offset for unit in units],
                    sizes=[unit.size for unit in units],
                ),
                {
                    index: (unit.original_request_index, unit.original_chunk_index, unit.size)
                    for index, unit in enumerate(units)
                },
            ))
            id_generator += 1

        result_partitions.append(new_partition)

    return result_partitions

def partition_by_files(
    file_stream_requests: List[FileChunks], n: int
) -> List[List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]]:
    """
    Partitions a list of file read requests into n parts by distributing
    whole FileChunks objects.

    This method preserves the continuity of chunks within a FileChunks object
    but may result in less balanced partitions compared to stream_files.
    The partitioning is deterministic.

    Args:
        file_stream_requests: A list of FileChunks objects representing the
                              total work to be done.
        n: The number of partitions to divide the work into.

    Returns:
        A list of n partitions. Each partition is a list of tuples, where each
        tuple contains a FileChunks object and its corresponding source map.
    """
    if n <= 0:
        raise ValueError("Number of partitions (n) must be a positive integer.")
    
    if not file_stream_requests:
        return [[] for _ in range(n)]

    # 1. Sort the FileChunks objects from largest to smallest total size.
    sorted_requests = sorted(
        file_stream_requests,
        key=lambda request: request.total_size(),
        reverse=True
    )

    # 2. Distribute the sorted FileChunks objects into n partitions.
    partitions: List[List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]] = [[] for _ in range(n)]
    partition_sizes: List[int] = [0] * n

    for request in sorted_requests:
        min_size_idx = partition_sizes.index(min(partition_sizes))
        
        # Create the source map. Since we aren't changing the range order within
        # the FileChunks object, the mapping is direct.
        #
        # Keyed on request.id, NOT the request's position in file_stream_requests: the map's contract is
        # "original FileChunks.id and range index" (see DistributedStreamer.rank_dicts_map), which is what
        # partition_by_chunks emits and what the receiving ranks look tensors up by. Today the only
        # production caller happens to assign id == position, so the two agree by accident - but a caller
        # that does not would silently receive the wrong tensor under the `files` policy.
        source_map = {
            chunk_idx: (request.id, chunk_idx, request.sizes[chunk_idx])
            for chunk_idx in range(len(request.sizes))
        }

        partitions[min_size_idx].append((request, source_map))
        partition_sizes[min_size_idx] += request.total_size()

    return partitions


def _span_ends(sizes: List[int], n: int) -> Tuple[List[int], List[int]]:
    """Where each rank's span ends, and the prefix sums used to find them.

    The prefix array is returned because the caller needs it too: a rank's byte total is one
    subtraction from it, and building it twice would double the only pass that touches every
    tensor.

    Cuts the stream into n parts with the smallest largest part.

    Binary search on the answer. For a candidate ceiling, walk left to right and cut whenever the next
    tensor would not fit; the ceiling is feasible when that needs at most n parts. The smallest
    feasible ceiling is the best possible, because every rank waits for the slowest and the tensors
    cannot be split.

    The search starts at max(largest tensor, total // n) because no partition can beat either bound:
    one tensor has to live somewhere whole, and the work has to go somewhere.

    The search asks the same question about 38 times - once per halving of the byte range - so how a
    single probe is answered decides whether this is free or slow.

    A probe must not step through the tensors one at a time. The prefix sums are what avoid it: each
    rank's cut becomes one binary search over the running totals, so a probe touches about as many
    entries as there are ranks, rather than all 150,000 tensors of a large model.

    Measured on 90,000 tensors: 19 ms this way, 705 ms stepping through them - about 1.2 s on a model
    with 150,000, on the load path. Both give the identical partition, and the slow one is shorter and
    reads more obviously, which is exactly why it is worth a warning here.
    """
    total = len(sizes)

    prefix = [0] * (total + 1)
    for index, size in enumerate(sizes):
        prefix[index + 1] = prefix[index] + size

    def cuts(ceiling: int) -> Optional[List[int]]:
        ends: List[int] = []
        start = 0
        for _ in range(n):
            if start == total:
                ends.append(total)      # fewer tensors than ranks: the rest get nothing
                continue

            # The furthest tensor whose cumulative bytes from `start` still fit under the ceiling.
            end = bisect_right(prefix, prefix[start] + ceiling) - 1
            if end == start:
                return None             # one tensor alone is over the ceiling
            ends.append(end)
            start = end

        return ends if start == total else None

    low = max(max(sizes), prefix[total] // n)
    high = prefix[total]
    while low < high:
        middle = (low + high) // 2
        if cuts(middle) is not None:
            high = middle
        else:
            low = middle + 1

    return cuts(low), prefix


def _flatten(file_stream_requests: List[FileChunks]):
    """The whole workload as one stream, in the order it sits on disk.

    Files keep the caller's order. It is the order the caller listed them in, usually the shard order,
    and imposing our own would only change which files land in which span - never how many runs there
    are, since a run always breaks at a file boundary anyway.

    Ranges WITHIN a file are put in ascending offset order, which is what makes adjacency in this list
    mean adjacency on disk. partition_by_chunks sorts the same way when it rebuilds, for the same
    reason. Size breaks ties so a zero-sized range sorts before a range starting at the same offset;
    placed after, it would sit at the far end of the preceding range and break the run twice.
    """
    paths: List[str] = []
    offsets: List[int] = []
    sizes: List[int] = []
    origins: List[Tuple[int, int]] = []

    for request in file_stream_requests:
        order = sorted(range(len(request.sizes)),
                       key=lambda i: (request.offsets[i], request.sizes[i]))
        for index in order:
            paths.append(request.path)
            offsets.append(request.offsets[index])
            sizes.append(request.sizes[index])
            origins.append((request.id, index))

    return paths, offsets, sizes, origins


def _build_span(paths, offsets, sizes, origins, start: int, end: int, first_id: int):
    """The FileChunks for one span, plus the next free id.

    One FileChunks per file, and no sorting: a span is already grouped by path and already ascending,
    so a single pass over it is enough. That is why this policy is cheaper than partition_by_chunks,
    which sorts every unit by size and then re-sorts each rank's units by offset.
    """
    partition: List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]] = []
    generated = first_id

    index = start
    while index < end:
        path = paths[index]
        stop = index
        while stop < end and paths[stop] == path:
            stop += 1

        partition.append((
            FileChunks(
                id=generated,
                path=path,
                offsets=offsets[index:stop],
                sizes=sizes[index:stop],
            ),
            {
                position - index: (origins[position][0], origins[position][1], sizes[position])
                for position in range(index, stop)
            },
        ))
        generated += 1
        index = stop

    return partition, generated


def partition_by_spans(
    file_stream_requests: List[FileChunks], n: int
) -> List[List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]]:
    """Partition by giving each rank one contiguous span of the model.

    The files are treated as ONE concatenated stream, never partitioned one at a time. Per-file is
    worse on both counts: its floor is files x n runs rather than files, and it cannot balance a big
    tensor in one file against small ones in another - measured, 3.69x imbalance on 2000 small files
    where this reaches 1.00x.

    Whole tensors only, so the broadcast between ranks is unaffected: a tensor is identified by its
    original file and index, and cuts fall between tensors.

    See design_partition_spans.md for why, and plan_partition_spans.md for the measurements.
    """
    if n <= 0:
        raise ValueError("Number of partitions (n) must be a positive integer.")

    if not file_stream_requests:
        return [[] for _ in range(n)]

    paths, offsets, sizes, origins = _flatten(file_stream_requests)

    if not sizes:
        return [[] for _ in range(n)]

    ends, _ = _span_ends(sizes, n)

    partitions = []
    generated = 0
    start = 0
    for end in ends:
        partition, generated = _build_span(paths, offsets, sizes, origins, start, end, generated)
        partitions.append(partition)
        start = end

    return partitions


@dataclasses.dataclass(frozen=True)
class RankSpan:
    """One rank's share, plus the two totals that used to come from having every rank's share."""

    # This rank's files. The same shape every policy returns, for one rank.
    partition: List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]

    # Bytes per rank, for the partition log. One subtraction per rank from the prefix sums, so the
    # log costs nothing rather than costing every other rank's objects.
    sizes_by_rank: List[int]

    # Every tensor in the model, across all ranks. This is the count the broadcast loop counts down
    # to zero, so it must stay a GLOBAL total - a rank that only knew its own share would wait for
    # tensors that were never coming, or stop while others were still sending.
    #
    # No partition needed: every tensor is assigned exactly once, so the total is just the input.
    total_chunks: int


def partition_span_for_rank(
    file_stream_requests: List[FileChunks], n: int, rank: int
) -> RankSpan:
    """The span for one rank, without building the other ranks' objects.

    Same cuts as partition_by_spans - they need every tensor's size either way - but only this rank's
    FileChunks are constructed. Measured on 150k tensors at n=8: 129 ms building all of them against
    68 ms building one. The rest of the input pass is shared, which is why it is not a straight
    eighth.

    Safe for the broadcast, and worth stating because it looks as though it should not be: a rank
    receiving a tensor it did not read takes the tensor's identity from the broadcast METADATA, never
    from a partition. The only lookup into a rank's own map happens on the sending side, for chunks it
    read itself.
    """
    if n <= 0:
        raise ValueError("Number of partitions (n) must be a positive integer.")

    if not 0 <= rank < n:
        raise ValueError(f"rank {rank} is outside the {n} partitions")

    if not file_stream_requests:
        return RankSpan(partition=[], sizes_by_rank=[0] * n, total_chunks=0)

    paths, offsets, sizes, origins = _flatten(file_stream_requests)

    if not sizes:
        return RankSpan(partition=[], sizes_by_rank=[0] * n, total_chunks=0)

    ends, prefix = _span_ends(sizes, n)

    sizes_by_rank = []
    start = 0
    for end in ends:
        sizes_by_rank.append(prefix[end] - prefix[start])
        start = end

    span_start = ends[rank - 1] if rank > 0 else 0
    partition, _ = _build_span(paths, offsets, sizes, origins, span_start, ends[rank], 0)

    return RankSpan(
        partition=partition,
        sizes_by_rank=sizes_by_rank,
        total_chunks=len(sizes),
    )

# Dict[int, Tuple[int, int, int] maps the chunk index in the corresponding
# FileChunks object to the original request index, chunk index, and chunk size.

def get_partition_policy() -> str:
    partition_policy = os.getenv("RUNAI_STREAMER_PARTITION_POLICY")
    if partition_policy is not None:
        return partition_policy
    else:
        return "spans"

def partition(file_stream_requests: List[FileChunks], n: int) -> List[List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]]:
    partition_policy = get_partition_policy()
    if partition_policy == "files":
        return partition_by_files(file_stream_requests, n)
    elif partition_policy == "chunks":
        return partition_by_chunks(file_stream_requests, n)
    elif partition_policy == "spans":
        return partition_by_spans(file_stream_requests, n)
    else:
        raise ValueError(f"Invalid partition policy: {partition_policy}")

def get_total_number_of_chunks(partitions: List[List[Tuple[FileChunks, dict]]]) -> int:
    if partitions is None or len(partitions) == 0:
        return 0
    return sum(sum(len(fc.sizes) for fc, _ in p) for p in partitions)

def get_total_size_of_partition(partition: List[Tuple[FileChunks, dict]]) -> int:
    if partition is None or len(partition) == 0:
        return 0
    return sum(fc.total_size() for fc, _ in partition)

def count_runs(partition: List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]) -> int:
    """How many sequential reads this rank's share decomposes into.

    A run is a group of ranges that continue one another inside one file, so it can be served by a
    single sequential read. A range that does not continue the previous one starts a new run.

    THE NUMBER THAT PRICES DIRECT I/O, which is why it is worth logging at all. Each break costs a
    pad in the ring buffer, because a direct read needs its destination and its file offset to leave
    the same remainder; adjacent ranges need no pad. Past the pad budget the ring gives up on aligned
    placement and reads buffered instead - correct, silent, and slower. Nothing else reports that.

    It also bounds coalescing: the C++ assigner merges ranges that arrive adjacent and in ascending
    order, so one run becomes one transfer and a broken run cannot be merged with anything.

    The floor is one per file a rank touches, since a run cannot span two files. Under `spans` that
    floor is reached. Under `chunks` the count approaches one run per tensor.
    """
    runs = 0
    for file_chunks, _ in partition:
        cursor = None
        for offset, size in zip(file_chunks.offsets, file_chunks.sizes):
            if offset != cursor:
                runs += 1
            cursor = offset + size
    return runs


def log_rank_span_info(rank: int, partition: List[Tuple[FileChunks, Dict[int, Tuple[int, int, int]]]]):
    """What this rank will actually read: files, tensors, and sequential reads.

    Logged per rank rather than for every rank at once, because a rank now builds only its own share -
    nobody holds them all. That is also where the number belongs: it describes this rank's I/O.
    """
    files = len(partition)
    tensors = sum(len(file_chunks.sizes) for file_chunks in (fc for fc, _ in partition))
    runs = count_runs(partition)

    logger.debug(
        "[RunAI Streamer][Distributed] Rank %d reads %d tensors from %d files in %d sequential "
        "reads (mean run %.1f tensors; the floor is one read per file)",
        rank, tensors, files, runs, tensors / runs if runs else 0.0,
    )


def log_partition_info(sizes_by_rank: List[int]):
    """Bytes per rank.

    Takes the sizes rather than the partitions, so a rank that built only its own share can still log
    every rank's. The numbers come from the prefix sums the cut search already produced - see
    RankSpan.sizes_by_rank.
    """
    log_string = "[RunAI Streamer][Distributed] Partitions sizes:"
    for rank, size in enumerate(sizes_by_rank):
        log_string += f" {rank}: {humanize.naturalsize(size, binary=True)} ; "
    logger.debug(log_string)


def partition_for_rank(
    file_stream_requests: List[FileChunks], n: int, rank: int
) -> RankSpan:
    """One rank's share, whichever policy is configured.

    The point of this function is that the caller does not have to know which policy can build a
    single share cheaply. `spans` can, and does. The other two cannot - their assignment depends on
    every unit's placement, so there is nothing to skip - and they build everything and slice it,
    exactly as before.
    """
    if get_partition_policy() == "spans":
        return partition_span_for_rank(file_stream_requests, n, rank)

    partitions = partition(file_stream_requests, n)

    if not 0 <= rank < len(partitions):
        raise ValueError(f"rank {rank} is outside the {len(partitions)} partitions")

    return RankSpan(
        partition=partitions[rank],
        sizes_by_rank=[get_total_size_of_partition(p) for p in partitions],
        total_chunks=get_total_number_of_chunks(partitions),
    )

