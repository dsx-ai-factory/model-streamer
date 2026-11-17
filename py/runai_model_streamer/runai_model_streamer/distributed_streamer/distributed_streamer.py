# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any, Dict, List, Iterator, NamedTuple, Optional, Tuple
import torch
import torch.distributed as dist
import os
from datetime import timedelta
import socket

from runai_model_streamer.file_streamer import (
    FileStreamer,
    FileChunks,
)

from runai_model_streamer.file_streamer.file_streamer import host_or_device, resolve_device

from runai_model_streamer.file_streamer.device_support import device_support

from runai_model_streamer.torch_device.torch_device import (
    can_measure_device_memory,
    device_free_memory,
)

from runai_model_streamer.file_streamer.requests_iterator import (
    RunaiStreamerMemoryLimitException,
    device_ring_budget,
    device_ring_floor,
    device_ring_sizing,
    SlotBounds,
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME,
    RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME,
    DEFAULT_MEMORY_LIMIT_STRING,
)

from runai_model_streamer.distributed_streamer.distribution_source import (
    EmptyDistributionSource,
    SlotRange,
)

from runai_model_streamer.distributed_streamer.prediction import (
    predicted_slots,
)

from runai_model_streamer.distributed_streamer.staged_source import (
    StagedDistributionSource,
)

from runai_model_streamer.distributed_streamer.direct_source import (
    DirectDistributionSource,
)

from runai_model_streamer.distributed_streamer.partition import (
    partition,
    partition_for_rank,
    get_total_number_of_chunks,
    get_partition_policy,
    log_partition_info,
    log_rank_span_info,
)

from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
)

from timeit import default_timer as timer

import logging
import humanize

logger = logging.getLogger(__name__)

def metadata_rows(file_stream_requests: List[FileChunks]) -> int:
    """How many ranges one metadata tensor must be able to describe.

    EVERY RANGE IN THE REQUEST. A slot cannot hold more than the share it is filled from, and a
    share cannot hold more than the whole list - so this bounds any slot without modelling how the
    ranges pack. A height derived from the packing would be a second copy of a rule that has
    already drifted once, and it would fail mid-load, with the collective already posted.

    Every rank computes it from the same list, which is what makes the broadcast shapes match. At
    32 bytes a row it costs about 2.4 MiB for a model of 76 thousand ranges, against a staging
    buffer of at least a gibibyte."""
    return max(1, sum(len(file_chunks.sizes) for file_chunks in file_stream_requests))


SENT_RING_BOUNDS = SlotBounds()
"""The bounds a sent ring is built with, and so the bounds its size must be chosen under.

NO RANGE CAP. A cap belongs to a metadata tensor's height, and nothing describes a predicted slot -
every rank works the layout out instead. Capping here would end a slot before its bytes ran out, so
the group would broadcast more slots, each one part empty.

ONE VALUE for the sizing, the prediction and the ring. They were separate parameters, and the
sizing read only the byte budget - so it chose a slot the row cap could never fill."""

DIST_BUFFER_ALIGNMENT = 512
"""What a staging buffer's START is aligned to, not its contents.

The ranges inside are packed one after another, so whether one can be read as its dtype follows
from its own offset. A buffer starting at an odd address would make that follow from wherever torch
allocated instead, and every range would be copied."""
DEFAULT_BUFFER_MIN_BYTESIZE = 1024 * 1024 * 1024


def dist_buffer_bytesize(max_chunk: int) -> int:
    """One staging buffer: the minimum, or the largest chunk if that is bigger.

    TESTS ONLY may lower the minimum. It is what gives the gate in set_is_distributed its slack:
    the free memory that gate compares against is read before the first collective, so it does not
    yet show the NCCL communicator. A gibibyte per buffer leaves room for that and more. Lowered,
    the threshold falls to about max_chunk and the communicator stops being small beside it."""
    minimum = int(os.environ.get("RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE", str(DEFAULT_BUFFER_MIN_BYTESIZE)))
    return max(minimum, max_chunk)


def is_global_distribution() -> bool:
    """Whether the ranks distribute across the whole job, rather than within each node."""
    return int(os.environ.get("RUNAI_STREAMER_DIST_GLOBAL", "0")) == 1


def aligned_device_buffer(bytesize: int, device: torch.device) -> torch.Tensor:
    """One device buffer whose START is aligned, over-allocated so the slice still has room.

    EVERY buffer a rank receives into comes from here. An offset means the same thing in a sender's
    buffer and in a receiver's copy only while both begin on the same remainder, and torch's
    allocator is not asked to promise that."""
    raw = torch.empty(bytesize + DIST_BUFFER_ALIGNMENT, dtype=torch.uint8, device=device)
    return raw[(-raw.data_ptr()) % DIST_BUFFER_ALIGNMENT:]


def allocate_dist_staging(max_chunk: int, device: torch.device, rows: int,
                          receive_bytesize: Optional[int] = None,
                          send_buffer: bool = True) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every device allocation the broadcast loop makes: the staging buffers and two metadata tensors.

    Each staging buffer is over-allocated by DIST_BUFFER_ALIGNMENT and sliced to the first aligned
    address, so the same offset lands on the same remainder in BOTH buffers - which is what a
    receiving rank relies on to read a sender's layout.

    receive_bytesize is what the LARGEST sender will broadcast, which is the slot size where the
    ranks send their ring slots. It defaults to this rank's own staging size, which is right only
    while every rank stages.

    send_buffer is False for a rank that sends its ring slots: it has nothing to stage through, and
    the buffer would be a device allocation nothing ever writes to.

    dist_device_bytes_required() measures THIS function, and a test allocates from it and compares, so
    the free-memory check cannot disagree with the allocation about how many buffers there are."""
    buffer_bytesize = dist_buffer_bytesize(max_chunk)
    receive_bytesize = buffer_bytesize if receive_bytesize is None else receive_bytesize

    def staging(bytesize: int) -> torch.Tensor:
        return aligned_device_buffer(bytesize, device)

    def metadata() -> torch.Tensor:
        # One row per range, plus row 0 for the count.
        return torch.zeros(rows + 1, 4, dtype=torch.int64, device=device)

    send = staging(buffer_bytesize) if send_buffer else None
    return send, staging(receive_bytesize), metadata(), metadata()


def dist_device_bytes_required(max_chunk: int, rows: int) -> int:
    """What allocate_dist_staging() takes on the device WHEN THE RANKS STAGE.

    It is the gate in set_is_distributed, which runs before the group has a slot shape, so it cannot
    describe a rank that sends its ring slots: that rank allocates no send buffer, sizes its receive
    buffer from the agreed slot, and takes a device ring on top that this never counted. What bounds
    THAT rank is device_ring_budget, out of which agreed_slot_shape reserves the buffers it receives
    into - two where depth can pay for them, one where it cannot.

    So this stays the STAGED figure. On the direct path it is not even conservative: that rank
    receives into a buffer the size of the agreed slot, which is at least the largest range, while
    this counts dist_buffer_bytesize(max_chunk). The gate runs before the group has a slot shape,
    so it cannot know the real figure; what bounds that rank is device_ring_budget instead."""
    staging = dist_buffer_bytesize(max_chunk) + DIST_BUFFER_ALIGNMENT
    metadata = (rows + 1) * 4 * 8      # int64
    return 2 * staging + 2 * metadata


DEFAULT_BROADCAST_TIMEOUT = timedelta(seconds=600)

# PROCESS GROUPS ARE CACHED FOR THE LIFE OF THE PROCESS, and that is a fix, not an optimization.
# gloo's group teardown can block inside the DESTRUCTOR waiting on the peer's sockets - after a
# matched barrier, and silently, since a destructor cannot raise - and a teardown that races a peer
# has been seen to corrupt the rank's TCPStore client, after which every group the rank tries to
# create fails with "Socket Timeout" no matter how long the timeout. Creating and destroying groups
# per stream rolled those dice on every load (and on every distributed test); reusing one group per
# topology and timeout takes teardown off the healthy path entirely. The topology cannot change
# within a torch job, so a cached group never goes stale - only a POISONED one, whose stream failed,
# is evicted and retired (see release_group), because its pairs may already be closed.
#
# HIT OR MISS must be decided identically on every rank, because creating a group is a collective: a
# rank that misses while its peers hit waits for them until the timeout. The key components are
# job-global (the rank layout, the timeout env var) - and eviction, which is rank-local knowledge,
# is reconciled by the vote in cached_or_created before any entry is reused.
_topology_cache: Optional[Tuple[int, int, List[List[int]]]] = None
_group_cache: Dict[Tuple, Any] = {}

# The default process group the caches belong to. "The life of the process" above really means the
# life of the JOB: a process that destroys the default group and initializes a new one may come back
# with a different layout, and groups of the dead job must not serve the new one.
_cache_job: Any = None


def _caches_for_this_job() -> Any:
    """Reset the caches when the default process group has changed since they were filled, and
    return the current job's token.

    Called before every cache read. Comparison is by identity: init_process_group builds a new
    default group object, so a cache filled under another one belongs to a dead job. The dead job's
    groups are retired, not dropped, for the reason _retire gives. The returned token lets
    INSTANCE state - a discovered layout, a held distribution group - make the same staleness
    check, since an instance can outlive the job it was filled under just as the module can."""
    global _cache_job, _topology_cache
    current = dist.group.WORLD if dist.is_initialized() else None
    if current is _cache_job:
        return current
    for key in list(_group_cache):
        if key != ("retired",):
            _retire(_group_cache.pop(key))
    _topology_cache = None
    _cache_job = current
    return current


def _retire_discovery() -> None:
    """Retire the discovery group and forget the topology, after a vote failed on it.

    Its pairs may be closed, so keeping it would fail every later vote too. Dropping the topology
    makes the next load rerun find_local_ranks, which recreates the discovery group - a global
    collective, so every rank of the job repeats it together."""
    global _topology_cache
    discovery = _group_cache.pop(("discovery",), None)
    if discovery is not None:
        _retire(discovery)
    _topology_cache = None


def _retire(entry: Any) -> None:
    """Keep a reference to a group (or list of them) that must never be used again.

    Deliberately a leak: dropping the last reference runs gloo's destructor, which is the blocking,
    store-corrupting teardown this module exists to avoid (see _group_cache). A retired group costs
    its sockets until the process ends, and retirement only happens after a failed stream, so the
    cost is bounded by failures."""
    _group_cache.setdefault(("retired",), []).append(entry)


def write_metadata_rows(metadata: torch.Tensor, ranges: List[SlotRange]) -> None:
    """Describe a filled slot for the ranks that will receive it.

    The row is [file, index, SIZE, OFFSET], which is not SlotRange's field order: broadcast() reads
    the size from column 2 and the offset from column 3. Row 0 carries the count.
    """
    if len(ranges) + 1 > metadata.shape[0]:
        # Raised before anything is written. Half a row set is worse than none: the collective is
        # posted either way, and the receivers decode whatever got there.
        raise ValueError(
            f"{len(ranges)} ranges need {len(ranges) + 1} metadata rows, and this tensor has "
            f"{metadata.shape[0]}"
        )

    metadata[0, 0] = len(ranges)
    if ranges:
        # ONE TENSOR FOR THE WHOLE BLOCK. A tensor built per row is a construction and a copy each
        # time, and a slot can carry tens of thousands of rows - and on a device metadata tensor
        # each of those copies is its own transfer.
        rows = torch.tensor([(r.file_id, r.range_index, r.size, r.offset) for r in ranges],
                            dtype=metadata.dtype)
        metadata[1:len(ranges) + 1].copy_(rows)


class AgreedRing(NamedTuple):
    """What the group agreed to build on every rank.

    buffer_size is the only part every rank must match: it is the broadcast unit and an input to the
    layout each rank computes for its peers. Depth and the receive count are local, and a rank that
    cannot afford what was agreed may lower either on its own."""

    buffer_size: int
    num_buffers: int
    receive_buffers: int

    @property
    def shape(self) -> Tuple[int, int]:
        """What stream_files takes: the ring, without the buffers received into."""
        return (self.buffer_size, self.num_buffers)


def agreed_slot_shape(min_free_bytes: Optional[int],
                      file_stream_requests: List[FileChunks],
                      bounds: SlotBounds = SENT_RING_BOUNDS) -> Optional[AgreedRing]:
    """The ring every rank of the group will send from, with what it receives into, or None when no
    ring fits.

    NO COLLECTIVE OF ITS OWN. It is a pure function of values that are already equal on every rank:
    the smallest free memory in the group, which get_min_device_free_memory has agreed, and the
    whole file list, which is the caller's request. Agreement comes from identical inputs, not from
    another reduction - and a second reduction would be a second place for the ranks to disagree.

    THE WHOLE LIST, never this rank's share. A share is floored by the largest range IN IT, so two
    ranks would answer differently. And the size is what a RECEIVER allocates too, so it has to hold
    the largest range any sender will broadcast - which only the whole list knows.

    THE SAME LIST. Not in the same order, any more: the size comes from the largest range and the
    total, and neither moves when the list is shuffled. It used to be packed in list order to score
    a candidate depth, which made ORDER part of the contract between ranks.

    NO CALLER-NAMED DEPTH anywhere on this path. A rank that chose its own would wait on
    collectives its peers never post, and the device ring no longer takes one from anybody: its
    depth is the two slots that let reading and sending overlap.

    EVERY INPUT IS READ PER RANK, and two of them still move the answer: the device memory limit
    and its fraction, which only ever LOWER what the data asked for. A rank given different ones
    sizes differently, which is what check_the_group_agreed is for.

    min_free_bytes may be None, where no agreement was reached - a group whose device torch cannot
    measure, or one not using a device backend at all. There is then no ring to agree on, and the
    answer is None.

    ONE SLOT OF THE BUDGET IS NOT THE RING'S. A rank that sends its slots also receives its peers',
    into a buffer of the same size, and get_chunks allocates that AFTER stream_files has already
    taken the ring. Sizing the ring to the whole budget would leave nothing for it, and the failure
    would be a raw allocation error on some ranks while the others wait in the collective.

    THE SLOT PAYS FOR IT, down to the largest range, and the depth only when the slot cannot. Two
    slots is the least that can overlap - the library refills a slot only once the collective has
    handed it back - and losing that is the one cliff both models measure, while a slot somewhat
    below what the data asked for is flat over a wide range. Shrinking is safe because every rank
    computes this from the same numbers and so shrinks together; what would break a peer is one
    rank alone choosing a different size.

    A one slot ring is still right when the stream fits inside it - there is nothing to overlap
    there - and it is still better than the host path when it does not. The host path copies every
    range, so what a single slot costs is the overlap, not the device ring. A caller that asks for a
    depth of 1 is taken at its word.
    """
    # TWO TO RECEIVE INTO, so a transfer lands while the clones out of the one before it still run.
    # Asked for as part of the shape, not reserved out of it afterwards: a reservation that shrank
    # a shape already chosen is a second solver, and the two disagree.
    for receive_buffers in (2, 1):
        shape = device_ring_sizing(min_free_bytes, file_stream_requests, bounds,
                                   extra_slots=receive_buffers)
        if shape is not None:
            return AgreedRing(shape[0], shape[1], receive_buffers)

    # Not even one slot and somewhere to receive into. Every rank reaches that from the same
    # numbers, so the whole group takes the host path together.
    return None


def get_backends() -> dict:
    """Map each device type of the default process group to the backend that serves it.

    dist.get_backend() returns the string given to init_process_group, so a device-prefixed one such as
    "cuda:nccl,cpu:gloo" never equals "nccl". dist.get_backend_config() (torch 2.0 and later) always
    uses the "device:backend[,device:backend]" form, whichever form was given.
    """
    return dict(entry.split(":", 1) for entry in dist.get_backend_config().split(","))


def get_backend_for_device(device: str) -> Optional[str]:
    """The backend of the default process group that serves device, or None if no backend does."""
    return get_backends().get(str(device).split(":", 1)[0])

class DistributedStreamer:
    def __init__(self) -> None:
        self.file_streamer = FileStreamer()
        self.params = _distributedStreamerParams()
        self.distributed_streamer = _distributedStreamer(self.file_streamer)
        self.is_distributed = False
        self._ring_info: Optional[Tuple[int, int, int, str]] = None

    def __enter__(self) -> DistributedStreamer:
        self.file_streamer.__enter__()
        self.distributed_streamer.__enter__()
        return self

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        try:
            self.distributed_streamer.__exit__(exc_type, exc_value, traceback)
        finally:
            # Ensure the second __exit__ is always called
            self.file_streamer.__exit__(exc_type, exc_value, traceback)
        return
    
    def get_group_size(self) -> int:
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    def get_device_free_memory(self, device: str) -> int:
        """Free bytes on this rank's device.

        Raises rather than returning "unknown", so the caller cannot put a non number into its
        reduction by accident. The caller catches this and reduces a sentinel instead - leaving the
        reduction is what strands the peers."""
        free = device_free_memory(device)
        if free is None:
            raise RuntimeError(
                f"torch cannot measure free memory on {device}, so this rank cannot enter the "
                f"reduction - can_measure_device_memory() must be checked first"
            )
        return free

    def get_min_device_free_memory(self, device: str) -> int:
        """The smallest free memory among the ranks of the distribution group: every rank under
        RUNAI_STREAMER_DIST_GLOBAL, otherwise the ranks on this node.

        The group is created here, by every rank, before anyone decides. That is what lets a node fall
        back while another distributes: creating a group is the only collective that needs every rank
        of the job, and everything after it runs inside the group. Every rank must call this, whatever
        its own free memory is."""
        group = self.distributed_streamer.create_group(self.params)

        # A RANK THAT CANNOT MEASURE STILL REDUCES, with -1. Raising here instead would leave its
        # peers in the all_reduce below until the group timeout, and they would wait for a reading
        # that is never coming. Free memory is never negative, so nothing else produces this value.
        try:
            mine = self.get_device_free_memory(device)
        except Exception as error:
            logger.warning(
                "[RunAI Streamer][Distributed] Warning: could not read the free memory on %s: %s",
                device, error)
            mine = -1

        # This is the first collective on the group, so it creates the NCCL communicator, and the
        # reading above does not yet show that memory - about 634 MiB on a B200.
        #
        # LEFT UNCORRECTED, and the headroom is in the two readers of this figure, not here. A rank
        # that sends its ring slots is bounded by device_ring_budget, which takes half of what this
        # returns. A rank that stages is bounded by the gate in set_is_distributed, whose threshold
        # starts at two staging buffers - see dist_device_bytes_required - so it carries gibibytes
        # of slack against those megabytes. A warmup collective here would make the reading exact
        # and buy nothing either reader needs, while adding one more place for a group to stall.
        free_memory = torch.tensor([mine], dtype=torch.int64, device=device)
        dist.all_reduce(free_memory, op=dist.ReduceOp.MIN, group=group)
        smallest = int(free_memory.item())

        # TOGETHER, and not as a fallback. A device torch cannot read is a device the caller cannot
        # load on either, so there is nothing to fall back to - and every rank raises here, at the
        # same point, rather than one of them leaving the others behind. The rank that failed has
        # the cause in its own log.
        if smallest < 0:
            raise RuntimeError(
                "a rank of this distribution group could not read the free memory on its device - "
                "see that rank's log for the cause"
            )
        return smallest

    def set_is_distributed(self, is_distributed: bool, device: str) -> None:
        # check if distributed streaming should be used

        # CLEARED FIRST, because this object is reused across loads. A figure measured before the
        # last model occupied the device would size a ring far too large, and nothing downstream
        # can tell a stale reading from a fresh one.
        self.params.min_device_free_memory = None

        if not is_distributed:
            self.is_distributed = False
            return

        # environment variable to override default distributed streaming
        # by default, use distributed streaming only for the nccl backend
        enable_dist = os.environ.get("RUNAI_STREAMER_DIST", "auto")
        if enable_dist == "0":
            self.is_distributed = False
        elif enable_dist == "1":
            self.is_distributed = True
        elif enable_dist == "auto":
            self.is_distributed = True
        else:
            raise ValueError(f"Invalid value for RUNAI_STREAMER_DIST: {enable_dist}")

        # check if torch distributed is initialized and there are more than one process
        if self.is_distributed:
            self.is_distributed = dist.is_initialized()
            if not self.is_distributed:
                logger.info("[RunAI Streamer][Distributed] Torch distributed is not initialized - fallback to non distributed streaming")
            self.is_distributed = self.is_distributed and self.get_group_size() > 1

        # Do not distribute if backend type does not match device type
        # In auto mode, do not distribute if backend is not nccl
        backend_name = None
        if self.is_distributed:
            backend_name = get_backend_for_device(device)
            if backend_name is None:
                logger.info("[RunAI Streamer][Distributed] Note: Torch distributed has no backend for %s device (%s) - fallback to non distributed streaming", device, dist.get_backend_config())
                self.is_distributed = False
            elif backend_name == "nccl" and device == "cpu":
                # torch 2.0 lists a plain nccl group under cpu as well, but nccl cannot broadcast CPU tensors
                logger.info("[RunAI Streamer][Distributed] Note: Torch distributed backend %s is not supported for CPU device - fallback to non distributed streaming", backend_name)
                self.is_distributed = False
            elif backend_name == "gloo" and device != "cpu":
                logger.info("[RunAI Streamer][Distributed] Note: Torch distributed backend %s is not supported for %s device - fallback to non distributed streaming", backend_name, device)
                self.is_distributed = False
            elif backend_name == "xccl" and torch.device(device).type != "xpu":
                logger.info("[RunAI Streamer][Distributed] Note: Torch distributed backend %s is not supported for %s device - fallback to non distributed streaming", backend_name, device)
                self.is_distributed = False
            elif enable_dist == "auto" and backend_name not in ("nccl", "xccl"):
                logger.info("[RunAI Streamer][Distributed] Note: Torch distributed backend %s is not supported by default for distributed streaming - To allow this backend, set RUNAI_STREAMER_DIST to `1`", backend_name)
                self.is_distributed = False

        # check if there is enough free memory on the device
        # Gated on whether torch can MEASURE this device, not on whether the streamer can read into
        # it: distributing moves torch tensors, so it has no such limit. mem_get_info is missing for
        # several devices torch otherwise supports, and skipping the check there is right - refusing
        # would fall back from a device that may well have the room.
        #
        # EVERY RANK TAKES THIS BRANCH OR NONE DOES, which the collectives below need. The ranks of
        # a distributed job run one kind of device under one backend, so both terms are facts about
        # the JOB - there is nothing here for the ranks to agree at runtime.
        if self.is_distributed and can_measure_device_memory(device) and backend_name in ("nccl", "xccl"):
            free_memory = self.get_min_device_free_memory(device)
            # KEPT, not only compared. It is the one free-memory figure the group agrees on, and a
            # ring they all send from has to be sized from it rather than from each rank's own.
            self.params.min_device_free_memory = free_memory
            required = dist_device_bytes_required(self.params.max_chunk, self.params.metadata_rows)
            if free_memory < required:
                logger.warning(f"[RunAI Streamer][Distributed] Warning: Not enough memory on the device for distributed streaming - fallback to non distributed streaming, free memory: {free_memory} bytes, required minimun: {required} bytes")
                self.is_distributed = False
                # THE GROUP IS KEPT, although this rank will not stream through it. Creating one needs
                # every rank of the JOB, while this decision is taken per node - so a node that
                # released would ask for a new group alone on the next read, and block there until the
                # timeout. __exit__ releases it, where every rank does so together.

    def stream_files(
            self,
            file_stream_requests: List[FileChunks],
            credentials: Optional[S3Credentials],
            device: str,
            is_distributed: bool,
            owned: bool = True,
            discover_ranks: bool = True,
            ring_buffers: Optional[int] = None,
    ) -> None:
        """discover_ranks=False skips the discovery of the local ranks, a collective over all the ranks of the process
        group, for a read that runs without its peers. Distributed streaming discovers them regardless."""

        # Normalised HERE so no spelling of the host reaches a comparison against "cpu": those decide
        # whether a buffer is handed over and whether this host can broadcast, and None or
        # torch.device("cpu") silently answered no to both.
        device = host_or_device(device)

        # Cleared before dispatch, so a call that builds no ring - or raises before it does - reports
        # nothing rather than whatever the previous call left behind.
        self._ring_info = None

        self.params.set_params(file_stream_requests, discover_ranks or is_distributed)

        # check if distributed streaming can be used
        self.set_is_distributed(is_distributed, device)

        if not self.is_distributed:
            # Passed on as asked. Whether a device destination lets the ring be handed over is the
            # FileStreamer's to decide, because that is where the tensor is copied to the device.
            self.file_streamer.stream_files(
                file_stream_requests, credentials, device, owned=owned, ring_buffers=ring_buffers
            )
            built_ring = True
        else:
            self.distributed_streamer.stream_files(file_stream_requests, credentials, device,
                                                   self.params, owned=owned,
                                                   ring_buffers=ring_buffers)
            # A rank whose share of the partition is empty returns before building anything, leaving the
            # iterator from the safetensors metadata read in place. Reporting THAT as the model ring is
            # worse than reporting nothing: `1 x 8 Bytes` against the model's byte total reads exactly
            # like a ring clamped to a single buffer, which is the failure this report exists to expose.
            built_ring = self.distributed_streamer.reading_from_storage

        requests_iterator = getattr(self.file_streamer, "requests_iterator", None)
        if built_ring and requests_iterator is not None:
            self._ring_info = (
                self.params.my_global_rank,
                requests_iterator.num_buffers,
                requests_iterator.buffer_size,
                # What the slots are MADE of. A device ring and a host ring of the same shape read
                # very differently, and the line reporting them cannot tell them apart otherwise.
                requests_iterator.slots.label,
            )

    def ring_info(self) -> Optional[Tuple[int, int, int, str]]:
        """(rank, num_buffers, buffer_size, label) for the ring the last stream_files built, or
        None if it built none.

        Captured during stream_files rather than read from the streamer on demand. The FileStreamer is
        shared with the safetensors metadata reads, so its requests_iterator outlives the call that
        created it: asking it later answers "the last ring anyone built", which is a different question
        and gives a wrong answer on exactly the rank that read no model data.

        An accessor rather than letting callers walk into the streamer, because which layer performs the
        read depends on is_distributed. The rank belongs here too: it is the piece FileStreamer cannot
        know, which is why the ring is reported at INFO from the session boundary (SafetensorsStreamer)
        and only at DEBUG where it is built.

        None is normal, not an error: a rank whose share of the partition is empty reads nothing, and
        list_files-only use never streams at all."""
        return self._ring_info

    def get_chunks(self) -> Iterator:
        if not self.file_streamer:
            raise ValueError("Streamer not initialized")
        
        if not self.is_distributed:
            for item in self.file_streamer.get_chunks():
                yield item
        else:
           for item in self.distributed_streamer.get_chunks():
                yield item
        return

class _distributedStreamerParams:
    def __init__(self) -> None:
        self.num_processes_on_node = None
        # The job the layout below was discovered under. An instance can be reused across
        # init_process_group cycles, and the new job's layout may differ, so the layout is
        # rediscovered when the token no longer matches (see discover_local_ranks).
        self.discovered_job = None
        self.my_global_rank = 0
        self.groups_by_ranks = []
        self.broadcast_timeout = DEFAULT_BROADCAST_TIMEOUT
        self.max_chunk = 0
        # One row per range of the request, set by set_params. One row until then, which is the
        # smallest a metadata tensor can be - a request with nothing in it.
        self.metadata_rows = 1
        # The smallest free memory in the group, once it has been agreed; None where no agreement
        # was needed. A ring every rank sends from is sized from THIS, not from what this rank
        # happens to have free.
        self.min_device_free_memory = None
 
    def get_group_size(self) -> int:
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    def get_broadcast_timeout(self) -> timedelta:
        timeout_val = os.environ.get("RUNAI_STREAMER_DIST_TIMEOUT")
        if timeout_val:
            return timedelta(seconds=int(timeout_val))
        else:
            return DEFAULT_BROADCAST_TIMEOUT

    def find_local_ranks(self) -> Tuple[int, int, List[List[int]]]:
        """
        Returns the number of processes on the node, global rank and the list of ranks on each node

        The answer cannot change within a torch job, so it is computed once per process and cached.

        The gather runs on a dedicated gloo subgroup rather than the global group: hostnames travel
        through host memory there, so no backend allocates device memory that the user application
        may never have allocated itself. The subgroup is KEPT, not destroyed - it costs a few
        sockets, and tearing a gloo group down is what _group_cache's comment warns about.
        """
        global _topology_cache
        if not dist.is_initialized():
            return 1, 0, [[0]]

        world_size = dist.get_world_size()

        if world_size == 1:
            return 1, 0, [[0]]

        _caches_for_this_job()
        if _topology_cache is not None:
            return _topology_cache

        my_global_rank = dist.get_rank()

        # 1. Discover all peers on all nodes (this is a global collective)
        #
        # THE CONSTANT TIMEOUT, not this load's broadcast_timeout: the group is kept and serves
        # every later load's cache vote, so its patience must not be whatever the FIRST load's
        # environment happened to say - a first load configured for seconds would make every later
        # vote give up on a peer the later load's own timeout still allows for.
        discovery_group = dist.new_group(ranks=list(range(world_size)),
                                         timeout=DEFAULT_BROADCAST_TIMEOUT, backend="gloo")
        _group_cache[("discovery",)] = discovery_group
        all_hostnames = [None] * world_size
        dist.all_gather_object(all_hostnames, socket.gethostname(), group=discovery_group)

        # 2. Create a list of rank lists, one for each unique host.
        # e.g., [[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15]]
        unique_hostnames = sorted(list(set(all_hostnames)))
        groups_by_ranks = []
        for hostname in unique_hostnames:
            ranks_on_host = [r for r, h in enumerate(all_hostnames) if h == hostname]
            groups_by_ranks.append(ranks_on_host)

        my_group_size = 1
        for i, ranks_list in enumerate(groups_by_ranks):
            if my_global_rank in ranks_list:
                my_group_size = len(ranks_list)
                break

        _topology_cache = (my_group_size, my_global_rank, groups_by_ranks)
        return _topology_cache

    def set_params(
            self,
            file_stream_requests: List[FileChunks], 
            discover_ranks: bool,
    ) -> None:

        self.broadcast_timeout = self.get_broadcast_timeout()

        # find the size of the maximal chunk for the reusable buffer
        max_chunks_per_file = (fc.max_chunk_size() for fc in file_stream_requests if fc.sizes)
        self.max_chunk = max(max_chunks_per_file, default=0)

        # Held beside max_chunk, because the two together are what the staging allocation costs.
        self.metadata_rows = metadata_rows(file_stream_requests)

        if discover_ranks:
            self.discover_local_ranks()
        else:
            self.set_rank_without_discovery()

    def discover_local_ranks(self) -> None:
        # set the value of RUNAI_STREAMER_PROCESS_GROUP_SIZE to be the number of processes in the distribution group (or 1 if process group is not initialized)
        # This is useful for auto adjustments in the CPP layer, which depends on the number of processes of this workload
        current_job = _caches_for_this_job()
        if self.discovered_job is not current_job or _topology_cache is None:
            # Two ways the instance's layout can be stale. The job changed: the instance outlived
            # init_process_group, and the new job's layout may differ. Or the MODULE dropped its
            # topology (_retire_discovery, after a failed vote) while this instance kept its copy:
            # serving that copy would skip find_local_ranks, the discovery group would never be
            # recreated, and the next load's cache vote would silently not happen - reopening the
            # asymmetric-eviction hole the vote exists to close. Rediscovering is cheap when the
            # module cache is warm: find_local_ranks answers from it without a collective.
            self.num_processes_on_node = None
        if self.num_processes_on_node is None:
            self.num_processes_on_node, self.my_global_rank, self.groups_by_ranks = self.find_local_ranks()
            self.discovered_job = current_job
        if os.environ.get("RUNAI_STREAMER_PROCESS_GROUP_SIZE") is None:
            os.environ["RUNAI_STREAMER_PROCESS_GROUP_SIZE"] = str(self.num_processes_on_node)

    def set_rank_without_discovery(self) -> None:
        # RUNAI_STREAMER_PROCESS_GROUP_SIZE is not guessed, so the CPP layer defaults to 1: a count larger than the readers lowers their throughput
        if dist.is_initialized():
            self.my_global_rank = dist.get_rank()

class _distributedStreamer:
    # What every rank will send, per rank and per turn. None means the metadata still travels,
    # which is the safe answer: it is what a stream falls back to, and what any path that never
    # sized a ring to send from gets.
    predicted: Optional[List[List[List[SlotRange]]]] = None

    def __init__(self, file_streamer : FileStreamer) -> None:
        self.file_streamer = file_streamer
        self.device = None
        self.sends_directly = False
        self.slot_shape = None
        self.receive_buffers = 0
        self.rank_file_chunks_list = {} # post Partitioning FileChunks to be streamed by this rank
        self.rank_dicts_map = {} # maps post partitioning FileChunks.id and chunk index to the original FileChunks.id and chunk index and chunk size
        self.total_chunks_to_read = 0
        self.group_size = 1
        self.rank = 0
        self.original_group_rank = 0
        self.distribution_group = None
        # The job the held group belongs to; a handle held across init_process_group cycles is the
        # dead job's and must not be returned for the new one (see create_distribution_group).
        self.group_job = None
        self.reading_from_storage = False
        self.local_group_global_ranks = []
        self.is_error = False
        self.my_global_rank = 0
        self.groups_by_ranks = []
        self.broadcast_timeout = DEFAULT_BROADCAST_TIMEOUT
        self.num_processes_on_node = 1
        self.max_chunk = 0
        self.owned = True

    def __enter__(self) -> _distributedStreamer:
        return self

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        if self.distribution_group:
            # A failed stream poisons the group: a timed-out collective closes gloo pairs, so the
            # next stream must not inherit it from the cache. The barrier's own failure counts.
            poisoned = exc_type is not None
            try:
                # Only attempt a barrier if there was no exception.
                if not poisoned:
                    torch.distributed.barrier(group=self.distribution_group)
            except BaseException:
                poisoned = True
                raise
            finally:
                # ALWAYS let go of the group; evict and retire it only when it is poisoned.
                self.release_group(poisoned=poisoned)
        return

    def create_distribution_group(self) -> Optional[dist.ProcessGroup]:
        current_job = _caches_for_this_job()
        if self.distribution_group and self.group_job is current_job:
            return self.distribution_group
        # Either there is no group, or the held one belongs to a job that is gone - a reused
        # instance can outlive init_process_group. The dead job's handle is dropped, not retired:
        # the module cache held the reference, and _caches_for_this_job already retired it.
        self.distribution_group = None
        self.group_job = current_job

        if is_global_distribution():
            # create global group
            world_size = dist.get_world_size()

            all_ranks = list(range(world_size))
            self.local_group_global_ranks = all_ranks
            key = ("global", world_size, self.broadcast_timeout)
            group = self.cached_or_created(
                key, lambda: dist.new_group(ranks = all_ranks, timeout = self.broadcast_timeout))

        else:
            group = self.create_local_distribution_group()

        logger.debug(f"[RunAI Streamer][Distributed] Created distribution group with size {dist.get_world_size(group=group)}")
        return group

    @staticmethod
    def cached_or_created(key: Tuple, create) -> Any:
        """The cached entry for key, or a fresh one from create() - AGREED ACROSS THE JOB.

        Hit or miss is voted, not read locally. A rank whose stream failed evicted its entry
        (release_group), while the ranks whose streams succeeded kept theirs - in per-node mode a
        failure on one node never reaches the others - and creating a group needs every rank of the
        job, so a rank that recreates alone waits for peers that never come. One MIN over the kept
        discovery group settles it: everyone reuses, or everyone retires its entry and recreates.
        The vote travels through host memory on gloo, so it allocates nothing on any device.
        """
        _caches_for_this_job()
        discovery = _group_cache.get(("discovery",))
        have = key in _group_cache
        if discovery is not None:
            # ON THE CPU EXPLICITLY: torch.set_default_device would otherwise put the vote on the
            # application's device, and the gloo discovery group cannot reduce a device tensor.
            vote = torch.tensor([1 if have else 0], dtype=torch.int64, device="cpu")
            try:
                dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=discovery)
            except BaseException:
                # The vote itself failed, so the discovery group's pairs may be closed, and every
                # later vote would fail against it. Retire it and drop the topology: the next load
                # rediscovers through find_local_ranks, a global collective every rank repeats
                # together - the same symmetry argument as release_group's.
                _retire_discovery()
                raise
            have = bool(vote.item())
        if have:
            return _group_cache[key]
        stale = _group_cache.pop(key, None)
        if stale is not None:
            _retire(stale)
        created = create()
        _group_cache[key] = created
        return created

    def create_local_distribution_group(self) -> Optional[dist.ProcessGroup]:
        """
        Creates a torch.distributed.ProcessGroup containing all ranks on the current node.
        This version uses a coordinated creation pattern to avoid deadlocks.

        This function performs all_gather_object on the global group which cause nccl to allocate  device memory which otherwise may never be allocated by the user application
        To avoid this issue, we create a another global subgroup just for discovering the local ranks and then destroy that subgroup after the ranks are discovered.

        Note: find_local_ranks() must be called before this function
        """

        if not dist.is_initialized():
            return None

        # All processes must create ALL subgroups in the same order.
        # This is a global collective operation, done once for each subgroup.
        key = ("local", tuple(tuple(ranks) for ranks in self.groups_by_ranks), self.broadcast_timeout)
        created_groups = self.cached_or_created(
            key,
            lambda: [dist.new_group(ranks=ranks, timeout=self.broadcast_timeout)
                     for ranks in self.groups_by_ranks])

        # Each process finds which of the newly created groups it belongs to.
        my_local_group = None
        for i, ranks_list in enumerate(self.groups_by_ranks):
            if self.my_global_rank in ranks_list:
                my_local_group = created_groups[i]
                self.local_group_global_ranks = ranks_list # Save the mapping
                break

        return my_local_group

    def create_group(self, params: _distributedStreamerParams) -> dist.ProcessGroup:
        """The distribution group, created on first use and kept. Creating it needs every rank of the
        job, so every rank must call this together."""
        self.my_global_rank = params.my_global_rank
        self.groups_by_ranks = params.groups_by_ranks
        self.broadcast_timeout = params.broadcast_timeout
        self.distribution_group = self.create_distribution_group()
        return self.distribution_group

    def release_group(self, poisoned: bool = False) -> None:
        """Let go of the distribution group, if there is one. Local to this rank: no barrier.

        The group is NOT destroyed in either case: it stays in _group_cache for the next stream,
        because tearing a gloo group down is the hazard that cache exists to avoid (see its
        comment). poisoned=True is the error path - the group's pairs may already be closed, so it
        is evicted and retired rather than handed to the next stream. The eviction is LOCAL
        knowledge: in per-node mode the other nodes' streams may have succeeded and kept their
        entries, which is why the next stream VOTES before reusing the cache (cached_or_created) -
        this rank's missing entry makes every rank recreate together."""
        group = self.distribution_group
        self.distribution_group = None
        if group is None or not poisoned:
            return
        for key, cached in list(_group_cache.items()):
            if key == ("retired",):
                continue
            if cached is group or (isinstance(cached, list) and group in cached):
                del _group_cache[key]
                _retire(cached)

    def set_rank(self) -> None:
        self.rank = dist.get_rank(group=self.distribution_group)
        self.original_group_rank = dist.get_rank()

    def stream_files(
            self,
            file_stream_requests: List[FileChunks],
            credentials: Optional[S3Credentials],
            device: str,
            params : _distributedStreamerParams,
            owned: bool,
            ring_buffers: Optional[int] = None,
    ) -> None:
        """ring_buffers is the caller's depth. It reaches the HOST ring only.

        The agreed shape ignores it. That shape decides whether the group sends ring slots or stages
        through host memory, and the two post a different number of collectives per round, so a rank
        that answered differently would wait for messages its peers never send.

        owned=False hands the caller a view into a buffer the ring refills. It holds only while the
        caller uses each tensor before asking for the next, and keeps to one stream: the event that
        allows a refill is recorded when the caller asks again, so a tensor kept past that point can
        change under it."""

        # Resolved once, so the staging buffers and the stream an event is recorded on cannot
        # drift onto different devices.
        self.device = resolve_device(device)
        self.owned = owned
        self.max_chunk = params.max_chunk
        self.metadata_rows = params.metadata_rows
        # the ranks sharing this host: the divisor for the node-total memory limit (see rank_memory_limit)
        self.num_processes_on_node = params.num_processes_on_node

        # create distribution group for configuring timeout and possibly broadcast locally in each node 
        # The group must be created before partitioning the tensors, in case the group will be local
        self.create_group(params)
        self.group_size = dist.get_world_size(group=self.distribution_group)

        # set rank in the new distribution group
        self.set_rank()

        if self.original_group_rank == 0:
            logger.info(f"[RunAI Streamer][Distributed] Using distributed streaming between {self.group_size} processes")

        # Partition tensors between processes in the distribution group.
        #
        # Only THIS rank's share is built where the policy allows it. Every rank used to build all of
        # them and discard the rest, which on a 150k-tensor model is most of the work: 184 ms against
        # 113 ms for one share, and 702 ms for the `chunks` policy that does it the old way.
        #
        # The two things the discarded shares were needed for come back as plain numbers:
        # sizes_by_rank for the log, and total_chunks for the broadcast countdown.
        my_span = partition_for_rank(
            file_stream_requests,
            dist.get_world_size(group=self.distribution_group),
            self.rank,
        )

        if self.rank == 0:
            log_partition_info(my_span.sizes_by_rank)

        # GLOBAL, not this rank's share. The broadcast loop counts it down as chunks are sent AND
        # received, so a rank that counted only its own would stop while others were still sending.
        self.total_chunks_to_read = my_span.total_chunks

        # read partition
        self.rank_file_chunks_list = []

        # map partitions's file chunks index to the index in the original file list
        self.rank_dicts_map = {}
        for fc, d in my_span.partition:
            self.rank_file_chunks_list.append(fc)
            self.rank_dicts_map[fc.id] = d

        log_rank_span_info(self.original_group_rank, my_span.partition)

        # Send the ring's own slots where the library can fill them, and stage through a host ring
        # where it cannot. agreed_slot_shape is a pure function of the free memory the group already
        # reduced and of the WHOLE file list, so every rank should answer the same - and the check
        # below is there because "should" is not "does".
        #
        # THE CALLER'S DEPTH IS NOT HONOURED HERE, and that is the point. The shape decides which
        # path the whole group takes, so a rank that answered differently would wait for collectives
        # its peers never post. Depth is the one input a caller can set per rank, and it buys little
        # past a few slots, so the group agrees without asking.
        agreed = agreed_slot_shape(params.min_device_free_memory, file_stream_requests)
        self.slot_shape = agreed.shape if agreed else None
        self.receive_buffers = agreed.receive_buffers if agreed else 0
        self.sends_directly = (
            device_support(self.device) is not None and self.slot_shape is not None
        )
        self.check_the_group_agreed()

        # A device that COULD be filled, with no shape the group can send from. Said once, by one
        # rank, because every rank reaches it from the same numbers - and said at all because the
        # stream otherwise stages through host memory with nothing to show why.
        if (self.original_group_rank == 0
                and not self.sends_directly
                and device_support(self.device) is not None):
            if params.min_device_free_memory is None:
                why = "the group agreed no free memory figure to size one from"
            else:
                budget = device_ring_budget(params.min_device_free_memory, file_stream_requests)
                largest = device_ring_floor(file_stream_requests) // 2
                why = (f"{humanize.naturalsize(budget, binary=True)} does not hold a slot to send, "
                       f"one to receive and the largest range "
                       f"({humanize.naturalsize(largest, binary=True)})")
            logger.info(
                f"[RunAI Streamer][Distributed] no ring fits on {self.device} that a slot can be "
                f"broadcast from: {why} - staging through host memory instead"
            )

        # WHAT EVERY RANK WILL SEND, worked out instead of received. A receiver knows the ranges and
        # where they sit before anything arrives, so the batch metadata never has to travel.
        #
        # NO COLLECTIVE, and this is a REQUIREMENT, not a convenience: every rank must be given the
        # same file list, in the same order, and the same streamer settings. That is what makes the
        # answer here already equal everywhere. Agreeing it over the wire would add a wait to a loop
        # whose whole point is to remove one, and a second place for the ranks to disagree.
        #
        # BEFORE the empty-share return below. A rank with nothing to read still receives from its
        # peers, so it needs the prediction as much as a rank that reads.
        self.predicted = None
        if self.sends_directly:
            # ONE PASS over the file list, not one per rank. Working a share out reads every
            # tensor of the model, and the cut points are the same for everyone, so asking rank by
            # rank would repeat the whole pass for an answer already in hand.
            shares = partition(file_stream_requests, self.group_size)

            # WHOLE PLANS, not streams. The loop that sends them posts one collective and consumes
            # the one before it, so it has to know how many turns there are before it starts.
            self.predicted = [
                list(predicted_slots(file_stream_requests, self.group_size, rank,
                                     self.slot_shape[0], SENT_RING_BOUNDS.max_ranges,
                                     shares[rank]))
                for rank in range(self.group_size)
            ]

        if len(self.rank_file_chunks_list) == 0:
            self.reading_from_storage = False
            logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: No chunks to read from storage")
            return

        # read files
        #
        # owned=False whatever the caller asked for: this ring is internal. Nobody outside holds a
        # slot - a staged source copies each range out of it, and a sent slot is broadcast and then
        # cloned - so giving one away would only buy a fresh allocation per submission. Ownership on
        # both paths is the per tensor clone the loop makes.
        if self.sends_directly:
            # memory_limit is not passed: it caps HOST memory, and this ring is on the device. The
            # shape is what bounds it, and sent_from refuses a ring the library cannot write into.
            #
            # The same max_ranges_per_slot the metadata tensor has rows for. The ring packs by
            # bytes, so without it a slot of small tensors holds more ranges than a peer can be told
            # about, and write_metadata_rows refuses the batch.
            shallower = None
            try:
                self.file_streamer.stream_files(
                    self.rank_file_chunks_list, credentials, str(self.device),
                    owned=False, slot_shape=self.slot_shape, sent_from=True,
                    max_ranges_per_slot=SENT_RING_BOUNDS.max_ranges,
                )
            except RunaiStreamerMemoryLimitException as error:
                # SHALLOWER, never smaller, and never a host ring. The slot size is what the group
                # broadcasts, so a rank that changed it would send something its peers cannot
                # receive, and a host ring cannot be broadcast from at all - NCCL does not send host
                # memory. Depth is the only part of the shape this rank may decide alone.
                #
                # The shape was sized from the group's MINIMUM free memory, so reaching here means
                # something took device memory on this rank after the group agreed. The broadcasts
                # are synchronous, so the overlap this gives up slows every peer, not just this one.
                # ONLY THE RING'S OWN OOM. That exception also carries a dictated shape this
                # layer refused outright, and a packing failure raised after the submissions are
                # already live - retrying either asks for the same refusal again, and the second
                # trips the streamer's "a stream is still live" guard, which would replace the real
                # diagnosis. The allocator's error is what a shallower ring can answer.
                if not isinstance(error.__cause__, torch.cuda.OutOfMemoryError):
                    raise

                if self.slot_shape[1] <= 1:
                    raise

                shallower = (self.slot_shape[0], 1)

            # OUTSIDE the handler, so the attempt that failed has let go of its memory. The pool
            # allocates every slot up front, and while an exception is being handled its traceback
            # keeps the slots it did manage to take - retrying in there asks for one more on top of
            # them, and can fail where this succeeds.
            if shallower is not None:
                self.slot_shape = shallower
                logger.warning(
                    f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: not enough "
                    f"device memory for the ring the group agreed - retrying with one slot of "
                    f"{humanize.naturalsize(self.slot_shape[0], binary=True)}, which reads and "
                    f"sends in turn instead of together"
                )
                self.file_streamer.stream_files(
                    self.rank_file_chunks_list, credentials, str(self.device),
                    owned=False, slot_shape=self.slot_shape, sent_from=True,
                    max_ranges_per_slot=SENT_RING_BOUNDS.max_ranges,
                )
        else:
            # The caller's depth reaches the host ring and only the host ring. The device path
            # ignores it on purpose - its shape is agreed across the group - so this is the one
            # call that forwards it.
            self.file_streamer.stream_files(
                self.rank_file_chunks_list, credentials, "cpu",
                memory_limit=self.rank_memory_limit(), owned=False,
                ring_buffers=ring_buffers,
            )
        self.reading_from_storage = True

    def check_the_group_agreed(self) -> None:
        """Every rank packs to the same slot size, or they read each other at the wrong offsets.

        The inputs are meant to be equal everywhere, but some are read per rank - the ring depth
        setting, the device memory limit and its fraction - and any of them moves the slot size. A
        rank that packed differently is not an error anyone would see: the load finishes and the
        weights are wrong.

        Zero where this rank stages through host memory, so a group that split between the two
        paths is caught here rather than in a collective its peers never post.

        ONE COLLECTIVE PER STREAM, carrying the value and its negation so a single MAX answers both
        ends."""
        if self.distribution_group is None:
            return

        size = self.slot_shape[0] if self.sends_directly else 0
        both = torch.tensor([size, -size], dtype=torch.int64, device=self.device)
        dist.all_reduce(both, op=dist.ReduceOp.MAX, group=self.distribution_group)
        largest, smallest = int(both[0].item()), -int(both[1].item())
        if largest != smallest:
            raise RuntimeError(
                f"the ranks of this group did not agree on the slot size - {smallest} to {largest} "
                f"bytes - so they would lay their ranges out differently. It is part of the "
                f"contract between ranks, not a local choice: check that every rank was given the "
                f"same {RUNAI_STREAMER_DEVICE_MEMORY_LIMIT_ENV_VAR_NAME} and "
                f"{RUNAI_STREAMER_DEVICE_MEMORY_FRACTION_ENV_VAR_NAME}"
            )

    def rank_memory_limit(self) -> int:
        """This rank's share of RUNAI_STREAMER_MEMORY_LIMIT, which is a NODE total.

        The ranks sharing a host share its RAM, so reading the variable per rank silently multiplies it
        by the number of ranks on the node - which is how a 40 GB setting becomes hundreds of GB
        resident. Divide by the ranks on THIS node, not the world size: with tensor parallelism spanning
        hosts, world size counts ranks whose RAM sits on other machines, and every buffer would be
        shrunk by the number of nodes for no benefit. It is also the denominator that matches the
        partitioning, which is done within the node group - so the ranks sharing a host are exactly the
        set that collectively reads one copy of the data.

        -1 (unlimited) and 0 (largest range) are modes rather than quantities, and pass through unchanged.

        This replaces the previous forced -1. Unlimited sized the host buffer to the whole of this rank's
        share, which is the model-sized allocation the ring buffer exists to remove; the ring makes a
        bounded limit cheap, so there is no longer a reason to opt out of one by default."""
        configured = os.environ.get(RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME)
        limit = int(configured) if configured is not None else int(DEFAULT_MEMORY_LIMIT_STRING)
        if limit <= 0:
            return limit

        ranks = max(1, self.num_processes_on_node or 1)
        # Floored at the largest range: a rank has to be able to buffer one whole tensor or streaming
        # cannot progress, and with many ranks on a node the plain share can fall below that. A rank
        # driven down to that floor usually gets exactly ONE buffer - its share pays for a single
        # largest-range buffer and no more, i.e. the sequential drain-then-refill path - so ring depth on
        # such a node needs a larger node total, not a larger floor. "Usually" because max_chunk is the
        # largest range across ALL ranks: a rank holding nothing that big, or less data than the floor,
        # can still get a deeper ring.
        per_rank = max(limit // ranks, self.max_chunk)

        if self.original_group_rank == 0:
            logger.info(
                f"[RunAI Streamer][Distributed] CPU memory limit "
                f"{humanize.naturalsize(limit, binary=True)} is a node total: {ranks} rank(s) on this "
                f"node -> {humanize.naturalsize(per_rank, binary=True)} per rank"
            )
        return per_rank

    def get_chunks(self) -> Iterator:
        if not self.file_streamer:
            raise ValueError("Streamer not initialized")
        
        BUFFER_BYTESIZE = dist_buffer_bytesize(self.max_chunk)

        # The receive buffer must hold the LARGEST slot any peer will broadcast. Where the ranks send
        # their ring slots that is the agreed slot size, not this rank's staging size - and a receive
        # buffer that is too small does not raise, because slicing it CLAMPS, so the mismatch would
        # reach the collective as a size disagreement.
        #
        # Allocated THERE, not here, so the free-memory check in set_is_distributed measures the same
        # allocations this loop makes.
        (data_buffer, received_buffer,
         batch_metadata_tensor, received_metadata_tensor) = allocate_dist_staging(
            self.max_chunk, self.device, self.metadata_rows,
            receive_bytesize=self.slot_shape[0] if self.sends_directly else None,
            # Only a rank that STAGES something needs one. A rank sending its ring slots has no use
            # for it, and a rank with an empty share sends nothing at all.
            send_buffer=self.reading_from_storage and not self.sends_directly,
        )
        
        if not self.reading_from_storage:
            source = EmptyDistributionSource()
        elif self.sends_directly:
            source = DirectDistributionSource(self.file_streamer, self.rank_dicts_map)
        else:
            # Opened HERE, not above: the direct source pulls whole slots through get_slots instead,
            # and starting this generator as well would leave two of them live on one streamer.
            source = StagedDistributionSource(
                self.file_streamer.get_chunks(),
                data_buffer[:BUFFER_BYTESIZE],
                self.rank_dicts_map,
                # THE METADATA'S OWN HEIGHT, so a slot ends on its bytes and never on its rows.
                # These were two numbers, and the smaller one silently shaped every slot.
                self.metadata_rows,
            )

        # The stream the clones and the collective run on. Held so slot_is_free can see it move,
        # and only where a source refills its slots - a staged source ignores the event.
        send_stream = None
        if self.sends_directly:
            support = device_support(self.device)
            if support is not None:
                send_stream = support.current_stream(self.device)

        chunks_to_read = [self.total_chunks_to_read]

        # The slot this rank is holding, in a cell so the finally below and the predicted loop -
        # which is a generator of its own - name the same one.
        outstanding = [None]

        try:
            if self.predicted is not None:
                # The rounds are known, so they can be pipelined. The metadata loop below cannot:
                # a receiver there learns the size of the next transfer from the transfer before it.
                for file_id, chunk_index, buffer in self.stream_predicted(
                        source, received_buffer, send_stream, outstanding):
                    yield file_id, chunk_index, self.own(buffer) if self.owned else buffer
                return

            while chunks_to_read[0] > 0:
                # --- Fill the send buffer ---
                filled = source.next_slot()
                outstanding[0] = filled
                ranges = filled.ranges if filled else []
                chunk_count_in_batch = len(ranges)
                current_data_size = len(filled.memory) if filled else 0

                write_metadata_rows(batch_metadata_tensor, ranges)

                logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Prefilled buffer with {chunk_count_in_batch} chunks and total size {humanize.naturalsize(current_data_size)}")

                # --- Broadcast ---
                #
                # Owned here rather than at broadcast()'s two yields, so there is no second site to
                # miss - a missed one is a tensor that looks right and is overwritten next round.
                #
                # The staged source needs no event and ignores the one it is given: the slot is
                # next written by its copy_ on this same stream, and received_buffer by
                # dist.broadcast on NCCL's, which torch orders against this one - a dependency
                # master already relies on for borrowed views. A source the library refills has no
                # such ordering, and that is what release() takes an event for.
                for file_id, chunk_index, buffer in self.broadcast(
                    chunks_to_read,
                    batch_metadata_tensor,
                    received_metadata_tensor,
                    filled.memory if filled else None,
                    received_buffer,
                    chunk_count_in_batch):
                    yield file_id, chunk_index, self.own(buffer) if self.owned else buffer

                # AFTER the yields. The slot is this rank's send buffer, and broadcast() hands out
                # views into it, so releasing earlier would let the next fill overwrite them.
                if filled:
                    source.release(filled, self.slot_is_free(send_stream))
                outstanding[0] = None

        except RuntimeError as e:
            # Check if the error is a timeout
            self.is_error = True
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                logger.exception(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: broadcast timed out - Could not complete broadcast.")
            else:
                logger.exception(f"[RunAI Streamer][Distributed] rank {self.original_group_rank} error: {e}")
            raise e
        except Exception as e:
            self.is_error = True
            logger.exception(f"[RunAI Streamer][Distributed] rank {self.original_group_rank} error: {e}")
            raise e
        except GeneratorExit:
            # A caller that walked away. The slot is NOT given back here: releasing resumes the
            # library's own generator, which reads a whole further slot before it returns, and a
            # generator nobody kept is closed by the collector - so that read would happen at an
            # arbitrary moment, possibly during interpreter shutdown.
            #
            # Nothing is lost by skipping it, and the bytes are safe for a reason worth stating,
            # because it is NOT that the collective runs on this stream - an async_op send runs on
            # NCCL's own. A slot nobody releases is never refilled, stream_predicted joins the send
            # it posted on the way out, and the slot is a torch allocation that NCCL recordStreams,
            # so the allocator will not hand that block to anyone else while the send reads it.
            outstanding[0] = None
            raise
        finally:
            # A FINALLY, because an exception from the broadcast leaves the slot held with no
            # other path back. A source the library refills is resumed BY the release, and the
            # event goes with it, so the pool can wait on that event instead of the whole stream.
            if outstanding[0] is not None:
                try:
                    # NO GUARD HERE, deliberately. Asking it again would re-raise whatever stopped
                    # the loop - a caller that moved streams fails the check at the release above,
                    # leaves the slot held, and would fail it a second time here, so the slot is
                    # never given back and the ring is stranded by the very code meant to save it.
                    #
                    # STILL ON send_stream. Dropping the check does not change which stream the
                    # event belongs on: the broadcast is already enqueued there and still reads the
                    # slot. A caller that moved has lost its own clones either way, and it has the
                    # ValueError; the peers have no such warning.
                    source.release(outstanding[0], self.slot_is_free(send_stream, guard=False))
                except Exception:
                    # NEVER RAISES OUT OF HERE. This runs while another exception is on its way up,
                    # or while the generator is being closed, and replacing either with a failure to
                    # tidy up would hide what actually went wrong.
                    logger.exception(
                        f"[RunAI Streamer][Distributed] rank {self.original_group_rank}: could not "
                        f"release the slot that was in flight"
                    )

    def slot_is_free(self, send_stream: Any = None, guard: bool = True) -> Optional[Any]:
        """An event marking the end of this round, or None where the device has no events.

        RECORDED ON send_stream, the stream the broadcast was enqueued on. That broadcast
        reads the slot as this rank's send buffer, so it is the work a refill must wait for. An
        event on any other stream lets the next fill overwrite bytes NCCL is still sending, which
        reaches the peers as wrong data and raises nothing.

        Recorded after both the collective and the clones are enqueued, so it follows both. It does
        NOT cover what a caller does later with a borrowed view: under owned=False the yields are
        views into the slot, and the caller may still hold them. Releasing after the yields is what
        bounds that, not this event.

        send_stream is None on the staged path, which has no broadcast to follow and ignores the
        event. guard=False asks for the event without the check below, which the cleanup path
        needs: see the finally in get_chunks.
        """
        support = device_support(self.device)
        if support is None:
            return None

        # NO SEND STREAM is the staged path, which has no broadcast of its own to follow and
        # ignores the event it is given: the slot is next written by its own copy_ on this stream.
        # There is nothing for a move to break and nothing but the current stream to record on.
        if send_stream is None:
            done = support.new_event()
            done.record(support.current_stream(self.device))
            return done

        # One event marks the work on ONE stream. A caller that moves leaves clones on a stream
        # this event never covers, and the library refills the slot while they still read it -
        # wrong bytes, nothing raised. The non-distributed ring refuses the same move.
        #
        # SAMPLED AT RELEASE, so it catches a caller that moved and stayed, not one that moved and
        # came back before releasing. The other site checks no more than this.
        if guard and support.current_stream(self.device) != send_stream:
            raise ValueError(
                "get_chunks moved to another stream part way through - every range of one stream "
                "must be read on the stream it started on, because one event per slot can only "
                "mark the work on one"
            )

        # AFTER the check, so a move that is refused leaves no event behind.
        done = support.new_event()
        done.record(send_stream)
        return done

    @staticmethod
    def own(buffer: torch.Tensor) -> torch.Tensor:
        """A private copy of buffer.

        A PLAIN CLONE, and the simplicity is the point. An earlier version over-allocated and sliced
        to an aligned address, to carry the staging alignment across the copy. The tensor it returned
        starts part way into a larger storage, and vLLM died with an illegal address in its first
        forward pass as soon as it stopped cloning our tensors for us.

        The slot layout is not lost, because it was never a promise about this tensor: it says
        where ranges sit INSIDE the staging buffers, so that one rank's offsets are usable on every
        other, and that layout is unchanged. What a caller needs is alignment for the dtype, which
        SafetensorsStreamer checks on every tensor whatever path produced it.
        """
        return buffer.clone()

    def stream_predicted(self, source, received_buffer: torch.Tensor, send_stream,
                         outstanding: List) -> Iterator:
        """Post one broadcast, then consume the one before it.

        A transfer runs while the caller is still working through the tensors of the slot before it.
        ONE turn of lookahead, not two: a second live post would need a second ring slot, and the
        source hands out one at a time.

        A rank with fewer slots than its peers is skipped on the turns it has nothing for. Every
        rank holds the same prediction, so they skip the same turns and no collective is half
        posted."""
        support = device_support(self.device)
        turns = max((len(plan) for plan in self.predicted), default=0)

        # AS MANY AS THE GROUP AGREED ROOM FOR. Two are used in turn, so a transfer lands while the
        # clones out of the one before it still run. One means no overlap, which is what a budget
        # with no room for a second buffer buys.
        receiving = [received_buffer]
        if turns and self.receive_buffers > 1:
            try:
                receiving.append(aligned_device_buffer(received_buffer.numel(), self.device))
            except torch.cuda.OutOfMemoryError:
                # THIS RANK ALONE, and nothing has to agree. The count changes no collective - every
                # rank still posts the same transfers in the same order - so a rank short of memory
                # gives up the overlap and keeps loading instead of failing the whole group.
                logger.warning(
                    f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: not enough "
                    f"device memory for a second receive buffer - receiving into one, which reads "
                    f"and sends in turn instead of together"
                )
        single = len(receiving) == 1
        # The clones out of that buffer. Posting into it again waits for them.
        reusable = [None] * len(receiving)

        posted = None
        pending_event = None
        started = False
        receives = 0

        def consume(entry):
            """Wait for a posted transfer, hand its ranges over, and return the slot's event."""
            work, buffer, rows, mine, index = entry
            # Does not block the host: it makes this stream wait on the collective, so the clones
            # the caller makes below are ordered after it.
            work.wait()
            for row in rows:
                yield row.file_id, row.range_index, buffer[row.offset: row.offset + row.size]
            if mine:
                return self.slot_is_free(send_stream)
            # The clones out of this receive buffer. Posting into it again waits here.
            event = support.new_event()
            event.record(support.current_stream(self.device))
            reusable[index] = event
            return None

        try:
            for turn in range(turns):
                for broadcasting_rank in range(self.group_size):
                    plan = self.predicted[broadcasting_rank]
                    rows = plan[turn] if turn < len(plan) else []
                    if not rows:
                        continue

                    total = rows[-1].offset + rows[-1].size
                    global_broadcasting_rank = self.local_group_global_ranks[broadcasting_rank]
                    mine = global_broadcasting_rank == self.original_group_rank

                    # A SEND NEEDS ITS SLOT FIRST, and asking for one gives the previous one back. If
                    # the posted turn is also a send, consuming it here is what keeps this to one live
                    # slot.
                    #
                    # With ONE receive buffer the posted transfer is still in it, so it has to be
                    # consumed before anything posts into it again.
                    if posted is not None and (single or (mine and posted[3])):
                        done = yield from consume(posted)
                        if posted[3]:
                            pending_event = done
                        posted = None

                    if mine:
                        if started:
                            source.release(outstanding[0], pending_event)
                            # CLEARED AS IT GOES BACK. next_slot below can raise, and the finally in
                            # get_chunks would otherwise hand the same slot back a second time.
                            outstanding[0] = None
                        filled = source.next_slot()
                        outstanding[0] = filled
                        started = True
                        self.check_fill_matches_prediction(rows, filled.ranges if filled else [],
                                                           filled.memory if filled else None)
                        buffer = filled.memory
                        # NOT an index: a send has no receive buffer. -1 would be a valid one.
                        index = None
                    else:
                        index = receives % len(receiving)
                        receives += 1
                        if reusable[index] is not None:
                            support.current_stream(self.device).wait_event(reusable[index])
                            reusable[index] = None
                        buffer = receiving[index][:total]

                    work = dist.broadcast(buffer, global_broadcasting_rank,
                                          group=self.distribution_group, async_op=True)

                    # RECORDED BEFORE THE CONSUME BELOW, which yields to the caller and can raise
                    # there. Until this assignment our own send exists only as a local, and the
                    # cleanup on the way out would have nothing to join.
                    previous, posted = posted, (work, buffer, rows, mine, index)

                    if previous is not None:
                        done = yield from consume(previous)
                        if previous[3]:
                            pending_event = done

            if posted is not None:
                done = yield from consume(posted)
                if posted[3]:
                    pending_event = done
                # JOINED, so the cleanup below has nothing left to do on the way out.
                posted = None

            if started:
                source.release(outstanding[0], pending_event)
                outstanding[0] = None

            # THE NET UNDER THE WHOLE SCHEME. A peer reads these bytes at offsets it computed instead of
            # received, so a slot the ring filled and no rank expected is never sent, and the weights in
            # it go missing with nothing raised.
            leftover = source.next_slot()
            if leftover is not None:
                # Held, so the finally in get_chunks hands it back instead of stranding the ring.
                outstanding[0] = leftover
                raise RuntimeError(
                    "the ring had a slot left that no rank predicted - the prediction and the stream "
                    "disagree about how much this rank reads"
                )
        finally:
            # OUR OWN SEND MAY STILL BE READING THE SLOT. A send is posted with async_op=True, so
            # NCCL runs it on its own stream: posting makes NCCL wait on this stream, never the
            # other way round. Only wait() puts the dependency back, and on the way out of a
            # failed round nothing else has.
            #
            # HERE, not in get_chunks. Closing this generator runs this finally FIRST, so the
            # release event recorded there lands after the collective.
            #
            # wait() enqueues a stream wait rather than blocking the host, and it does not yield,
            # so it is safe while the generator is being closed. Its own failure is swallowed: this
            # is cleanup, and replacing the exception that brought us here would hide it.
            #
            # CARRIED TO send_stream, because wait() orders the stream that is CURRENT and the
            # release event is recorded on send_stream. The two are the same stream until a caller
            # moves, and this path is reached after exactly that - so without the bridge the join
            # would land on the stream the caller moved to and the release event would cover
            # nothing. An event and a wait_event, which order streams without blocking the host.
            if posted is not None and posted[3]:
                try:
                    posted[0].wait()
                    if support is not None and send_stream is not None:
                        joined = support.new_event()
                        joined.record(support.current_stream(self.device))
                        send_stream.wait_event(joined)
                except Exception:
                    logger.debug(f"[RunAI Streamer][Distributed] Rank "
                                 f"{self.original_group_rank}: could not join the outstanding send "
                                 f"while cleaning up")

    def check_fill_matches_prediction(self, rows: List[SlotRange],
                                      ranges: List[SlotRange],
                                      sent: Optional[torch.Tensor]) -> None:
        """The ring filled exactly what this rank's peers predicted it would.

        RANGE BY RANGE, not a byte count. The peers read these bytes at the offsets they predicted
        and name them by the ids they predicted, so an id or an offset that differs sends the right
        bytes under the wrong names - and every peer agrees, so nothing downstream can notice. The
        real layout is in hand here, which makes the exact comparison the cheap one."""
        if list(rows) != list(ranges):
            raise RuntimeError(
                f"rank {self.original_group_rank}: the ring laid out {list(ranges)} and the "
                f"prediction said {list(rows)}. The packer and the prediction disagree, so the "
                f"peers would read these bytes under the wrong names"
            )

        # The peers size their receive from the prediction, so this one is a hang if it is wrong.
        expected = rows[-1].offset + rows[-1].size if rows else 0
        actual = len(sent) if sent is not None else 0
        if actual != expected:
            raise RuntimeError(
                f"rank {self.original_group_rank}: the ring filled {actual} bytes and the "
                f"prediction said {expected}"
            )

    def broadcast(self,
                  chunks_to_read: List[int],
                  batch_metadata_tensor: torch.Tensor,
                  received_metadata_tensor: torch.Tensor,
                  sent: Optional[torch.Tensor],
                  received_buffer: torch.Tensor,
                  chunk_count_in_batch: int) -> Iterator:
        """sent is this rank's outgoing slot, or None when it has nothing to send this round.

        It comes from the distribution source, so it is already the used prefix and already on the
        device the collective sends from. A direct source is read into its slot, so there is no
        buffer of ours to take a view of."""

        total_broadcast_chunks = 0

        for broadcasting_rank in range(self.group_size):
            # Map the local rank to its corresponding GLOBAL rank
            global_broadcasting_rank = self.local_group_global_ranks[broadcasting_rank]

            # self.original_group_rank is the GLOBAL rank of the current process
            if global_broadcasting_rank == self.original_group_rank:
                logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Broadcasting")
                chunks_to_read[0] -= chunk_count_in_batch
                total_broadcast_chunks += chunk_count_in_batch
                # broadcast metadata
                dist.broadcast(batch_metadata_tensor, global_broadcasting_rank, group=self.distribution_group)

                # broadcast data
                if chunk_count_in_batch > 0:
                    logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Broadcasting data of size {humanize.naturalsize(len(sent))}")
                    dist.broadcast(sent, global_broadcasting_rank, group=self.distribution_group)

                # yield
                    for orig_req_idx, orig_chunk_idx, chunk_size, chunk_offset in batch_metadata_tensor[1 : chunk_count_in_batch + 1].tolist():
                        yield orig_req_idx, orig_chunk_idx, sent[chunk_offset : chunk_offset + chunk_size]
            else:
                # receive metadata
                logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Receiving metadata from rank {global_broadcasting_rank}")
                dist.broadcast(received_metadata_tensor, global_broadcasting_rank, group=self.distribution_group)

                received_chunk_count_in_batch = received_metadata_tensor[0, 0].item()

                if received_chunk_count_in_batch == 0:
                    logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: No chunks to receive from rank {global_broadcasting_rank}")
                    continue

                total_broadcast_chunks += received_chunk_count_in_batch

                # receive data
                logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Receiving data from rank {global_broadcasting_rank}")
                last_meta = received_metadata_tensor[received_chunk_count_in_batch]
                total_data_size = last_meta[3].item() + last_meta[2].item() # offset plus size of last chunk in batch

                chunks_to_read[0] -= received_chunk_count_in_batch

                # CHECKED, because the slice would otherwise clamp: a receive buffer smaller than
                # what the sender broadcasts becomes a size mismatch inside the collective, which is
                # an abort under gloo and a hang under NCCL.
                if total_data_size > len(received_buffer):
                    raise RuntimeError(
                        f"rank {global_broadcasting_rank} is broadcasting {total_data_size} bytes "
                        f"and this rank's receive buffer holds {len(received_buffer)}; the ranks "
                        f"did not agree on a slot size"
                    )

                received_data_buf_view = received_buffer[:total_data_size]

                dist.broadcast(received_data_buf_view, global_broadcasting_rank, group=self.distribution_group)
                logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Received data of size {humanize.naturalsize(total_data_size)}")

                for orig_req_idx, orig_chunk_idx, chunk_size, chunk_offset in received_metadata_tensor[1 : received_chunk_count_in_batch + 1].tolist():
                    yield orig_req_idx, orig_chunk_idx, received_data_buf_view[chunk_offset : chunk_offset + chunk_size]

        if total_broadcast_chunks == 0 and chunks_to_read[0] > 0:
            logger.error(f"[RunAI Streamer][Distributed] Error: rank {self.rank} is missing {chunks_to_read[0]} chunks")
            raise RuntimeError("No chunks to read")
