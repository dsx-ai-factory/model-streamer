# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import List, Iterator, Optional, Tuple
import torch
import torch.distributed as dist
import os
import time
from datetime import timedelta
import socket

from runai_model_streamer.file_streamer import (
    FileStreamer,
    FileChunks,
)

from runai_model_streamer.file_streamer.file_streamer import host_or_device

from runai_model_streamer.torch_device.torch_device import (
    can_measure_device_memory,
    device_free_memory,
)

from runai_model_streamer.file_streamer.requests_iterator import (
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    DEFAULT_MEMORY_LIMIT_STRING,
    device_ring_sizing,
    host_ring_sizing,
)

from runai_model_streamer.distributed_streamer.partition import (
    partition,
    partition_for_rank,
    get_total_number_of_chunks,
    get_partition_policy,
    log_partition_info,
    log_rank_span_info,
)
from runai_model_streamer.distributed_streamer.prediction import predicted_slots

from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
)

from timeit import default_timer as timer

import logging
import humanize

logger = logging.getLogger(__name__)

RUNAI_STREAMER_CUDA_ALIGNMENT_ENV_VAR = "RUNAI_STREAMER_CUDA_ALIGNMENT"
DEFAULT_DIST_BUFFER_ALIGNMENT = 512
MAX_DIST_BUFFER_ALIGNMENT = 1024 * 1024


def get_dist_buffer_alignment() -> int:
    """
    Get the alignment for the distributed buffer.
    """
    value = int(os.environ.get(RUNAI_STREAMER_CUDA_ALIGNMENT_ENV_VAR, str(DEFAULT_DIST_BUFFER_ALIGNMENT)))
    if value < 0 or value > MAX_DIST_BUFFER_ALIGNMENT:
        raise ValueError(f"Invalid value for RUNAI_STREAMER_CUDA_ALIGNMENT: {value}, must be between 0 and {MAX_DIST_BUFFER_ALIGNMENT} bytes")
    return value


MAX_CHUNKS_PER_BATCH = 256
DEFAULT_BUFFER_MIN_BYTESIZE = 1024 * 1024 * 1024


def dist_buffer_bytesize(max_chunk: int) -> int:
    """One staging buffer: the minimum, or the largest chunk if that is bigger."""
    minimum = int(os.environ.get("RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE", str(DEFAULT_BUFFER_MIN_BYTESIZE)))  # environment variable used for testing
    return max(minimum, max_chunk)


def is_global_distribution() -> bool:
    """Whether the ranks distribute across the whole job, rather than within each node."""
    return int(os.environ.get("RUNAI_STREAMER_DIST_GLOBAL", "0")) == 1


def allocate_dist_staging(max_chunk: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every device allocation the broadcast loop makes: two staging buffers and two metadata tensors.

    Each staging buffer is over-allocated by the alignment and sliced to the first aligned address, so
    an offset that is a multiple of the alignment lands aligned in BOTH buffers - which is what a
    receiving rank relies on to read a sender's layout.

    dist_device_bytes_required() measures THIS function, and a test allocates from it and compares, so
    the free-memory check cannot disagree with the allocation about how many buffers there are."""
    buffer_bytesize = dist_buffer_bytesize(max_chunk)
    alignment = get_dist_buffer_alignment()

    def staging() -> torch.Tensor:
        raw = torch.empty(buffer_bytesize + alignment, dtype=torch.uint8, device=device)
        return raw if alignment <= 1 else raw[(-raw.data_ptr()) % alignment:]

    def metadata() -> torch.Tensor:
        return torch.zeros(MAX_CHUNKS_PER_BATCH + 1, 4, dtype=torch.int64, device=device)

    return staging(), staging(), metadata(), metadata()


def dist_device_bytes_required(max_chunk: int) -> int:
    """What allocate_dist_staging() takes on the device."""
    staging = dist_buffer_bytesize(max_chunk) + get_dist_buffer_alignment()
    metadata = (MAX_CHUNKS_PER_BATCH + 1) * 4 * 8      # int64
    return 2 * staging + 2 * metadata


def aligned_offset(base_ptr: int, current_offset: int, alignment: int) -> int:
    """Return the smallest offset >= current_offset such that (base_ptr + offset) % alignment == 0."""
    if alignment <= 1:
        return current_offset
    total_addr = base_ptr + current_offset
    padding = (-total_addr) % alignment
    return current_offset + padding


DEFAULT_BROADCAST_TIMEOUT = timedelta(seconds=600)

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

        # The first collective on a group creates its NCCL communicator, which takes device memory.
        # Done first, with the operation the stream uses, so the free memory read below is what is
        # left for the staging buffers.
        warmup = torch.zeros(1, dtype=torch.uint8, device=device)
        dist.broadcast(warmup, src=dist.get_global_rank(group, 0), group=group)

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
        if self.is_distributed:
            backend_name = dist.get_backend()
            if backend_name == "nccl" and device == "cpu":
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
        if self.is_distributed and can_measure_device_memory(device) and dist.get_backend() in ("nccl", "xccl"):
            free_memory = self.get_min_device_free_memory(device)
            required = dist_device_bytes_required(self.params.max_chunk)
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
            self.distributed_streamer.stream_files(file_stream_requests, credentials, device, self.params, owned=owned)
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
        self.my_global_rank = 0
        self.groups_by_ranks = []
        self.broadcast_timeout = DEFAULT_BROADCAST_TIMEOUT
        self.max_chunk = 0
 
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

        This function performs all_gather_object on the global group which cause nccl to allocate  device memory which otherwise may never be allocated by the user application
        To avoid this issue, we create a another global subgroup just for discovering the local ranks and then destroy that subgroup after the ranks are discovered.
        """
        if not dist.is_initialized():
            return 1, 0, [[0]]

        world_size = dist.get_world_size()

        if world_size == 1:
            return 1, 0, [[0]]

        my_global_rank = dist.get_rank()

        # 1. Discover all peers on all nodes (this is a global collective)
        # create global group just for discovering the local ranks
        sandbox_global_group = dist.new_group(ranks=list(range(world_size)), timeout=self.broadcast_timeout)
        all_hostnames = [None] * world_size
        dist.all_gather_object(all_hostnames, socket.gethostname(), group=sandbox_global_group)
        dist.destroy_process_group(group=sandbox_global_group)
        del sandbox_global_group
 
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

        return my_group_size, my_global_rank, groups_by_ranks

    def set_params(
            self,
            file_stream_requests: List[FileChunks], 
            discover_ranks: bool,
    ) -> None:

        self.broadcast_timeout = self.get_broadcast_timeout()

        # find the size of the maximal chunk for the reusable buffer
        max_chunks_per_file = (fc.max_chunk_size() for fc in file_stream_requests if fc.sizes)
        self.max_chunk = max(max_chunks_per_file, default=0)

        if discover_ranks:
            self.discover_local_ranks()
        else:
            self.set_rank_without_discovery()

    def discover_local_ranks(self) -> None:
        # set the value of RUNAI_STREAMER_PROCESS_GROUP_SIZE to be the number of processes in the distribution group (or 1 if process group is not initialized)
        # This is useful for auto adjustments in the CPP layer, which depends on the number of processes of this workload
        if self.num_processes_on_node is None:
            self.num_processes_on_node, self.my_global_rank, self.groups_by_ranks = self.find_local_ranks()
        if os.environ.get("RUNAI_STREAMER_PROCESS_GROUP_SIZE") is None:
            os.environ["RUNAI_STREAMER_PROCESS_GROUP_SIZE"] = str(self.num_processes_on_node)

    def set_rank_without_discovery(self) -> None:
        # RUNAI_STREAMER_PROCESS_GROUP_SIZE is not guessed, so the CPP layer defaults to 1: a count larger than the readers lowers their throughput
        if dist.is_initialized():
            self.my_global_rank = dist.get_rank()

class _distributedStreamer:
    def __init__(self, file_streamer : FileStreamer) -> None:
        self.file_streamer = file_streamer
        self.device = None
        self.rank_file_chunks_list = {} # post Partitioning FileChunks to be streamed by this rank
        self.rank_dicts_map = {} # maps post partitioning FileChunks.id and chunk index to the original FileChunks.id and chunk index and chunk size
        self.total_chunks_to_read = 0
        self.group_size = 1
        self.rank = 0
        self.original_group_rank = 0
        self.distribution_group = None
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
            try:
                # Only attempt a barrier if there was no exception.
                if exc_type is None:
                    torch.distributed.barrier(group=self.distribution_group)
            finally:
                # ALWAYS destroy the group.
                self.release_group()
        return

    def create_distribution_group(self) -> Optional[dist.ProcessGroup]:
        if self.distribution_group:
            return self.distribution_group

        if is_global_distribution():
            # create global group
            world_size = dist.get_world_size()

            all_ranks = list(range(world_size))
            self.local_group_global_ranks = all_ranks
            group = dist.new_group(ranks = all_ranks, timeout = self.broadcast_timeout)

        else:
            group = self.create_local_distribution_group()

        logger.debug(f"[RunAI Streamer][Distributed] Created distribution group with size {dist.get_world_size(group=group)}")
        return group

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
         
        created_groups = [dist.new_group(ranks=ranks, timeout=self.broadcast_timeout) for ranks in self.groups_by_ranks]
        
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

    def release_group(self) -> None:
        """Destroy the distribution group, if there is one. Local to this rank: no barrier."""
        if self.distribution_group:
            torch.distributed.destroy_process_group(group=self.distribution_group)
            self.distribution_group = None

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
    ) -> None:

        self.device = torch.device(device)
        # ALWAYS, whatever the caller asked. The slot a tensor comes out of is a ring slot that will
        # be refilled, so a view handed over unowned would change under the caller. Costs nothing
        # here: the clone is device to device on the caller's own stream.
        self.owned = True
        self.max_chunk = params.max_chunk
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

        # THE WHOLE MODEL decides the ring shape, not this rank's share.
        #
        # Sizing from the share gives each rank a different slot size - it is floored by the largest
        # range that rank holds - and ranks that pack differently lay their ranges out differently,
        # which the prediction cannot survive. It is forced anyway: a slot is broadcast INTO a
        # receive buffer, so that buffer is the same size on every rank.
        #
        # Decided BEFORE the ring is built, from one reading of free memory taken across the group.
        shape = self.agreed_slot_shape(file_stream_requests, device)
        self.slot_size = shape[0] if shape is not None else 0
        self.check_slot_size_agrees()

        # WHAT EVERY RANK WILL SEND, computed rather than received. This REPLACES the metadata
        # broadcast: a receiver knows the ranges, their sizes and their offsets before anything
        # arrives, so it can post the data broadcast without waiting for a message first.
        #
        # Every term is identical on every rank - the whole file list, the group size, and the slot
        # size agreed above - so this needs no collective of its own.
        #
        # BEFORE the empty-share return below. A rank with nothing to read still receives from its
        # peers, so it needs the prediction just as much as one that reads.
        self.predicted = [
            list(predicted_slots(file_stream_requests, self.group_size, rank, self.slot_size))
            for rank in range(self.group_size)
        ]
        self.new_event = None
        self.current_stream = None

        if len(self.rank_file_chunks_list) == 0:
            self.reading_from_storage = False
            logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: No chunks to read from storage")
            return

        # read files
        #
        # THE REAL DEVICE, not "cpu": the ring is where the broadcast reads from, so it has to be
        # device memory. A device the C++ layer cannot write into falls back to the host ring on its
        # own, inside stream_files.
        #
        # owned=False whatever the caller asked for: this ring is internal, and a device ring refuses
        # to hand a slot over anyway. Ownership on this path is the per tensor clone in get_chunks.
        self.file_streamer.stream_files(
            self.rank_file_chunks_list, credentials, device,
            memory_limit=self.rank_memory_limit(), owned=False, slot_shape=shape,
            tight_packing=True,
        )
        self.reading_from_storage = True

        # Event A's factory, taken from the ring rather than from a vendor name. None for a host
        # ring: nothing there is enqueued, so a slot is free when the broadcast call returns.
        ring = self.file_streamer.requests_iterator
        slot_memory = ring.slots if ring is not None else None
        self.new_event = slot_memory.new_event if slot_memory is not None else None
        self.current_stream = slot_memory.current_stream if slot_memory is not None else None

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

    def agreed_slot_shape(self, file_stream_requests: List[FileChunks],
                          device: str) -> Optional[Tuple[int, int]]:
        """(buffer_size, num_buffers) every rank of the group will use, or None for the host ring.

        Every term is the same on every rank: the whole file list, the group size, and the SMALLEST
        free memory in the group. So this is agreed by construction, with one collective and no
        negotiation."""
        free = device_free_memory(device)
        if free is not None:
            smallest = torch.tensor([free], dtype=torch.int64, device=self.device)
            dist.all_reduce(smallest, op=dist.ReduceOp.MIN, group=self.distribution_group)
            free = int(smallest.item())

        # The FULL list, so the floor is the largest range in the model rather than in a share.
        shape = device_ring_sizing(free, file_stream_requests)
        if shape is not None:
            return shape

        # No device ring here - too little room, or a device the C++ layer cannot write into. The
        # host ring still has to agree, for the same reason, so it is sized from the whole model too.
        return host_ring_sizing(file_stream_requests, self.rank_memory_limit())

    def check_slot_size_agrees(self) -> None:
        """Every rank packs to the same slot size, or they predict different layouts.

        Each rank sizes its own ring from its own free memory, so this is not a given. One collective
        per stream, carrying the value and its negation so a single MAX answers both ends."""
        both = torch.tensor([self.slot_size, -self.slot_size], dtype=torch.int64, device=self.device)
        dist.all_reduce(both, op=dist.ReduceOp.MAX, group=self.distribution_group)
        largest, smallest = int(both[0].item()), -int(both[1].item())
        if largest != smallest:
            raise RuntimeError(
                f"the ranks of this group packed to different slot sizes - {smallest} to {largest} "
                f"bytes - so they would lay their ranges out differently. The slot size is part of "
                f"the contract between ranks, not a local choice"
            )

    def slot_release_event(self):
        """EVENT A: the slot may be refilled once this completes.

        Two readers to cover, and one event covers both because both are ordered on the stream it is
        recorded on. NCCL reads the slot as the broadcast source - a broadcast does not block the
        host, it is enqueued and the current stream is made to wait on it. The consumer's clones are
        enqueued on that same stream at the yields above.

        What is NOT on any torch stream is the C++ layer's next write into this slot, which is the
        whole reason an event is needed: nothing else orders a read of the ring against the refill of
        it. A host ring needs none - nothing there is enqueued."""
        if self.new_event is None:
            return None
        event = self.new_event()
        event.record(self.current_stream())
        return event

    def get_chunks(self) -> Iterator:
        """Every rank's slots, broadcast in the order every rank predicted.

        NO METADATA CROSSES THE WIRE. What a slot holds, how large it is and where each range sits
        inside it all come from predicted_slots(), computed from the file list, the group size and
        the slot size - which is why those three have to be identical on every rank.

        THE SLOT IS THE SEND BUFFER. Nothing is copied into a staging buffer on the way out. That
        copy was a host to device pass over the whole model, between the read and the wire.

        ONE TURN OF LOOKAHEAD. A broadcast is posted and the PREVIOUS one is consumed, so a transfer
        runs while the caller is still working through the tensors before it. The collectives
        themselves are still posted one at a time - overlapping those as well would mean holding two
        ring slots at once, which get_slots does not allow.

        A rank with fewer slots than its peers is skipped on the turns it has nothing for. Every rank
        computes the same prediction, so they skip the same turns and no collective is half posted."""
        if not self.file_streamer:
            raise ValueError("Streamer not initialized")

        # TWO receive buffers, alternating: one is being filled while the other is being read. One
        # would serialise the two and there would be nothing to overlap.
        #
        # NOT IN THE RING'S BUDGET YET. agreed_slot_shape sizes the ring alone, so the real peak is
        # (num_buffers + 2) slots. Keep two slots back in the sizing instead of adding them here.
        receiving = [torch.empty(self.slot_size, dtype=torch.uint8, device=self.device)
                     for _ in range(2)]
        # EVENT B: the clones out of that buffer. Posting into it again must wait for them.
        reusable = [None, None]

        slots = self.file_streamer.get_slots() if self.reading_from_storage else None
        pending_event = None
        started = False
        posted = None

        # IS NCCL STARVED? Timed on the DEVICE, not on the host: work.wait() does not block the
        # host, it makes this stream wait on the collective, so a host clock reads nothing either
        # way. A pair of events around that wait measures how long the stream really sat there.
        #
        # Near zero means the transfer finished before it was needed - NCCL is ahead and Python is
        # the bottleneck, so posting collectives further ahead would buy nothing. Large means the
        # opposite.
        profiling = os.environ.get("RUNAI_STREAMER_DIST_PROFILE", "0") != "0" and self.new_event
        waits = []
        started_at = time.perf_counter()

        def consume(entry):
            """Wait for a posted broadcast, hand its ranges over, and RETURN the slot's event.

            `yield from` carries that return value out, so the event travels with the work it
            belongs to instead of through a field that only looks safe because of the order the
            caller happens to use it in."""
            work, buffer, ranges, mine, index = entry
            # Does not block the host: it makes the current stream wait on the collective, so the
            # clones below are ordered after it.
            if profiling and len(waits) < 4096:
                before = self.new_event(enable_timing=True)
                before.record(self.current_stream())
                work.wait()
                after = self.new_event(enable_timing=True)
                after.record(self.current_stream())
                waits.append((before, after))
            else:
                work.wait()
            for chunk in ranges:
                yield chunk.file_id, chunk.range_index, self.own(
                    buffer[chunk.offset : chunk.offset + chunk.size]
                )
            if mine:
                # EVENT A, after the broadcast and after the clones - both on this stream.
                return self.slot_release_event()
            if self.new_event is not None:
                # EVENT B: the clones out of this receive buffer. Posting into it again waits here.
                event = self.new_event()
                event.record(self.current_stream())
                reusable[index] = event
            return None

        turn_of = 0
        turns = max((len(slots_of_rank) for slots_of_rank in self.predicted), default=0)
        for turn in range(turns):
            for broadcasting_rank in range(self.group_size):
                ranges = self.predicted[broadcasting_rank]
                if turn >= len(ranges):
                    continue
                ranges = ranges[turn]
                total = ranges[-1].offset + ranges[-1].size if ranges else 0
                source = self.local_group_global_ranks[broadcasting_rank]
                mine = source == self.original_group_rank

                # A SEND NEEDS ITS SLOT FIRST, and asking for one releases the previous. If the
                # posted turn is also a send, consuming it now is what keeps this to one live slot.
                if mine and posted is not None and posted[3]:
                    pending_event = yield from consume(posted)
                    posted = None

                if mine:
                    request, memory = (slots.send(pending_event) if started else next(slots))
                    started = True
                    buffer = memory if torch.is_tensor(memory) else torch.from_numpy(memory)
                    index = -1
                    if buffer.numel() != total:
                        raise RuntimeError(
                            f"rank {self.original_group_rank} slot {turn}: the ring filled "
                            f"{buffer.numel()} bytes but every rank predicted {total}. The "
                            f"prediction and the ring disagree, so the broadcast sizes would not "
                            f"match"
                        )
                else:
                    index = turn_of % len(receiving)
                    turn_of += 1
                    if reusable[index] is not None:
                        # Posting into a buffer the previous round is still being cloned out of.
                        # The collective is ordered behind this stream, so waiting here is enough.
                        self.current_stream().wait_event(reusable[index])
                        reusable[index] = None
                    buffer = receiving[index][:total]

                work = dist.broadcast(buffer, source, group=self.distribution_group, async_op=True)

                if posted is not None:
                    done = yield from consume(posted)
                    if posted[3]:
                        pending_event = done
                posted = (work, buffer, ranges, mine, index)

        if posted is not None:
            done = yield from consume(posted)
            if posted[3]:
                pending_event = done

        if profiling and waits:
            torch.cuda.synchronize()
            waited = sum(before.elapsed_time(after) for before, after in waits)
            wall = (time.perf_counter() - started_at) * 1000.0
            logger.info(
                "[RunAI Streamer][Distributed] Rank %s: the stream waited %.0f ms on %d collectives "
                "out of %.0f ms of streaming (%.1f%%). A small share means the transfers were ready "
                "before they were needed, so posting them further ahead would not help",
                self.original_group_rank, waited, len(waits), wall, 100.0 * waited / max(wall, 1e-9))

        if slots is not None:
            # DELIVERED, not closed. close() raises GeneratorExit at the yield, so the release after
            # it never runs and the last slot goes back without its event - the one slot whose
            # readers include a broadcast that may still be in flight.
            try:
                slots.send(pending_event)
                raise RuntimeError(
                    "the ring had a slot left that no rank predicted - the prediction and the "
                    "stream disagree about how much this rank reads"
                )
            except StopIteration:
                pass

    @staticmethod
    def own(buffer: torch.Tensor) -> torch.Tensor:
        """A private copy of buffer.

        A PLAIN CLONE, and the simplicity is the point. An earlier version over-allocated and sliced
        to an aligned address, to carry RUNAI_STREAMER_CUDA_ALIGNMENT across the copy. The tensor it
        returned starts part way into a larger storage, and vLLM died with an illegal address in its
        first forward pass as soon as it stopped cloning our tensors for us.

        Always, on this path: the buffer is a RING SLOT that will be refilled, or the one receive
        buffer that the next peer overwrites. Neither survives the turn it was handed out on."""
        return buffer.clone()
