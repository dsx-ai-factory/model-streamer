from __future__ import annotations
from typing import List, Iterator, Optional, Tuple
import torch
import torch.distributed as dist
import os
from datetime import timedelta
import socket

from runai_model_streamer.file_streamer import (
    FileStreamer,
    FileChunks,
)

from runai_model_streamer.file_streamer.requests_iterator import (
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    DEFAULT_MEMORY_LIMIT_STRING,
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
        self._ring_info: Optional[Tuple[int, int, int]] = None

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

    def get_cuda_free_memory(self) -> int:
        if not torch.cuda.is_available():
            return 0
        free_memory, total_memory = torch.cuda.mem_get_info()
        return free_memory

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
            elif enable_dist == "auto" and backend_name != "nccl":
                logger.info("[RunAI Streamer][Distributed] Note: Torch distributed backend %s is not supported by default for distributed streaming - To allow this backend, set RUNAI_STREAMER_DIST to `1`", backend_name)
                self.is_distributed = False

        # check if there is enough free memory on the device
        if self.is_distributed and torch.cuda.is_available() and dist.get_backend() == "nccl":
            free_memory = self.get_cuda_free_memory()
            if free_memory < 2 * self.params.max_chunk:
                logger.warning(f"[RunAI Streamer][Distributed] Warning: Not enough memory on the device for distributed streaming - fallback to non distributed streaming, free memory: {free_memory} bytes, required minimun: {2 * self.params.max_chunk} bytes")
                self.is_distributed = False

    def stream_files(
            self,
            file_stream_requests: List[FileChunks],
            credentials: Optional[S3Credentials],
            device: str,
            is_distributed: bool,
    ) -> None:

        # Cleared before dispatch, so a call that builds no ring - or raises before it does - reports
        # nothing rather than whatever the previous call left behind.
        self._ring_info = None

        self.params.set_params(file_stream_requests)

        # check if distributed streaming can be used
        self.set_is_distributed(is_distributed, device)

        if not self.is_distributed:
            self.file_streamer.stream_files(file_stream_requests, credentials, device)
            built_ring = True
        else:
            self.distributed_streamer.stream_files(file_stream_requests, credentials, device, self.params)
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
            )

    def ring_info(self) -> Optional[Tuple[int, int, int]]:
        """(rank, num_buffers, buffer_size) for the ring the last stream_files built, or None if it
        built none.

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
    ) -> None:

        self.broadcast_timeout = self.get_broadcast_timeout()

        # find the size of the maximal chunk for the reusable buffer
        max_chunks_per_file = (fc.max_chunk_size() for fc in file_stream_requests if fc.sizes)
        self.max_chunk = max(max_chunks_per_file, default=0)

        # set the value of RUNAI_STREAMER_PROCESS_GROUP_SIZE to be the number of processes in the distribution group (or 1 if process group is not initialized)
        # This is useful for auto adjustments in the CPP layer, which depends on the number of processes of this workload
        if self.num_processes_on_node is None:
            self.num_processes_on_node, self.my_global_rank, self.groups_by_ranks = self.find_local_ranks()
        if os.environ.get("RUNAI_STREAMER_PROCESS_GROUP_SIZE") is None:
            os.environ["RUNAI_STREAMER_PROCESS_GROUP_SIZE"] = str(self.num_processes_on_node)

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
                torch.distributed.destroy_process_group(group=self.distribution_group)
                del self.distribution_group
                self.distribution_group = None
        return

    def create_distribution_group(self) -> Optional[dist.ProcessGroup]:
        if self.distribution_group:
            return self.distribution_group

        is_global_group = int(os.environ.get("RUNAI_STREAMER_DIST_GLOBAL", "0")) == 1

        if is_global_group:
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

    def set_rank(self) -> None:
        self.rank = dist.get_rank(group=self.distribution_group)
        self.original_group_rank = dist.get_rank()

    def stream_files(
            self,
            file_stream_requests: List[FileChunks],
            credentials: Optional[S3Credentials],
            device: str,
            params : _distributedStreamerParams
    ) -> None:

        self.device = torch.device(device)
        self.my_global_rank = params.my_global_rank
        self.groups_by_ranks = params.groups_by_ranks
        self.broadcast_timeout = params.broadcast_timeout
        self.max_chunk = params.max_chunk
        # the ranks sharing this host: the divisor for the node-total memory limit (see rank_memory_limit)
        self.num_processes_on_node = params.num_processes_on_node

        # create distribution group for configuring timeout and possibly broadcast locally in each node 
        # The group must be created before partitioning the tensors, in case the group will be local
        self.distribution_group = self.create_distribution_group()
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

        if len(self.rank_file_chunks_list) == 0:
            self.reading_from_storage = False
            logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: No chunks to read from storage")
            return

        # read files
        self.file_streamer.stream_files(
            self.rank_file_chunks_list, credentials, "cpu", memory_limit=self.rank_memory_limit()
        )
        self.reading_from_storage = True

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
        
        MAX_CHUNKS_PER_BATCH = 256
        DEFAULT_BUFFER_MIN_BYTESIZE = 1024 * 1024 * 1024
        BUFFER_MIN_BYTESIZE = int(os.environ.get("RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE", str(DEFAULT_BUFFER_MIN_BYTESIZE))) # environment variable used for testing
        BUFFER_BYTESIZE = max(BUFFER_MIN_BYTESIZE, self.max_chunk)
        alignment = get_dist_buffer_alignment()

        # Over-allocate by alignment so we can slice to the first aligned address within each buffer.
        # This guarantees data_buffer.data_ptr() % alignment == 0 and
        # received_buffer.data_ptr() % alignment == 0, so any offset that is a multiple of
        # alignment is also aligned in both buffers — required for alignment to hold on receiving ranks.
        data_buffer_raw = torch.empty(BUFFER_BYTESIZE + alignment, dtype=torch.uint8, device=self.device)
        received_buffer_raw = torch.empty(BUFFER_BYTESIZE + alignment, dtype=torch.uint8, device=self.device)
        data_buffer = data_buffer_raw if alignment <= 1 else data_buffer_raw[(-data_buffer_raw.data_ptr()) % alignment:]
        received_buffer = received_buffer_raw if alignment <= 1 else received_buffer_raw[(-received_buffer_raw.data_ptr()) % alignment:]

        batch_metadata_tensor = torch.zeros(MAX_CHUNKS_PER_BATCH + 1, 4, dtype=torch.int64, device=self.device)
        received_metadata_tensor = torch.zeros(MAX_CHUNKS_PER_BATCH + 1, 4, dtype=torch.int64, device=self.device)
        
        ready_chunks_iterator = self.file_streamer.get_chunks()
        
        def chunk_generator():
            yield from ready_chunks_iterator
            while True:
                yield None

        chunks_to_read = [self.total_chunks_to_read]
        chunk_gen = chunk_generator()
        leftover_chunk = [None]

        try:
            while chunks_to_read[0] > 0:
                # --- Fill staging buffer ---
                current_data_size, chunk_count_in_batch = self.prefill(
                    chunk_gen,
                    MAX_CHUNKS_PER_BATCH,
                    BUFFER_BYTESIZE,
                    data_buffer,
                    received_buffer,
                    batch_metadata_tensor,
                    received_metadata_tensor,
                    leftover_chunk,
                    alignment)

                logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Prefilled buffer with {chunk_count_in_batch} chunks and total size {humanize.naturalsize(current_data_size)}")

                # --- Broadcast ---
                yield from self.broadcast(
                    chunks_to_read,
                    batch_metadata_tensor,
                    received_metadata_tensor,
                    data_buffer,
                    received_buffer,
                    chunk_count_in_batch,
                    current_data_size)
               
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

    def prefill(self,
                chunk_gen: Iterator,
                max_chunks_per_batch: int,
                buffer_bytesize: int,
                data_buffer: torch.Tensor,
                received_buffer: torch.Tensor,
                batch_metadata_tensor: torch.Tensor,
                received_metadata_tensor: torch.Tensor,
                leftover_chunk: List[Tuple[int, int, torch.Tensor]],
                alignment: int) -> Tuple[int, int]:

        current_data_size = 0
        chunk_count_in_batch = 0
        base_ptr = data_buffer.data_ptr()

        while chunk_count_in_batch < max_chunks_per_batch and self.reading_from_storage:

            # Prioritize the leftover chunk from the previous batch.
            if leftover_chunk[0]:
                chunk_item = leftover_chunk[0]
                leftover_chunk[0] = None
            else:
                chunk_item = next(chunk_gen)

            if chunk_item is None:
                break

            ready_request_id, ready_chunk_index, cpu_buffer = chunk_item
            chunk_size = cpu_buffer.numel()

            chunk_offset = aligned_offset(base_ptr, current_data_size, alignment)
            if chunk_offset + chunk_size > buffer_bytesize:
                # This chunk doesn't fit, so we save it for the next batch.
                leftover_chunk[0] = chunk_item
                break

            data_buffer[chunk_offset : chunk_offset + chunk_size].copy_(cpu_buffer.squeeze())

            orig_req_idx, orig_chunk_idx, _ = self.rank_dicts_map[ready_request_id][ready_chunk_index]
            batch_metadata_tensor[chunk_count_in_batch + 1].copy_(
                torch.tensor([orig_req_idx, orig_chunk_idx, chunk_size, chunk_offset])
            )

            current_data_size = chunk_offset + chunk_size
            chunk_count_in_batch += 1

        batch_metadata_tensor[0, 0] = chunk_count_in_batch

        return current_data_size, chunk_count_in_batch

    def broadcast(self,
                  chunks_to_read: List[int],
                  batch_metadata_tensor: torch.Tensor,
                  received_metadata_tensor: torch.Tensor,
                  data_buffer: torch.Tensor,
                  received_buffer: torch.Tensor,
                  chunk_count_in_batch: int,
                  current_data_size: int) -> Iterator:        

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
                    logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Broadcasting data of size {humanize.naturalsize(current_data_size)}")
                    dist.broadcast(data_buffer[:current_data_size], global_broadcasting_rank, group=self.distribution_group)

                # yield
                    for j in range(chunk_count_in_batch):
                        meta = batch_metadata_tensor[j + 1]
                        offset, size = meta[3].item(), meta[2].item()
                        yield meta[0].item(), meta[1].item(), data_buffer[offset : offset + size]
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

                received_data_buf_view = received_buffer[:total_data_size]

                dist.broadcast(received_data_buf_view, global_broadcasting_rank, group=self.distribution_group)
                logger.debug(f"[RunAI Streamer][Distributed] Rank {self.original_group_rank}: Received data of size {humanize.naturalsize(total_data_size)}")

                for j in range(received_chunk_count_in_batch):
                    meta = received_metadata_tensor[j + 1]
                    offset, size = meta[3].item(), meta[2].item()
                    yield meta[0].item(), meta[1].item(), received_data_buf_view[offset : offset + size]

        if total_broadcast_chunks == 0 and chunks_to_read[0] > 0:
            logger.error(f"[RunAI Streamer][Distributed] Error: rank {self.rank} is missing {chunks_to_read[0]} chunks")
            raise RuntimeError("No chunks to read")
