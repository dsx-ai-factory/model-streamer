from __future__ import annotations
from typing import Collection, Iterator, Optional
import torch
import glob
import os
import fcntl
import shutil
import logging
import humanize
from timeit import default_timer as timer
from typing import List

from runai_model_streamer.file_streamer import FileChunks

from runai_model_streamer.distributed_streamer import DistributedStreamer

import runai_model_streamer.safetensors_streamer.safetensors_pytorch as safetensors_pytorch

from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
    is_s3_path,
    is_gs_path,
    is_azure_path,
    s3_glob,
    s3_pull_files,
    gcs_glob,
    gcs_pull_files,
    azure_glob,
    azure_pull_files,
)

logger = logging.getLogger(__name__)

SAFETENSORS_PATTERN = "*.safetensors"

def list_safetensors(path: str, s3_credentials : Optional[S3Credentials] = None) -> List[str]:
    """
    List all safetensors files in the given path.
    This function is not recursive.
    """
    if is_s3_path(path):
        files = s3_glob(path, [SAFETENSORS_PATTERN], s3_credentials)
    elif is_gs_path(path):
        files = gcs_glob(path, [SAFETENSORS_PATTERN])
    elif is_azure_path(path):
        files = azure_glob(path, [SAFETENSORS_PATTERN])
    else:
        files = glob.glob(os.path.join(path, SAFETENSORS_PATTERN))
    
    return files

def pull_files(model_path: str,
                dst: str,
                allow_pattern: Optional[List[str]] = None,
                ignore_pattern: Optional[List[str]] = None,
                s3_credentials : Optional[S3Credentials] = None) -> None:
    """
    Pull all safetensors files in the given path.
    This function is recursive.
    """
    if is_s3_path(model_path):
        return s3_pull_files(model_path, dst, allow_pattern, ignore_pattern, s3_credentials)
    if is_gs_path(model_path):
        return gcs_pull_files(model_path, dst, allow_pattern, ignore_pattern)
    if is_azure_path(model_path):
        return azure_pull_files(model_path, dst, allow_pattern, ignore_pattern)
    raise NotImplementedError("pull files is not implemented for file system paths")

class ObjectStorageModel:
    """
    Process-safe, idempotent wrapper for downloading model files from object storage.

    Multiple processes calling pull_files() concurrently with the same dst will
    serialize via a file lock. The first process downloads; the rest wait and skip
    (sentinel-based idempotency).

    Use as a context manager. The sentinel is written and the lock is released on
    clean exit. If an exception is raised inside the block the lock is still released
    but the sentinel is NOT written, so the next process will retry the download.

    Locking mechanism:
        Uses fcntl.flock, which is supported on Linux only (not Windows).
        The lock file is placed at dst + ".lock".

        Supported: multiple processes on the same machine, whether dst is a local
        disk or a network-mounted filesystem (NFS, EFS, etc.) — the kernel manages
        the lock locally.

        Not supported: processes on different machines sharing the same dst over
        a network filesystem. fcntl.flock does not provide reliable cross-host
        locking on NFS (NFSv3 and earlier silently ignore it; NFSv4 depends on
        mount options and server support).

    Example::

        with ObjectStorageModel(model_path=url, dst=cache_dir) as obj:
            obj.pull_files(allow_pattern=["*.safetensors"])
            obj.pull_files(ignore_pattern=["*.safetensors"])
    """

    SENTINEL_NAME = ".runai_complete"

    def __init__(
        self,
        model_path: str,
        dst: str,
        s3_credentials: Optional[S3Credentials] = None,
    ) -> None:
        if not (is_s3_path(model_path) or is_gs_path(model_path) or is_azure_path(model_path)):
            raise ValueError(
                f"model_path {model_path!r} is not a supported object storage path "
                "(expected s3://, gs://, or az://)"
            )
        self.dir = dst.rstrip("/") or "/"
        self._model_path = model_path if model_path.endswith("/") else model_path + "/"
        self._s3_credentials = s3_credentials
        self._lock_path = self.dir + ".lock"
        self._sentinel = os.path.join(self.dir, self.SENTINEL_NAME)
        self._lock_file = None
        self._lock_file = open(self._lock_path, "a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_SH)  # shared: fast path for already-downloaded
            if os.path.exists(self._sentinel):
                self._skip = True
            else:
                # Release SH before requesting EX. Direct SH→EX promotion can
                # deadlock if two processes both try to upgrade simultaneously:
                # each blocks waiting for the other to release its SH, and
                # neither ever runs again. flock(2) does not detect this cycle.
                # Re-check sentinel after acquiring EX since another process may
                # have completed the download in the gap.
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                fcntl.flock(self._lock_file, fcntl.LOCK_EX)
                if os.path.exists(self._sentinel):
                    self._skip = True
                    # Downgrade to SH: no further writes needed, allowing other
                    # waiting processes to proceed in parallel rather than serialize.
                    fcntl.flock(self._lock_file, fcntl.LOCK_SH)
                else:
                    self._skip = False
                    if os.path.exists(self.dir):
                        shutil.rmtree(self.dir)
                    os.makedirs(self.dir, exist_ok=True)
        except BaseException:
            # BaseException (not Exception) is intentional: KeyboardInterrupt and
            # SystemExit must also release the lock, otherwise sibling processes
            # waiting on flock will block indefinitely after a Ctrl+C or SIGTERM.
            if self._lock_file is not None:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()
            raise

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        if self._skip:
            return
        pull_files(self._model_path, self.dir, allow_pattern, ignore_pattern, self._s3_credentials)

    def __del__(self) -> None:
        # Use getattr in case __init__ raised before self._lock_file was assigned.
        lock_file = getattr(self, '_lock_file', None)
        if lock_file is not None and not lock_file.closed:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def __enter__(self) -> ObjectStorageModel:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is None and not self._skip:
                downloaded = [f for f in os.listdir(self.dir) if f != self.SENTINEL_NAME]
                if not downloaded:
                    raise RuntimeError(
                        f"No files were downloaded to {self.dir!r} — "
                        "verify that the model path is correct"
                    )
                try:
                    with open(self._sentinel, "w"):
                        pass
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to write download sentinel {self._sentinel!r}: {exc}"
                    ) from exc
        finally:
            lock_file = getattr(self, '_lock_file', None)
            if lock_file is not None:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                finally:
                    lock_file.close()
        return False


class SafetensorsStreamer:
    def __init__(self) -> None:
        self.file_streamer = DistributedStreamer()
        self.files_to_tensors_metadata = {}
        self.total_size = 0
        self.device_str = None
        self.start_time = None

    def __enter__(self) -> SafetensorsStreamer:
        self.file_streamer.__enter__()
        return self

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        return self.file_streamer.__exit__(exc_type, exc_value, traceback)

    def _log_throughput(self) -> None:
        # The session-level throughput: one SafetensorsStreamer session may drive several FileStreamer
        # requests (memory-limited chunking today, concurrent submissions with the ring buffer), so the
        # overall figure belongs here, not in FileStreamer. Emitted from get_tensors() completion (active
        # consumption) rather than __exit__, which under the vllm generator loader can run at interpreter
        # shutdown after the logging handlers are torn down (record silently dropped).
        if self.start_time is None:
            # stream_files() never ran, so there is nothing to report. A throughput log must never raise
            # (it would break get_tensors()), so bail out instead of computing timer() - None.
            return
        size = self.total_size
        elapsed_time = timer() - self.start_time
        throughput = size / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            f"[RunAI Streamer] Overall time to stream {humanize.naturalsize(size, binary=True)} of all files to {self.device_str}: {round(elapsed_time, 2)}s, {humanize.naturalsize(throughput, binary=True)}/s"
        )

    def stream_file(
            self,
            path: str,
            s3_credentials : Optional[S3Credentials] = None,
            device: Optional[str] = "cpu",
            is_distributed: bool = False,
            tensor_names: Optional[Collection[str]] = None,
        ) -> None:
        return self.stream_files([path], s3_credentials, device, is_distributed, tensor_names)


    def stream_files(
            self,
            paths: List[str],
            s3_credentials : Optional[S3Credentials] = None,
            device: Optional[str] = "cpu",
            is_distributed: bool = False,
            tensor_names: Optional[Collection[str]] = None,
        ) -> None:
        # None = no filter (today's behavior). Empty = certainly a caller bug, reject it up front.
        if tensor_names is not None and len(tensor_names) == 0:
            raise ValueError("tensor_names must not be empty - pass None to load every tensor")

        self.files_to_tensors_metadata = {}
        self.total_size = 0
        self.device_str = device

        file_stream_requests: List[FileChunks] = []

        # Start the throughput clock here, right before the first submission: prepare_request issues the
        # metadata reads (the first submissions), so the measurement spans first submission -> last response.
        self.start_time = timer()

        # metadata is created on cpu and each process reads it individually
        safetensors_metadatas = safetensors_pytorch.prepare_request(self.file_streamer, paths, s3_credentials)

        # Accumulates across every file - a name only counts as "unknown" if it matched in NONE
        # of them, not just the one being processed right now.
        found_names = set() if tensor_names is not None else None

        for i in range(len(paths)):
            (file_offset, tensors_metadata, tensor_sizes) = safetensors_metadatas[i]
            path = paths[i]

            if tensor_names is None:
                self.files_to_tensors_metadata[i] = tensors_metadata
                self.total_size += sum(tensor_sizes)
                file_stream_requests.append(FileChunks.contiguous(i, path, file_offset, tensor_sizes))
                continue

            # Filtered tensors may have gaps between them, so compute each one's absolute file
            # offset directly instead of the cumulative walk FileChunks.contiguous() does.
            kept_metadata = [tm for tm in tensors_metadata if tm.name in tensor_names]
            found_names.update(tm.name for tm in kept_metadata)
            kept_offsets = [file_offset + tm.offsets.start for tm in kept_metadata]
            kept_sizes = [tm.get_bytesize() for tm in kept_metadata]

            self.files_to_tensors_metadata[i] = kept_metadata
            self.total_size += sum(kept_sizes)
            file_stream_requests.append(FileChunks(i, path, kept_offsets, kept_sizes))

        if tensor_names is not None:
            missing = set(tensor_names) - found_names
            if missing:
                raise ValueError(f"tensor_names not found in checkpoint: {sorted(missing)}")

        self.file_streamer.stream_files(
            file_stream_requests,
            credentials=s3_credentials,
            device=device,
            is_distributed=is_distributed,
            tensor_names=tensor_names,
        )
        self._log_ring()

    def _log_ring(self) -> None:
        """Report the ring once per load, at the session boundary.

        Here rather than where the ring is built, for two reasons. The rank: a distributed load builds
        one ring per rank, and FilesRequestsIteratorWithBuffer cannot know which rank it is, so at INFO
        it would emit N unattributable copies. And the count: stream_files runs three times per load,
        twice for the safetensors metadata (`1 x 8 Bytes`), so only this one describes the model read.

        Worth INFO at all because the depth is DERIVED - from the limit, the ranks on the node and the
        largest tensor - not taken from the setting, so a ring clamped to one buffer is otherwise
        indistinguishable from the configured four, and they perform very differently. A failure to
        report must never break a load, hence the broad except."""
        try:
            ring = self.file_streamer.ring_info()
            if ring is None:
                return
            rank, num_buffers, buffer_size = ring
            logger.info(
                f"[RunAI Streamer] Rank {rank}: CPU ring {num_buffers} x "
                f"{humanize.naturalsize(buffer_size, binary=True)} = "
                f"{humanize.naturalsize(num_buffers * buffer_size, binary=True)} "
                f"for {humanize.naturalsize(self.total_size, binary=True)} in {len(self.files_to_tensors_metadata)} file(s)"
            )
        except Exception:
            logger.debug("[RunAI Streamer] could not report the CPU ring", exc_info=True)

    def get_tensors(self) -> Iterator[torch.tensor]:
        for file_index, ready_chunk_index, buffer in self.file_streamer.get_chunks():
            tensor_metadata = self.files_to_tensors_metadata[file_index][ready_chunk_index]
            yield tensor_metadata.name, safetensors_pytorch.create_torch_tensor(
                buffer, tensor_metadata
            )

        # All tensors delivered (the underlying get_chunks() is exhausted). Log the session throughput here,
        # during active consumption while the logging system is still alive.
        self._log_throughput()
