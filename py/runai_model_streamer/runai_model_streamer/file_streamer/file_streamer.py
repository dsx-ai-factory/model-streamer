from typing import Dict, List, Iterator, Optional, Tuple, Union
from runai_model_streamer.libstreamer.libstreamer import (
    runai_probe_direct_block_size,
    SUCCESS_ERROR_CODE,
    runai_start,
    runai_end,
    runai_set_credentials,
    runai_request,
    runai_response,
    runai_response_str,
    runai_list_files,
)
from runai_model_streamer.file_streamer.requests_iterator import (
    DIRECT_IO_BLOCK,
    FilesRequestsIteratorWithBuffer,
    FilesRequest,
    FileChunks,
)

from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
    is_s3_path,
    is_gs_path,
    is_azure_path,
    get_s3_credentials_module,
)

import torch

import logging

logger = logging.getLogger(__name__)

s3_credentials_module = get_s3_credentials_module()

class RunaiStreamerInvalidInputException(Exception):
    pass

def host_or_device(device: Optional[Union[str, torch.device]]) -> str:
    """The device as a string. Every spelling of the host becomes "cpu".

    Two decisions read this: whether the ring hands its buffer over, and whether get_chunks copies.
    They have to agree. Comparing the string makes them disagree for torch.device("cpu"), which is
    not equal to "cpu" but which .to() returns unchanged - so the buffer would be recycled under the
    caller and never copied. Only the host needs settling - a device destination copies either way,
    so its spelling is passed on as the caller wrote it rather than rewritten.

    Raises RunaiStreamerInvalidInputException for a device torch does not know. It used to travel as
    far as the first .to(), which is after a whole submission has been read."""
    if device is None:
        return "cpu"
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise RunaiStreamerInvalidInputException(f"Unusable device {device!r}: {error}") from error
    return "cpu" if resolved.type == "cpu" else str(device)

def homogeneous_paths(paths: List[str]) -> bool:
    if not paths:
        return True  # Empty list is homogeneous by default

    def path_type_fn(path: str):
        if is_s3_path(path):
            return is_s3_path
        elif is_gs_path(path):
            return is_gs_path
        elif is_azure_path(path):
            return is_azure_path
        else:
            return None

    first_type = path_type_fn(paths[0])
    for path in paths[1:]:
        if path_type_fn(path) != first_type:
            return False
    return True

class FileStreamer:
    def __init__(self) -> None:
        # Initialized here (not only in __enter__) so methods such as list_files
        # can be used without entering the context manager
        self.streamer = None
        self.s3_session = None
        self.s3_credentials = None    # resolved credentials (from handle_object_store)
        self._credentialed_streamer = None   # the C++ handle we already applied credentials to
        # Live submissions: id -> the request whose buffer it is filling. Several are in flight at once,
        # so responses are demuxed by id rather than assumed to belong to a single current submission.
        self.live_requests: Dict[int, FilesRequest] = {}
        self.outstanding = 0          # responses still owed across ALL live submissions

    def __enter__(self) -> "FileStreamer":
        self.streamer = runai_start()
        self.device_str = None
        self.s3_session = None
        self.s3_credentials = None
        self._credentialed_streamer = None
        return self

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        # Overall throughput is logged by SafetensorsStreamer (the user-facing session), not here: one session
        # spans multiple FileStreamer requests (already so under a memory limit, and more so with the ring
        # buffer's concurrent submissions), so it is not a property of any single request. __exit__ only does
        # the C++ handle cleanup.
        #
        # End AND clear the handle: list_files() now reuses self.streamer when set, so a call after the
        # context must not pass the freed C++ pointer to runai_list_files (a dangling handle -> crash/UAF).
        # Clearing makes post-context listing start a fresh temporary streamer (the owns_streamer path).
        if self.streamer:
            runai_end(self.streamer)
            self.streamer = None
            self._credentialed_streamer = None

    def handle_object_store(self, path: str, streamer, credentials: Optional[S3Credentials] = None) -> str:
        # Two independent concerns, deliberately gated separately so they can't desync:
        #   1. Resolve the credentials ONCE (self.s3_session is None). boto3 resolution is AWS-only, so it runs
        #      only for S3 paths (is_s3_path) - the backend type is only known from the path.
        #   2. Apply them ONCE PER C++ handle (streamer is not self._credentialed_streamer). Credentials are
        #      streamer-scoped state in the C++ layer, so a new handle (e.g. a temporary streamer created by an
        #      out-of-context list_files) needs its own set - the resolution flag must NOT suppress that.
        # Nothing is applied when there is nothing to apply (no credentials -> the C++ default/ambient provider
        # chain is used, and the streamer's set-once credentials stay free).
        if s3_credentials_module and is_s3_path(path):
            if self.s3_session is None:
                self.s3_session, self.s3_credentials = s3_credentials_module.get_credentials(credentials)
            if streamer is not None and self._has_credentials() and streamer is not self._credentialed_streamer:
                runai_set_credentials(streamer, self.s3_credentials)
                self._credentialed_streamer = streamer
        return path

    def _has_credentials(self) -> bool:
        # True when the resolved credentials carry at least one value (else there is nothing to set)
        c = self.s3_credentials
        return c is not None and any(
            v is not None for v in (c.access_key_id, c.secret_access_key, c.session_token, c.region_name, c.endpoint)
        )


    def list_files(
        self,
        prefix: str,
        is_recursive: bool = True,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> List[Tuple[str, int]]:
        # Listing reuses the streamer (its object-storage clients / backend handle / credentials).
        # When called outside the context manager (self.streamer is None), start a temporary
        # streamer just for this call and end it afterwards.
        owns_streamer = self.streamer is None
        streamer = runai_start() if owns_streamer else self.streamer

        results: List[Tuple[str, int]] = []
        try:
            # first object-storage path (the prefix): under RUNAI_STREAMER_NO_BOTO3_SESSION=0 this resolves S3
            # credentials from the environment via boto3 and applies them once; else the C++ default chain is
            # used. Inside the try so a temp streamer is still ended if credential resolution/apply throws.
            self.handle_object_store(prefix, streamer)
            runai_list_files(
                streamer,
                prefix,
                lambda path, size: results.append((path, size)),
                is_recursive=is_recursive,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        finally:
            if owns_streamer:
                runai_end(streamer)
        return results

    def stream_files(
            self,
            file_stream_requests: List[FileChunks],
            credentials: Optional[S3Credentials] = None,
            device: Optional[str] = "cpu",
            memory_limit: Optional[int] = None,
            owned: bool = True,
) -> None:
        # The previous stream has to be drained before another can start, because two things here are
        # single slots rather than per submission:
        #
        #   1. self.requests_iterator. Assigning the new one below drops the ONLY reference to the
        #      previous ring's pool - live_requests, which holds the other handle, is cleared a few
        #      lines after it - so numpy frees it while the C++ workers still hold raw range_dsts
        #      pointers into it. A use after free into reallocated heap, not merely a lost response.
        #   2. get_global_file_and_range resolves a response through self.requests_iterator. Even with
        #      the old pool kept alive, an earlier submission's buffer_index would be looked up against
        #      the NEW ring's buffer list: a different buffer, plausible looking bytes, and no error.
        #
        # Submission ids are unique and the responder demuxes by them, so attribution is not what
        # breaks - lifetime and ownership are. To lift this restriction and allow overlapping
        # streams: keep live_requests and outstanding across calls, hang the owning iterator off the
        # FilesRequest instead of off self, release buffers to that owner, and let each iterator live
        # until its last submission drains. Then delete this check.
        if self.outstanding > 0:
            raise ValueError(
                f"cannot start a new stream while {self.outstanding} response(s) are outstanding from "
                f"the previous one - consume or close get_chunks() before calling stream_files again"
            )

        if not homogeneous_paths([file_stream_request.path for file_stream_request in file_stream_requests]):
            raise RunaiStreamerInvalidInputException("Cannot stream files from multiple source types in parallel")

        # Normalised here too because this entry point is public in its own right.
        self.device_str = host_or_device(device)

        for file_stream_request in file_stream_requests:
            # first object-storage path resolves + applies the credentials to the streamer, once
            file_stream_request.path = self.handle_object_store(file_stream_request.path, self.streamer, credentials)

        # The ring is laid out to the block the mounts require, measured by the library.
        #
        # A submission is all filesystem or all object storage - mixing them is rejected with
        # UnsupportedBackendMix, and homogeneous_paths() above says the same thing on this side - so
        # the first path decides for the whole list. An object-storage submission has no mount, no
        # O_DIRECT and no alignment requirement, so it skips the probe entirely rather than asking a
        # question that can only answer "nothing to measure".
        #
        # Asked AFTER handle_object_store() above, so these paths are the ones that will be read.
        paths = [request.path for request in file_stream_requests]
        object_storage = bool(paths) and (is_s3_path(paths[0]) or is_gs_path(paths[0])
                                          or is_azure_path(paths[0]))

        direct_block = (DIRECT_IO_BLOCK if object_storage or not paths
                        else runai_probe_direct_block_size(self.streamer, paths))

        # Only a host destination hands its slots over. For any other, get_chunks yields
        # tensor.to(device) and the caller never sees the slot, so handing it over would throw away
        # a buffer per submission and fault a fresh one in, for nothing.
        self.requests_iterator: FilesRequestsIteratorWithBuffer = FilesRequestsIteratorWithBuffer.with_memory_mode(
            file_stream_requests, memory_limit, direct_block, owned and self.device_str == "cpu"
        )
        self.live_requests = {}
        self.outstanding = 0

        self._submit_while_buffers_free()

    def _submit_while_buffers_free(self) -> None:
        """Fill the ring: build and submit a request for every free buffer, until the data runs out.

        Keeping every buffer busy is the whole point - the C++ layer then always has work queued behind
        the submission being consumed, so the request boundary stops being a barrier. It is not extra
        parallelism: one submission already spans every worker (Assigner::assign divides its bytes across
        _num_workers).

        Removing that barrier is not the whole story, though, and the comment used to claim it was.
        Depth also helps when there is NO barrier to remove - a model small enough that the ring spans it
        and never recycles. Measured on Llama-3-8B over S3, 4 buffers beat 1 by 2.06% (n=11/10,
        permutation p=0.0043). The mechanism is not established; candidates are per-worker work
        granularity and interleaving at workload boundaries. Keep the depth, do not trust an argument
        that says it cannot matter here.

        The three range arrays are flat and grouped by file, in the order of paths - which is the order
        the request's files are in, and the order range_dsts was built in, so they pass through as is.

        range_dsts are RAW ADDRESSES into the ring's buffers: they keep nothing alive. A buffer must
        outlive its submission, until that submission's last response has been consumed - which is
        exactly when get_chunks releases it back to the pool."""
        while self.requests_iterator.has_free_buffer():
            request = self.requests_iterator.next_request()
            if request is None:
                return    # no data left to submit; the live submissions still have to be drained

            submission_id = runai_request(
                self.streamer,
                [file_request.path for file_request in request.files],
                [len(file_request.sizes) for file_request in request.files],
                [offset for file_request in request.files for offset in file_request.offsets],
                [size for file_request in request.files for size in file_request.sizes],
                request.range_dsts,
            )
            self.live_requests[submission_id] = request
            self.outstanding += request.num_ranges

    # This function iterates over the ready ranges of every live submission, in whatever order the C++
    # layer completes them. The indexes in a response are relative to its own submission and need to be
    # translated to the global index in the chunks list, which is what the response's request does.
    def get_chunks(self) -> Iterator:
        if not self.streamer:
            raise ValueError("Streamer not initialized")

        try:
            while self.live_requests:
                # runai_response blocks indefinitely (timeout 0) and returns None only on teardown. It
                # serves whichever submission completed a range first, so responses INTERLEAVE.
                response = runai_response(self.streamer)
                if response is None:
                    return    # teardown: no further responses are coming
                ret, submission_id, file_relative_index, range_relative_index, submission_done = response
                self.outstanding -= 1

                # Demux: the response names its own submission, and that submission names the request
                # whose frozen bases interpret it. An unknown id means a response from a submission this
                # streamer never made or has already retired - reading it would return wrong data. Use
                # if/raise (not assert) so this data-integrity check is not stripped under `python -O`.
                request = self.live_requests.get(submission_id)
                if request is None:
                    raise ValueError(f"response for unknown submission {submission_id}")

                # any per-sub-range error fails the whole stream (fail-fast); the finally below then
                # drains every live submission, not just this one
                if ret != SUCCESS_ERROR_CODE:
                    raise ValueError(
                        f"Could not receive response from libstreamer due to: {runai_response_str(ret)}"
                    )

                file_id, range_index, range_buffer = self.requests_iterator.get_global_file_and_range(
                    request, file_relative_index, range_relative_index
                )
                # create one dimensional tensor from the range buffer
                # we return a tensor of shape (1, range_buffer.size)
                # the data type of the original range_buffer, as created by the requests_iterator, is preserved (uint8)
                tensor = torch.from_numpy(range_buffer).view(1, -1)

                # currently file streamer is always reading a cpu buffer
                # so we don't need to move the tensor to the device
                # for future GDS/CUDA support we will need to move the tensor to the device (cpu or different device)
                if self.device_str == "cpu":
                    yield file_id, range_index, tensor
                else:
                    device_tensor = tensor.to(self.device_str)
                    yield file_id, range_index, device_tensor

                # Resumed past the yield, so the consumer is done with this tensor. If it was the
                # submission's last response, every view into its buffer has now been consumed - which
                # makes this the exact instant the buffer can be recycled. Refilling here is also what
                # provides backpressure: no new I/O is issued until a buffer actually comes free, so the
                # ring self throttles to consumer speed.
                if submission_done:
                    del self.live_requests[submission_id]
                    self.requests_iterator.release(request)
                    self._submit_while_buffers_free()
        finally:
            self.drain_live_submissions()

    def drain_live_submissions(self) -> None:
        """Consume the responses still owed by every live submission, however this generator ended.

        range_dsts are raw addresses into the ring's buffers, so a range still in flight writes into
        memory the next stream_files has already freed and handed to another submission - and its late
        response is then delivered to that submission's loop, which rejects it as unknown and abandons
        ITS responses in turn, so the streamer never recovers. A finally covers every exit: the
        fail-fast raises above, and a caller abandoning the generator (break, or raising inside its for
        loop - which safetensors_pytorch does), since closing a generator resumes it there."""
        # self.streamer is already cleared if the generator is collected after __exit__; runai_end has joined
        # the workers by then, so there is nothing left to wait for and the handle must not be touched.
        if self.streamer:
            while self.outstanding > 0:
                if runai_response(self.streamer) is None:
                    break   # teardown: no further responses are coming
                self.outstanding -= 1
        # Zero it even when the loop above could not run. Safe, and that needs saying, because this is
        # the counter stream_files' use-after-free guard reads: both paths that skip the loop mean the
        # responder is gone (runai_end has joined the workers), so no range can still be writing into
        # the pool. Leaving a stale count instead makes the guard reject every later stream_files on
        # this object - a FileStreamer re-entered after a generator outlived its context is bricked.
        self.outstanding = 0
        self.live_requests = {}

