# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Iterator, Optional, Tuple, Union
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
from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CPU
from runai_model_streamer.file_streamer.yield_order import yield_order_for
from runai_model_streamer.file_streamer.device_support import device_support
from runai_model_streamer.torch_device.torch_device import device_free_memory
from runai_model_streamer.file_streamer.requests_iterator import (
    DIRECT_IO_BLOCK,
    FilesRequestsIteratorWithBuffer,
    FilesRequest,
    FileChunks,
    SlotMemory,
    device_ring_budget,
    device_ring_floor,
    device_ring_sizing,
)
from runai_model_streamer.file_streamer.slot_pool import Slot

from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
    is_s3_path,
    is_gs_path,
    is_azure_path,
    get_s3_credentials_module,
)

import torch
from functools import partial
from humanize import naturalsize

import logging

logger = logging.getLogger(__name__)

# Resolved on FIRST USE, not at import. get_s3_credentials_module() already looks the module up
# lazily, but calling it here undid that: it imports runai_model_streamer_s3, which imports boto3,
# and building boto3's resource model costs ~160 ms of the ~170 ms this package takes to import -
# paid by every process, including a local load that never touches S3.
#
# _UNRESOLVED rather than None, because None is the answer when the module is not installed and must
# not send us looking again on every path.
_UNRESOLVED = object()
_s3_credentials_module = _UNRESOLVED


def s3_credentials_module():
    global _s3_credentials_module
    if _s3_credentials_module is _UNRESOLVED:
        _s3_credentials_module = get_s3_credentials_module()
    return _s3_credentials_module

class RunaiStreamerInvalidInputException(Exception):
    pass

# Enough for every dtype a safetensors file can carry - the widest element is 8 bytes. Torch's
# caching allocator is expected to hand back 512-byte blocks already, but slicing to an aligned start
# does not rely on that.
DEVICE_SLOT_ALIGNMENT = 512


def alloc_device_slot(slot_size: int, device: str) -> Slot:
    """One ring slot in DEVICE memory, for a caller streaming straight to a GPU.

    Over-allocated and sliced to an aligned start, so every range placed from the base keeps whatever
    alignment its own offset gives it.

    Here rather than beside the host allocator because this is the layer that knows the device, and
    the ring takes the allocator as a parameter for exactly that reason."""
    raw = torch.empty(slot_size + DEVICE_SLOT_ALIGNMENT, dtype=torch.uint8, device=device)
    shift = (-raw.data_ptr()) % DEVICE_SLOT_ALIGNMENT
    buffer = raw[shift: shift + slot_size]
    # From the BASE, not from the view: torch answers 0 for a zero-element tensor's data_ptr(), and a
    # ring of no ranges is a real case (an empty shard, or stream_files([])). numpy returns the true
    # pointer there, so taking it from the view would hand the library a null only on this path.
    return Slot(buffer, raw.data_ptr() + shift)


def device_slots(device: str, new_event: Callable[[], Any]) -> SlotMemory:
    """A ring made of device memory. direct_io=False because nothing reads straight into a device:
    the library fills these by copying out of its own host staging.

    new_event is passed in rather than looked up, so this names no vendor and a test can supply its
    own. Production takes it from DeviceSupport."""
    return SlotMemory(partial(alloc_device_slot, device=device), direct_io=False,
                      label=torch.device(device).type.upper(), new_event=new_event)


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
        # Here as well as in stream_files, so get_chunks on a streamer that never streamed can report
        # nothing rather than raise on its way out.
        # The ring, for as long as one is in use. None outside a stream: __exit__ frees it, so a
        # device ring does not hold device memory after the load it was built for.
        self.requests_iterator: Optional[FilesRequestsIteratorWithBuffer] = None
        # Where the ring's slots live, as runai_request needs it. Host until stream_files decides
        # otherwise: a wrong value here has the library write device addresses as host memory.
        self._device: Tuple[int, int] = (RUNAI_FILE_STREAMER_DEVICE_CPU, 0)

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

        # The ring too, and only AFTER runai_end: it has joined the workers, so no range can still be
        # writing into these slots - freeing them before that is a use after free. Dropped rather
        # than kept because a device ring holds up to its whole budget in device memory, and would
        # hold it for the life of this object, competing with the model it has just loaded.
        self.requests_iterator = None

    def handle_object_store(self, path: str, streamer, credentials: Optional[S3Credentials] = None) -> str:
        # Two independent concerns, deliberately gated separately so they can't desync:
        #   1. Resolve the credentials ONCE (self.s3_session is None). boto3 resolution is AWS-only, so it runs
        #      only for S3 paths (is_s3_path) - the backend type is only known from the path.
        #   2. Apply them ONCE PER C++ handle (streamer is not self._credentialed_streamer). Credentials are
        #      streamer-scoped state in the C++ layer, so a new handle (e.g. a temporary streamer created by an
        #      out-of-context list_files) needs its own set - the resolution flag must NOT suppress that.
        # Nothing is applied when there is nothing to apply (no credentials -> the C++ default/ambient provider
        # chain is used, and the streamer's set-once credentials stay free).
        # The PATH is tested first: resolving the module imports boto3, so asking about it for a local
        # path would pay that cost to answer a question the path has already settled.
        if is_s3_path(path) and s3_credentials_module():
            if self.s3_session is None:
                self.s3_session, self.s3_credentials = s3_credentials_module().get_credentials(credentials)
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
            ring_buffers: Optional[int] = None,
) -> None:
        """owned applies to a HOST destination only, and memory_limit to the host ring only.

        A caller always owns what it receives: to a device the tensor is a copy, by .to() out of a
        host ring or by a clone out of a device one, so the slot is never handed over and owned=False
        saves nothing there. It cannot be honoured literally either - yielding views would make a
        range's dtype view depend on its offset lining up with its element size, which the packing
        does not promise across mixed dtypes.

        memory_limit caps HOST memory, and says nothing about a GPU. A device ring is bounded by a
        fraction of free DEVICE memory, with RUNAI_STREAMER_DEVICE_MEMORY_LIMIT as a target that the
        largest range can raise - so the ring can exceed that setting, but never the fraction."""
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

        # Free the previous ring FIRST. Two reasons, and the order matters for both: assigning below
        # evaluates the new ring before dropping the old, so without this both exist at the peak -
        # two device rings is twice the budget - and the free memory read below would count the
        # previous ring's slots as taken, sizing the next one against a device that is not as full as
        # it looks. Safe here because the guard above proved nothing is in flight.
        self.requests_iterator = None

        # A ring in DEVICE memory when the destination is one the C++ layer can write into and it has
        # the room; otherwise the host ring and a copy per range, as before. The device ARGUMENT is
        # the switch - an engine asks for this by naming its device, which it has to do anyway, so
        # there is no second setting that could disagree with it. Decided before the probe below,
        # which it does not depend on: free memory and the range sizes name no mount.
        support = device_support(self.device_str)
        free = device_free_memory(self.device_str) if support else None
        sizing = device_ring_sizing(free, file_stream_requests, ring_buffers)

        if sizing is not None:
            buffer_size, num_buffers = sizing
            # DIRECT_IO_BLOCK rather than a probe: pread cannot target device memory, so the library
            # reads into its own pinned host staging and copies on from there. Nothing direct ever
            # lands on these slots, and the ring reads the block only when slots.direct_io is set.
            #
            # owned=False always, and memory_limit unread - see this method's docstring. A device
            # slot is reused, and the pool refuses an owned slot that is released with an event.
            self.requests_iterator = FilesRequestsIteratorWithBuffer(
                buffer_size, num_buffers, file_stream_requests, DIRECT_IO_BLOCK, owned=False,
                slots=device_slots(self.device_str, support.new_event),
            )
            # The index torch will ALLOCATE on, not the one the caller spelled: "cuda" with no
            # ordinal means the CURRENT device, which is not always 0, and the library has to be told
            # where the slots it writes into actually live. A vendor with no ordinals at all reports
            # None, and 0 is then the only device there is.
            ordinal = torch.empty(0, device=self.device_str).device.index
            self._device = (support.api_type, 0 if ordinal is None else ordinal)
        else:
            if support is not None:
                # The BUDGET, not free memory: it is the number that refused the ring, and a limit
                # far below free memory is the usual reason.
                why = ("torch cannot measure its free memory" if free is None else
                       f"its ring budget is "
                       f"{naturalsize(device_ring_budget(free, file_stream_requests), binary=True)} "
                       f"of the {naturalsize(free, binary=True)} free")
                logger.warning(
                    f"[RunAI Streamer] no device ring on {self.device_str}: {why}, and a ring needs "
                    f"at least {naturalsize(device_ring_floor(file_stream_requests), binary=True)}"
                    f" - reading into host memory and copying to the device instead"
                )
            # The ring is laid out to the block the mounts require, measured by the library.
            #
            # A submission is all filesystem or all object storage - mixing them is rejected with
            # UnsupportedBackendMix, and homogeneous_paths() above says the same thing on this side -
            # so the first path decides for the whole list. An object-storage submission has no mount,
            # no O_DIRECT and no alignment requirement, so it skips the probe entirely rather than
            # asking a question that can only answer "nothing to measure".
            #
            # Asked AFTER handle_object_store() above, so these paths are the ones that will be read.
            paths = [request.path for request in file_stream_requests]
            object_storage = bool(paths) and (is_s3_path(paths[0]) or is_gs_path(paths[0])
                                              or is_azure_path(paths[0]))

            direct_block = (DIRECT_IO_BLOCK if object_storage or not paths
                            else runai_probe_direct_block_size(self.streamer, paths))

            # Only a host ring read to a host destination hands its slots over, because that is the
            # only case where the caller receives the SLOT itself. A device destination gets a copy -
            # .to() out of this ring - so the slot is reused and handing it over would throw away a
            # buffer per submission and fault a fresh one in, for nothing.
            #
            # Keyed on the DESTINATION, never on "the caller cannot see the slot because we yield
            # tensor.to(device)": .to() RETURNS SELF when the device already matches, so that reading
            # hands the live slot back on any ring whose memory is already on the target.
            self._device = (RUNAI_FILE_STREAMER_DEVICE_CPU, 0)
            self.requests_iterator = FilesRequestsIteratorWithBuffer.with_memory_mode(
                file_stream_requests, memory_limit, direct_block,
                owned and self.device_str == "cpu", ring_buffers
            )
        # Built from the request list, which is where the file order is defined.
        self.yield_order = yield_order_for(file_stream_requests)

        self.live_requests = {}
        # Ranges of each live submission not yet HANDED OVER, which is not the same as not yet
        # received once responses are reordered. See the release in get_chunks.
        self.ranges_not_yielded = {}
        # Submissions the library has reported complete. Only to check our own counting - see the
        # release in get_chunks.
        self.reported_done = set()
        # When each submission's last RESPONSE arrived. Its slot is full from that moment but cannot
        # be refilled until the last range has been handed over, and the wait between the two is what
        # coarse recycling costs: one straggler pins the whole slot.
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
        while True:
            if not self.requests_iterator.has_free_buffer():
                if self.outstanding > 0:
                    return    # a response will free one; do not block the drain
                # Nothing in flight and nothing free: every slot is waiting on its clones. Returning
                # here would end the stream with data unsent - the caller's loop runs while
                # live_requests is non-empty, and we just emptied it.
                self.requests_iterator.wait_for_free_buffer()

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
                *self._device,
            )
            self.live_requests[submission_id] = request
            self.ranges_not_yielded[submission_id] = request.num_ranges
            self.outstanding += request.num_ranges

    # This function iterates over the ready ranges of every live submission, in whatever order the C++
    # layer completes them. The indexes in a response are relative to its own submission and need to be
    # translated to the global index in the chunks list, which is what the response's request does.
    def get_chunks(self) -> Iterator:
        if not self.streamer:
            raise ValueError("Streamer not initialized")

        try:
            # Asked once: the ring's memory does not change under a stream. Inside the try and behind
            # the live_requests check, because there is no iterator at all until stream_files has run
            # - get_chunks() before it is an empty generator, not an error.
            new_event = (self.requests_iterator.slots.new_event if self.live_requests else None)
            device_ring = new_event is not None

            while self.live_requests:
                # runai_response blocks indefinitely (timeout 0) and returns None only on teardown. It
                # serves whichever submission completed a range first, so responses INTERLEAVE.
                response = runai_response(self.streamer)
                if response is None:
                    return    # teardown: no further responses are coming
                # Not the release trigger any more: under reordering a submission's last response
                # can arrive before its earlier ranges have been handed over. Kept as a check on the
                # counting below, because the library sets it exactly once per submission.
                ret, submission_id, file_relative_index, range_relative_index, done = response
                self.outstanding -= 1

                # Demux: the response names its own submission, and that submission names the request
                # whose frozen bases interpret it. An unknown id means a response from a submission this
                # streamer never made or has already retired - reading it would return wrong data. Use
                # if/raise (not assert) so this data-integrity check is not stripped under `python -O`.
                request = self.live_requests.get(submission_id)
                if request is None:
                    raise ValueError(f"response for unknown submission {submission_id}")

                if done:
                    self.reported_done.add(submission_id)

                # any per-sub-range error fails the whole stream (fail-fast); the finally below then
                # drains every live submission, not just this one
                if ret != SUCCESS_ERROR_CODE:
                    raise ValueError(
                        f"Could not receive response from libstreamer due to: {runai_response_str(ret)}"
                    )

                file_id, range_index, range_buffer = self.requests_iterator.get_global_file_and_range(
                    request, file_relative_index, range_relative_index
                )
                # One dimensional, shape (1, n), uint8 as the requests_iterator made it. Built from
                # whatever the slot is MADE of: a host ring hands out a numpy view, a device ring a
                # torch one, and from_numpy accepts only the first.
                tensor = (range_buffer if torch.is_tensor(range_buffer)
                          else torch.from_numpy(range_buffer)).view(1, -1)

                # Park it, or take back the run it completes. Nothing is handed over for a response
                # that arrived early, so a response and a yield stop being one to one.
                #
                # The move to the device happens below, at the yield. A parked view costs nothing;
                # a parked device tensor would be a copy.
                #
                # Parking cannot stall the stream only because submissions are ISSUED in ordinal
                # order - next_request takes files in list order - so whatever the parked views are
                # waiting for is already in flight. Issue them out of that order and parked views
                # could hold every slot while the range they wait for has not been submitted.
                ready = self.yield_order.arrived(
                    file_id, range_index, (submission_id, request, file_id, range_index, tensor)
                )

                for ready_id, ready_request, ready_file, ready_index, ready_tensor in ready:
                    if device_ring:
                        # The owned tensor. Enqueued on the CALLER's stream and not waited on: stream
                        # order already puts it before whatever they do with the tensor next. It is
                        # also what makes the slot reusable at all - .to() below would return self.
                        ready_tensor = ready_tensor.clone()
                    elif self.device_str != "cpu":
                        ready_tensor = ready_tensor.to(self.device_str)
                    yield ready_file, ready_index, ready_tensor

                    # Resumed past the yield, so the consumer is done with this tensor. The buffer may
                    # be recycled once its LAST RANGE HAS BEEN HANDED OVER - not when its last response
                    # arrived, which under reordering is earlier and would recycle a slot that parked
                    # views still point into. Refilling here is also what provides backpressure: no new
                    # I/O is issued until a buffer actually comes free.
                    self.ranges_not_yielded[ready_id] -= 1
                    if self.ranges_not_yielded[ready_id] == 0:
                        # Our arithmetic against the library's own count. They can only disagree if
                        # this side is wrong, and the symptom would otherwise be a stuck generator:
                        # a slot never released, so no submission ever retires.
                        if ready_id not in self.reported_done:
                            raise ValueError(
                                f"submission {ready_id} ran out of ranges before the library "
                                f"reported it complete"
                            )
                        self.reported_done.discard(ready_id)
                        del self.ranges_not_yielded[ready_id]
                        del self.live_requests[ready_id]
                        # The clones out of this slot are enqueued, not finished. The event marks the
                        # last of them, and the pool holds the slot until it completes - nothing else
                        # orders the C++ layer's next write against our stream.
                        ready_event = None
                        if device_ring:
                            ready_event = new_event()
                            ready_event.record()
                        # The ring is gone once the context has exited, and a generator can still be
                        # resumed past that point. Nothing to hand the slot back to, and nothing will
                        # write into it again - runai_end joined the workers before it was freed.
                        if self.requests_iterator is not None:
                            self.requests_iterator.release(ready_request, ready_event)
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
        self.ranges_not_yielded = {}
        self.reported_done = set()

