from runai_model_streamer.libstreamer import (
    dll,
    t_streamer,
    RunaiFileStreamerDevice,
    RUNAI_FILE_STREAMER_DEVICE_CPU,
)
from typing import Callable, Dict, List, Optional, Tuple
import ctypes
import logging

log = logging.getLogger(__name__)

from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
)

SUCCESS_ERROR_CODE = 0
FINISHED_ERROR_CODE = 1
# common::ResponseCode::FileAccessError / TimedOut (see cpp/common/response_code/response_code.h)
FILE_ACCESS_ERROR_CODE = 2
TIMED_OUT_ERROR_CODE = 16

# Widths of the C types the range arrays are converted to (unsigned num_ranges, size_t offsets/sizes,
# void* destinations). ctypes wraps out-of-range values silently, so runai_request range-checks against
# these before converting.
UINT32_LIMIT = 1 << 32
UINT64_LIMIT = 1 << 64


def _credentials_param_arrays(s3_credentials: Optional[S3Credentials]):
    # Marshal S3Credentials into the (param_keys, param_values, num_params) key/value form used by
    # runai_set_credentials. Only set fields are included; keys are the plugin's canonical config-param names.
    items = []
    if s3_credentials is not None:
        for key, value in (
            ("access_key_id", s3_credentials.access_key_id),
            ("secret_access_key", s3_credentials.secret_access_key),
            ("session_token", s3_credentials.session_token),
            ("region", s3_credentials.region_name),
            ("endpoint", s3_credentials.endpoint),
        ):
            if value is not None:
                items.append((key, value))
    if not items:
        return None, None, 0
    keys_arr = (ctypes.c_char_p * len(items))(*[k.encode("utf-8") for k, _ in items])
    values_arr = (ctypes.c_char_p * len(items))(*[v.encode("utf-8") for _, v in items])
    return keys_arr, values_arr, len(items)


def runai_set_credentials(
    streamer: t_streamer,
    s3_credentials: Optional[S3Credentials] = None,
) -> None:
    """Set the streamer's object-storage credentials (set-once, streamer-scoped). Safe to call again with the
    same credentials; a different set after the first raises."""
    keys_arr, values_arr, num_params = _credentials_param_arrays(s3_credentials)
    error_code = dll.fn_runai_set_credentials(streamer, keys_arr, values_arr, num_params)
    if error_code != SUCCESS_ERROR_CODE:
        raise ValueError(
            f"Could not set credentials in libstreamer due to: {runai_response_str(error_code)}"
        )

def runai_start() -> t_streamer:
    streamer = t_streamer(0)
    error_code = dll.fn_runai_start(ctypes.byref(streamer))
    if error_code != SUCCESS_ERROR_CODE:
        # Changed from Exception to ValueError
        raise ValueError(
            f"Could not open streamer in libstreamer due to: {runai_response_str(error_code)}"
        )
    return streamer


def runai_end(streamer: t_streamer) -> None:
    return dll.fn_runai_end(streamer)

def runai_request(
    streamer: t_streamer,
    paths: List[str],
    num_ranges: List[int],
    range_offsets: List[int],
    range_sizes: List[int],
    range_dsts: List[int],
) -> int:
    """Multi-request submit. Non-blocking: returns the assigned submission id; use it to demux the
    responses from runai_response. Credentials are streamer-scoped (runai_set_credentials), not passed
    here. Many submissions may be in flight at once.

    A submission is a list of files, each carrying a list of RANGES. A range is an arbitrary
    (source offset, size) within its file with its own destination: ranges need not be contiguous in the
    file, need not be contiguous in memory, and need not be ordered. Exactly one response is issued per
    range, including a zero-sized one.

    paths carries one entry per file, however many ranges that file has. The three range arrays are flat,
    indexed identically, and grouped by file in the order of paths: file f's ranges occupy
    [sum(num_ranges[:f]), sum(num_ranges[:f+1])). Destinations must not overlap - that is the caller's
    responsibility and is not verified.

    range_dsts holds ABSOLUTE integer addresses - one complete pointer per range, not offsets from a
    base, and in no required order. Deliberately not memoryviews:
      - a destination need not be host memory at all. A CUDA device pointer has no Python buffer object,
        so a base-buffer + offsets signature would silently restrict destinations to the host.
      - destinations need not share a buffer, which is the same generality the range API exists for.
      - it avoids one memoryview -> address conversion per range on the submit path (~n per submission).
    The cost is that raw addresses keep nothing alive: THE CALLER MUST KEEP THE UNDERLYING BUFFERS ALIVE
    for the lifetime of the submission, until its last response has been consumed."""
    # Validate the shape here, at the FFI boundary: this is the last point at which a mismatch is a Python
    # error rather than undefined behaviour. The C side takes the range count as sum(num_ranges) and indexes
    # all three flat arrays up to it, while the arrays below are sized from len(range_sizes) - so a
    # sum(num_ranges) larger than the arrays reads past their end, in native code, with nothing raised.
    # ctypes will not catch it: an array built from too FEW initializers is silently zero-padded (a short
    # num_ranges would make a file lose all of its ranges, a short range_offsets would read from offset 0).
    # Costs one sum over the file count - tens per submission - not per range.
    num_files = len(paths)
    total_ranges = len(range_sizes)
    if len(num_ranges) != num_files:
        raise ValueError(
            f"num_ranges has {len(num_ranges)} entries but there are {num_files} paths"
        )
    if not (len(range_offsets) == len(range_dsts) == total_ranges):
        raise ValueError(
            f"range_offsets ({len(range_offsets)}), range_sizes ({total_ranges}) and range_dsts "
            f"({len(range_dsts)}) must be parallel"
        )

    # Range-check the VALUES before trusting them, because ctypes does not: it converts out-of-range
    # integers silently, so a Python-side check of the original list can agree while the C side sees
    # something else entirely. c_uint32(-1) becomes 4294967295 and c_uint32(2**32) becomes 0.
    # num_ranges is the dangerous one: [-1, 1] sums to 0 in Python - matching an empty range array and
    # passing the total check below - while C reads ~4 billion ranges for that file and indexes far past
    # the arrays. Checked per element; num_ranges is one entry per file, so this is tens of comparisons.
    if any(not (0 <= n < UINT32_LIMIT) for n in num_ranges):
        raise ValueError(
            f"every num_ranges value must fit an unsigned 32-bit integer, got {list(num_ranges)}"
        )
    if sum(num_ranges) != total_ranges:
        raise ValueError(
            f"sum(num_ranges) is {sum(num_ranges)} but the range arrays hold {total_ranges} entries"
        )

    # The flat arrays are checked with min()/max() rather than a per-element loop: they run at C speed, so
    # this stays negligible beside the ctypes conversion below even at hundreds of thousands of ranges,
    # whereas a Python-level loop over every range would not be. A negative size wraps to an enormous read
    # length, and a negative offset to an enormous seek.
    if total_ranges:
        if min(range_sizes) < 0 or max(range_sizes) >= UINT64_LIMIT:
            raise ValueError("every range size must fit an unsigned 64-bit integer")
        if min(range_offsets) < 0 or max(range_offsets) >= UINT64_LIMIT:
            raise ValueError("every range offset must fit an unsigned 64-bit integer")
        if min(range_dsts) < 0 or max(range_dsts) >= UINT64_LIMIT:
            raise ValueError("every range destination must be a valid address")

    c_paths = (ctypes.c_char_p * num_files)(*[path.encode("utf-8") for path in paths])
    c_num_ranges = (ctypes.c_uint32 * num_files)(*num_ranges)

    # The flat range arrays: one allocation each, no nested pointer array to keep alive across the call
    c_range_offsets = (ctypes.c_uint64 * total_ranges)(*range_offsets)
    c_range_sizes = (ctypes.c_uint64 * total_ranges)(*range_sizes)
    c_range_dsts = (ctypes.c_void_p * total_ranges)(*range_dsts)

    submission_id = ctypes.c_uint64()
    error_code = dll.fn_runai_request(
        streamer,
        ctypes.byref(submission_id),
        num_files,
        c_paths,
        c_num_ranges,
        c_range_offsets,
        c_range_sizes,
        c_range_dsts,
        # Always the host. range_dsts point into the ring's numpy buffer (requests_iterator.py,
        # `self._raw = np.empty(...)`), and the move to the caller's device happens afterwards in
        # FileStreamer.get_chunks, so the C layer is never asked for anything but CPU.
        RunaiFileStreamerDevice(RUNAI_FILE_STREAMER_DEVICE_CPU, 0),
    )
    if error_code != SUCCESS_ERROR_CODE:
        raise ValueError(
            f"Could not send runai_request to libstreamer due to: {runai_response_str(error_code)}"
        )
    return submission_id.value


def runai_response(
    streamer: t_streamer,
    timeout_ms: int = 0,
) -> Optional[Tuple[int, int, int, int, bool]]:
    """Multi-request response. Returns (response_code, submission_id, file_index, index, submission_done) for
    the next ready sub-range of any in-flight submission. response_code is that sub-range's own result -
    Success (0) or a specific per-sub-range error - and the other fields are valid in BOTH cases, so a
    multi-submission caller can attribute an error to its submission and still learn submission_done (whether
    that submission is now complete). A per-sub-range error is NOT raised here: the caller decides what to do
    with it (e.g. abort only on UnknownError, or fail just the affected submission and keep draining others).
    Returns None on teardown (FinishedError). Raises TimeoutError if timeout_ms elapses first (0 blocks
    indefinitely)."""
    submission_id = ctypes.c_uint64()
    file_index = ctypes.c_uint32()
    range_index = ctypes.c_uint32()
    submission_done = ctypes.c_int()
    error_code = dll.fn_runai_response(
        streamer,
        ctypes.byref(submission_id),
        ctypes.byref(file_index),
        ctypes.byref(range_index),
        ctypes.byref(submission_done),
        ctypes.c_uint(timeout_ms),
    )
    if error_code == FINISHED_ERROR_CODE:
        return None
    if error_code == TIMED_OUT_ERROR_CODE:
        raise TimeoutError(
            f"runai_response timed out after {timeout_ms} ms"
        )
    return (
        error_code,
        submission_id.value,
        file_index.value,
        range_index.value,
        bool(submission_done.value),
    )


def runai_response_str(response_code: int) -> str:
    return dll.fn_runai_response_str(response_code)


def runai_probe_direct_block_size(streamer, paths: List[str]) -> int:
    """The block destinations must be laid out at for these paths, so reads can use O_DIRECT.

    Asked of the library rather than restated here: this side lays out the ring, the library tests
    congruence against the same number, and a second copy could drift. A mismatch raises nothing - it
    just stops direct reads happening.

    Per REQUEST, not per process. The requirement belongs to the mount and a request can span several,
    so the library returns the largest any of them needs. Congruence at a power of two implies
    congruence at every smaller one, so that one number satisfies them all.

    The library probes - it opens and reads - so this is not free, but it is cached per mount.

    A failure to measure is not an error here: the library still returns a usable layout value (the
    host page size), and the next submission asks again. Nothing is cached on this side, so a mount
    that could not be probed today is measured tomorrow.
    """
    array = (ctypes.c_char_p * len(paths))(*[p.encode("utf-8") for p in paths])
    block = ctypes.c_size_t(0)

    error_code = dll.fn_runai_probe_direct_block_size(streamer, array, len(paths),
                                                      ctypes.byref(block))
    if error_code != 0:
        # Not raised. The value is still usable for layout, and the library logs why it could not
        # measure - aborting a model load because a probe was inconclusive would be far worse.
        log.debug(
            "Could not measure the direct-I/O block: %s. Laying out at %d bytes",
            runai_response_str(error_code), block.value)

    # A zero must never leave this function, and must not be papered over either. The caller divides
    # by this when placing ranges, so a zero becomes a ZeroDivisionError far from its cause - but
    # substituting a value would hide a library that broke its own contract. It promises a usable
    # block on every path, including the ones where it could not measure.
    if block.value == 0:
        raise RuntimeError(
            f"the streamer reported a direct-I/O block of 0 for {len(paths)} path(s), which cannot be "
            f"used to place destinations ({runai_response_str(error_code)})")

    return block.value


def runai_list_files(
    streamer: t_streamer,
    prefix: str,
    callback: Callable[[str, int], None],
    is_recursive: bool = True,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> None:
    # Credentials are streamer-scoped (runai_set_credentials); not passed here.
    # Exceptions raised inside a ctypes callback do not propagate through the C
    # call (they are routed to sys.unraisablehook). Capture the first one and
    # re-raise it after runai_list_files returns so callers can detect failures.
    callback_error: List[Exception] = []

    @dll.RunaiFileListCallback
    def _cb(path: bytes, size: int, _user_data: None) -> None:
        if callback_error:
            return
        try:
            callback(path.decode("utf-8"), size)
        except Exception as exc:
            callback_error.append(exc)

    def make_pattern_array(patterns):
        if not patterns:
            return None, 0
        arr = (ctypes.c_char_p * len(patterns))(
            *[p.encode("utf-8") for p in patterns]
        )
        return arr, len(patterns)

    allow_arr, num_allow = make_pattern_array(allow_patterns)
    ignore_arr, num_ignore = make_pattern_array(ignore_patterns)

    error_code = dll.fn_runai_list_files(
        streamer,
        prefix.encode("utf-8"),
        int(is_recursive),
        allow_arr, num_allow,
        ignore_arr, num_ignore,
        _cb, None,
    )
    # a callback failure is the root cause, so surface it before the error code
    if callback_error:
        raise callback_error[0]
    if error_code != SUCCESS_ERROR_CODE:
        raise ValueError(
            f"runai_list_files failed: {runai_response_str(error_code)}"
        )