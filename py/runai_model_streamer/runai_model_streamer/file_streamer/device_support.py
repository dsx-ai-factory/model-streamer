from __future__ import annotations
from typing import Any, Callable, Dict, NamedTuple, Optional

import torch

from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA


class DeviceSupport(NamedTuple):
    """What the streamer needs from a device vendor in order to read straight into its memory.

    THREE OPERATIONS, and only these. Everything else the device ring does is already vendor neutral:
    torch allocates a slot for any device it knows, the ring's sizing takes free bytes as a parameter
    rather than querying, and the slot pool only ever asks an event `.query()` and `.synchronize()`.

    So adding a vendor is one row in the table below, a value in the C API's device enum, and the C++
    implementation behind it - not a search for scattered torch.cuda calls."""

    # The C API device type for runai_request. Must match this vendor's value in device.h.
    api_type: int
    # Free bytes on one device of this kind, named the way torch names it ("cuda:0"). Returns 0 when
    # the vendor is built in but has no usable device here, and both readers - the ring's sizing and
    # the distributed memory check - take 0 as "do not use this device". A device that does not exist
    # (a bad ordinal) still raises, like a malformed device string: an absent vendor is a fallback, a
    # wrong device id is a bug, and hiding it as 0 would silently downgrade the caller.
    free_memory: Callable[[str], int]
    # A fresh, unrecorded event. The ring records one per slot and refills the slot only once it has
    # completed - nothing else orders the C++ layer's next write against the caller's stream.
    new_event: Callable[[], Any]


def _cuda_free_memory(device: str) -> int:
    if not torch.cuda.is_available():
        return 0
    free_bytes, _total = torch.cuda.mem_get_info(device)
    return free_bytes


_SUPPORTED: Dict[str, DeviceSupport] = {
    "cuda": DeviceSupport(
        api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA,
        free_memory=_cuda_free_memory,
        new_event=torch.cuda.Event,
    ),
}


def device_support(device: str) -> Optional[DeviceSupport]:
    """How to stream straight into `device`, or None when nothing here can.

    None covers the host and every device the C++ layer does not serve, because both mean the same
    thing to a caller: read into host memory and copy up, as before. A vendor that is absent from the
    table is therefore not an error - it is the path everyone used before device rings existed."""
    if not device:
        return None
    return _SUPPORTED.get(torch.device(device).type)
