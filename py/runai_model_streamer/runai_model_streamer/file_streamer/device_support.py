# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any, Callable, Dict, NamedTuple, Optional

import torch

from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA


class DeviceSupport(NamedTuple):
    """What the streamer needs from a device vendor in order to read straight into its memory.

    TWO OPERATIONS, and only these. Everything else the device ring does is already vendor neutral:
    torch allocates a slot for any device it knows, the ring's sizing takes free bytes as a parameter
    rather than querying, and the slot pool only ever asks an event `.query()` and `.synchronize()`.

    What belongs here: a fact needed only ABOUT THE DEVICES IN THIS TABLE. Events qualify - one is
    only ever created for a device the ring is already reading into. Free memory does not, and lives
    in torch_device: the distributed path asks it about devices this table will never list, because
    it moves torch tensors and so works wherever torch does.

    So adding a vendor is one row in the table below, a member of the C API's device enum, and the
    C++ implementation behind it - not a search for scattered torch.cuda calls."""

    # The C API device type for runai_request, and the ONE contract with the C ABI: it must be a
    # member of RunaiFileStreamerDeviceType, or the first submission is refused mid-stream.
    api_type: int
    # A fresh, unrecorded event. The ring records one per slot and refills the slot only once it has
    # completed - nothing else orders the C++ layer's next write against the caller's stream.
    new_event: Callable[[], Any]
    # The current stream OF A NAMED DEVICE, which is what the event must be recorded on.
    #
    # An event given no stream records on the device that is current, while a clone runs on the
    # device of the tensor it copies. Those are the same device for a process serving one GPU, which
    # is how vLLM and SGLang run - but the API takes any device string, and when they differ the
    # event times the wrong device and the slot is freed under the clones.
    current_stream: Callable[[str], Any]


_SUPPORTED: Dict[str, DeviceSupport] = {
    "cuda": DeviceSupport(
        api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA,
        new_event=torch.cuda.Event,
        current_stream=torch.cuda.current_stream,
    ),
}


def _is_rocm() -> bool:
    """Whether this torch is a ROCm build.

    ROCm reports its devices as "cuda" - the type string, torch.cuda.*, and torch.cuda.Event are all
    the same there as on NVIDIA. So the device type alone cannot tell the two apart, and the table
    below would hand an AMD device the NVIDIA row. The C++ layer then dlopen()s libcuda.so.1, which
    does not exist on such a machine, and the failure lands after the ring is allocated rather than
    where the device was chosen.

    torch.version.hip is the build's own answer: a version string on ROCm, None on CUDA."""
    return getattr(torch.version, "hip", None) is not None


def device_support(device: str) -> Optional[DeviceSupport]:
    """How to stream straight into `device`, or None when nothing here can.

    None covers the host and every device the C++ layer does not serve, because both mean the same
    thing to a caller: read into host memory and copy up, as before. A vendor that is absent from the
    table is therefore not an error - it is the path everyone used before device rings existed."""
    if not device:
        return None

    kind = torch.device(device).type

    # Keyed on the VENDOR, not on what torch calls the device. Every other accelerator torch knows
    # has a type string of its own - "xpu" for Intel - so the table refuses it by not listing it.
    # ROCm is the one that shares a string with the vendor above it.
    if kind == "cuda" and _is_rocm():
        return None

    return _SUPPORTED.get(kind)
