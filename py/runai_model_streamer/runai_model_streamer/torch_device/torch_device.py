from __future__ import annotations
from typing import Any, Optional

import torch


def _memory_backend(device: str) -> Optional[Any]:
    """The torch module that can measure this device - torch.cuda for "cuda:0" - or None.

    mem_get_info is defined per backend and several that torch otherwise supports do not have it, so
    its presence is the question, not whether torch knows the device."""
    backend = getattr(torch, torch.device(device).type, None)
    if backend is None or not hasattr(backend, "mem_get_info") or not backend.is_available():
        return None
    return backend


def can_measure_device_memory(device: str) -> bool:
    """Whether device_free_memory() can answer for this device.

    Asked on its own by a caller that must decide BEFORE reading - a collective, say, which every
    rank has to enter or none."""
    return _memory_backend(device) is not None


def device_free_memory(device: str) -> Optional[int]:
    """Free bytes on a torch device, or None when torch cannot measure this kind.

    Asks torch, not the streamer's device table. That table says where the C++ layer can WRITE, which
    is narrower: distributed streaming moves torch tensors and so works on any device torch supports.

    None means UNKNOWN, never empty, and each caller decides what unknown means for it. Reading it as
    0 would be wrong in opposite directions: a device ring would refuse a device that is in fact
    empty enough, and the distributed check would fall back from a device that is fine."""
    backend = _memory_backend(device)
    return backend.mem_get_info(device)[0] if backend is not None else None
