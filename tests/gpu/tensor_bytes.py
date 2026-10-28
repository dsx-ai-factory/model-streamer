# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bytes of a tensor, for the checks of the GPU test: a flat uint8 view, and its CRC32. Needs torch, no GPU: the
unit tests (test_tensor_bytes.py) run them on CPU tensors.

A tensor is compared by its bytes, never by value, so NaN payloads and negative zero compare right."""

import zlib

import torch

MIB = 1 << 20
D2H_CHUNK = 256 * MIB


def as_bytes(tensor):
    """The tensor's memory as a flat uint8 view. A complex tensor goes through view_as_real first, because older torch
    versions cannot view a complex tensor as bytes directly."""
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    return tensor.reshape(-1).view(torch.uint8)


def tensor_crc32(tensor, chunk=D2H_CHUNK):
    """CRC32 of the tensor's bytes, copied to the host in slices so a 2 GiB tensor needs no 2 GiB buffer."""
    flat = as_bytes(tensor)
    crc = 0
    for start in range(0, flat.numel(), chunk):
        crc = zlib.crc32(flat[start:start + chunk].cpu().numpy(), crc)
    return crc
