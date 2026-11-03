# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A small safetensors file, written by hand (numpy and torch are not needed), for a test that needs a file and not a model:
a few tensors with known bytes, to put somewhere and read back. A suite that tests with models of every shape has a
generator of its own, which controls the layout in ways that a test of a feature does not care about.
"""

import json
import os
import struct
import zlib


def tensor_bytes(name, size):
    """size bytes that are a function of the name, so that a test can write a tensor and know it again: not random, not
    all the same, and different for different names."""
    seed = zlib.crc32(name.encode())
    return bytes((seed + 31 * index + (index >> 8)) & 0xFF for index in range(size))


def write_safetensors(path, tensors):
    """Write a safetensors file. tensors: (name, dtype, shape, data) in file order, data the bytes of the tensor (its size must
    be the elements of the shape times the size of the dtype). Returns {name: (dtype, shape, crc32 of the data)}."""
    header, position, described = {}, 0, {}
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [position, position + len(data)]}
        described[name] = (dtype, list(shape), zlib.crc32(data))
        position += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)  # tensors start on a multiple of 8, as the writers of safetensors pad
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded)
        for _, _, _, data in tensors:
            handle.write(data)
    return described


def small_model(directory, name="model.safetensors"):
    """A file of three tensors of different dtypes and sizes (about 1.2 MB), in directory. Returns (path, described), where
    described is what write_safetensors returns: the dtype, shape and CRC32 of each tensor, to compare what is read."""
    tensors = [
        ("model.embed_tokens.weight", "BF16", (256, 1024), tensor_bytes("embed", 256 * 1024 * 2)),
        ("model.layers.0.input_layernorm.weight", "F32", (1024,), tensor_bytes("norm", 1024 * 4)),
        ("model.layers.0.mlp.up_proj.weight", "F16", (512, 512), tensor_bytes("up", 512 * 512 * 2)),
    ]
    path = os.path.join(directory, name)
    return path, write_safetensors(path, tensors)
