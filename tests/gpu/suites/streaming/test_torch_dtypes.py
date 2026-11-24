# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every dtype of the installed PyTorch that safetensors can write is in the streamer's table, under the name it writes.

A dtype that `save_file` refuses has no name in a safetensors file, so the streamer need not support it.
"""

import json
import os
import struct
import tempfile
import unittest

import torch
from safetensors import __version__ as SAFETENSORS_VERSION
from safetensors.torch import save_file

from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype

PROBE_BYTES = 16  # a whole number of elements of every dtype up to complex128


def torch_dtypes():
    found = set()
    for attribute in dir(torch):
        try:
            value = getattr(torch, attribute)
        except Exception:  # deprecated attributes can raise on access
            continue
        if isinstance(value, torch.dtype):
            found.add(value)
    return sorted(found, key=str)


def safetensors_name(directory, dtype):
    """(the header name safetensors writes for dtype, None), or (None, why it cannot store it)."""
    try:
        tensor = torch.zeros(PROBE_BYTES, dtype=torch.uint8).view(dtype)
    except Exception as error:
        return None, "torch cannot build a tensor of it (%s)" % type(error).__name__
    path = os.path.join(directory, "probe.safetensors")
    try:
        save_file({"probe": tensor}, path)
    except Exception as error:
        return None, "safetensors %s cannot write it (%s: %s)" % (SAFETENSORS_VERSION, type(error).__name__, error)
    with open(path, "rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_length))
    return header["probe"]["dtype"], None


class TorchDtypes(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="torch-dtypes-")

    def tearDown(self):
        self.directory.cleanup()

    def test_the_dtypes_of_torch_were_found(self):
        self.assertGreater(len(torch_dtypes()), 10, "torch %s: only %d dtypes found" % (torch.__version__,
                                                                                      len(torch_dtypes())))


def make_case(dtype):
    def case(self):
        name, why = safetensors_name(self.directory.name, dtype)
        if name is None:
            print("%s: not expected in the table, %s" % (dtype, why))
            return
        supported = safetensors_to_torch_dtype.get(name)
        if supported is None:
            self.fail("%s is stored by safetensors %s as %s, and the streamer has no such dtype: add it to "
                      "get_safetensors_dtype_map in safetensors_pytorch.py" % (dtype, SAFETENSORS_VERSION, name))
        self.assertIs(supported, dtype,
                      "safetensors %s writes %s for %s, and the streamer reads %s as %s" % (
                          SAFETENSORS_VERSION, name, dtype, name, supported))
        print("%s: supported, as %s" % (dtype, name))
    return case


for _dtype in torch_dtypes():
    setattr(TorchDtypes, "test_%s" % str(_dtype).replace(".", "_"), make_case(_dtype))


if __name__ == "__main__":
    unittest.main()
