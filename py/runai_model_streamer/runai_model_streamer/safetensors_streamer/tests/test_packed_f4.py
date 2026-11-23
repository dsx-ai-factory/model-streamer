# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import struct
import tempfile
import unittest

import torch

from runai_model_streamer.safetensors_streamer import safetensors_pytorch
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import (
    SafetensorMetadata,
    create_torch_tensor,
)
from runai_model_streamer.safetensors_streamer.safetensors_streamer import SafetensorsStreamer

HAS_F4 = "F4" in safetensors_pytorch.safetensors_to_torch_dtype


def metadata(shape, nbytes, name="w"):
    return SafetensorMetadata(name, {"shape": shape, "dtype": "F4", "data_offsets": [0, nbytes]})


def write_f4_file(path, entries):
    """entries: (name, logical shape, payload bytes). Hand-built, so it needs no F4 in the installed
    safetensors; the header is padded to 8 bytes as the reference writer does."""
    header, data, offset = {}, b"", 0
    for name, shape, payload in entries:
        header[name] = {"dtype": "F4", "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        data += payload
        offset += len(payload)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(data)


def reference_reader(directory):
    """The safetensors reference loader, or None with the reason when this install cannot read F4.

    Asks with a known-good file instead of catching whatever the real comparison raises: a version
    without F4 rejects even a valid header (0.5.x: InvalidHeaderDeserialization, which does not say
    "F4"), so the error cannot tell "unsupported" from "the reader disagrees with us". Probed once, with
    input that is right by construction, any failure here means unsupported - and every failure after it
    is a real one."""
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        return None, f"safetensors is not installed: {error}"

    probe = os.path.join(directory, "probe.safetensors")
    write_f4_file(probe, [("probe", [2, 2], b"\x21\x43")])
    try:
        load_file(probe)
    except Exception as error:  # noqa: BLE001 - the probe input is valid, so any failure is "no F4"
        return None, f"the installed safetensors cannot read F4: {error}"
    return load_file, None


@unittest.skipUnless(HAS_F4, "this torch has no float4_e2m1fn_x2")
class TestPackedF4(unittest.TestCase):
    """Safetensors records LOGICAL dimensions; float4_e2m1fn_x2 holds two values per byte along the last
    axis, so the bytes and the torch shape both come from the packed shape."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_the_packed_shape_halves_the_last_axis(self):
        self.assertEqual(metadata([4, 8], 16).get_storage_shape(), [4, 4])

    def test_a_valid_tensor_is_accepted_and_reshaped(self):
        meta = metadata([2, 6], 6)
        buffer = torch.arange(6, dtype=torch.uint8).view(1, -1)

        tensor = create_torch_tensor(buffer, meta)

        self.assertEqual(tuple(tensor.shape), (2, 3))
        self.assertEqual(tensor.dtype, safetensors_pytorch.safetensors_to_torch_dtype["F4"])

    def test_one_byte_per_logical_value_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            metadata([2, 6], 12)
        self.assertIn("Corrupted Tensor", str(raised.exception))

    def test_a_wrong_byte_count_is_rejected(self):
        with self.assertRaises(ValueError):
            metadata([2, 6], 5)

    def test_an_odd_last_axis_cannot_be_represented(self):
        with self.assertRaises(ValueError) as raised:
            metadata([2, 5], 5)
        self.assertIn("cannot be represented", str(raised.exception))

    def test_an_odd_last_axis_is_rejected_even_when_the_total_is_whole_bytes(self):
        # [2, 3] is 6 values = 3 bytes, which the safetensors reference accepts. Torch packs along the
        # last axis, so there is no shape for it.
        with self.assertRaises(ValueError) as raised:
            metadata([2, 3], 3)
        self.assertIn("cannot be represented", str(raised.exception))

    def test_a_scalar_cannot_be_represented(self):
        with self.assertRaises(ValueError):
            metadata([], 1)

    def test_an_empty_tensor_has_the_packed_shape(self):
        buffer = torch.empty((1, 0), dtype=torch.uint8)

        meta = metadata([3, 0], 0)
        self.assertEqual(tuple(create_torch_tensor(buffer, meta).shape), (3, 0))

        meta = metadata([0, 4], 0)
        self.assertEqual(tuple(create_torch_tensor(buffer, meta).shape), (0, 2))

    def test_an_empty_tensor_keeps_the_device_of_the_buffer(self):
        # The meta device stands in for CUDA, as in test_empty_tensor_device.
        buffer = torch.empty((1, 0), dtype=torch.uint8, device="meta")

        tensor = create_torch_tensor(buffer, metadata([0, 4], 0))

        self.assertEqual(tensor.device.type, "meta")
        self.assertEqual(tuple(tensor.shape), (0, 2))

    def test_an_empty_tensor_with_an_odd_last_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            metadata([0, 3], 0)

    def test_the_streamer_loads_what_the_reference_reader_loads(self):
        path = os.path.join(self.temp_dir, "model.safetensors")
        payload = bytes(range(16))
        write_f4_file(path, [("a", [4, 8], payload), ("empty", [0, 4], b"")])

        with SafetensorsStreamer() as streamer:
            streamer.stream_file(path)
            tensors = dict(streamer.get_tensors())

        self.assertEqual(tuple(tensors["a"].shape), (4, 4))
        self.assertEqual(tensors["a"].view(torch.uint8).reshape(-1).tolist(), list(payload))
        self.assertEqual(tuple(tensors["empty"].shape), (0, 2))

        load_file, unsupported = reference_reader(self.temp_dir)
        if load_file is None:
            self.skipTest(unsupported)

        # No guard: past the probe, a reader error is a finding, not a reason to skip.
        reference = load_file(path)

        for name, tensor in tensors.items():
            self.assertEqual(tensor.shape, reference[name].shape)
            self.assertTrue(torch.equal(tensor.view(torch.uint8), reference[name].view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
