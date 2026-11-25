# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from runai_model_streamer.safetensors_streamer.safetensors_pytorch import (
    SafetensorMetadata,
    create_torch_tensor,
)


def empty_metadata(shape, dtype="F32"):
    return SafetensorMetadata("w", {
        "shape": shape,
        "dtype": dtype,
        "data_offsets": [0, 0],
    })


class TestEmptyTensorDevice(unittest.TestCase):
    """An empty tensor has no bytes to view, but must still land where the buffer was put - the
    device the caller asked for. The meta device stands in for CUDA, so this needs no GPU: it is a
    non-CPU device that torch.empty only reaches when told to."""

    def test_an_empty_tensor_is_on_the_device_of_the_buffer(self):
        buffer = torch.empty((1, 0), dtype=torch.uint8, device="meta")

        tensor = create_torch_tensor(buffer, empty_metadata([0]))

        self.assertEqual(tensor.device.type, "meta")

    def test_shape_and_dtype_are_kept(self):
        buffer = torch.empty((1, 0), dtype=torch.uint8, device="meta")

        tensor = create_torch_tensor(buffer, empty_metadata([3, 0, 2], "BF16"))

        self.assertEqual(tuple(tensor.shape), (3, 0, 2))
        self.assertEqual(tensor.dtype, torch.bfloat16)

    def test_an_empty_tensor_on_the_host_stays_on_the_host(self):
        buffer = torch.empty((1, 0), dtype=torch.uint8)

        tensor = create_torch_tensor(buffer, empty_metadata([0]))

        self.assertEqual(tensor.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
