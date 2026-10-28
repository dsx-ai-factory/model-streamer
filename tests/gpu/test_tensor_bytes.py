# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the byte view and the CRC32 that the GPU test compares tensors by, on CPU tensors (they need torch, no GPU).
They are skipped where torch is not installed; the GPU job installs it before it runs them.

    python3 -m unittest tests/gpu/test_tensor_bytes.py
"""

import os
import sys
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:  # a machine without torch: nothing to test here
    torch = None

if torch is not None:
    import tensor_bytes  # noqa: E402


@unittest.skipIf(torch is None, "torch is not installed")
class AsBytes(unittest.TestCase):
    def test_a_view_of_the_memory_of_every_kind_of_tensor(self):
        cases = [
            (torch.arange(6, dtype=torch.float32).reshape(2, 3), 24),
            (torch.arange(5, dtype=torch.int16), 10),
            (torch.tensor([True, False, True]), 3),
            (torch.arange(4, dtype=torch.bfloat16), 8),
            (torch.zeros(3, dtype=torch.complex64), 24),
            (torch.tensor(7, dtype=torch.int64), 8),
            (torch.zeros(0, dtype=torch.float32), 0),
            (torch.zeros(3, 0, 2, dtype=torch.int8), 0),
        ]
        for tensor, size in cases:
            flat = tensor_bytes.as_bytes(tensor)
            self.assertEqual((flat.dtype, flat.dim(), flat.numel()), (torch.uint8, 1, size), str(tensor.dtype))

    def test_the_dtypes_that_only_a_newer_torch_has(self):
        """float8, the unsigned 16/32/64 bit types and F4 (two values to an element): present from some torch version
        on, so each is tried where this torch has it, which is every one of them on the newest-torch leg."""
        names = ["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "uint16", "uint32", "uint64", "float4_e2m1fn_x2"]
        present = [name for name in names if hasattr(torch, name)]
        for name in present:
            dtype = getattr(torch, name)
            tensor = torch.zeros(6, dtype=torch.uint8).view(dtype) if name.startswith(("float8", "float4")) else \
                torch.zeros(3, dtype=dtype)
            flat = tensor_bytes.as_bytes(tensor)
            self.assertEqual((flat.dtype, flat.dim()), (torch.uint8, 1), name)
            self.assertEqual(flat.numel(), tensor.numel() * tensor.element_size(), name)
            before = tensor_bytes.tensor_crc32(tensor)
            flat[0] ^= 1
            self.assertEqual(int(tensor_bytes.as_bytes(tensor)[0]), 1, name)
            self.assertNotEqual(tensor_bytes.tensor_crc32(tensor), before, name)

    def test_it_is_a_view_so_a_write_reaches_the_tensor(self):
        for tensor in (torch.zeros(4, dtype=torch.float32), torch.zeros(2, dtype=torch.complex64),
                       torch.zeros(3, dtype=torch.bool)):
            tensor_bytes.as_bytes(tensor)[0] ^= 1
            self.assertEqual(int(tensor_bytes.as_bytes(tensor).sum()), 1, str(tensor.dtype))

    def test_a_bit_flip_changes_exactly_one_byte(self):
        tensor = torch.arange(64, dtype=torch.int32)
        before = tensor_bytes.as_bytes(tensor).clone()
        tensor_bytes.as_bytes(tensor)[5] ^= 1
        after = tensor_bytes.as_bytes(tensor)
        self.assertEqual(int((before != after).sum()), 1)
        self.assertNotEqual(int(before[5]), int(after[5]))

    def test_the_bytes_are_the_bytes_of_the_file_format(self):
        tensor = torch.tensor([1.0, -2.0], dtype=torch.float32)
        self.assertEqual(bytes(tensor_bytes.as_bytes(tensor).tolist()), tensor.numpy().tobytes())
        complex_tensor = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
        self.assertEqual(bytes(tensor_bytes.as_bytes(complex_tensor).tolist()), complex_tensor.numpy().tobytes())


@unittest.skipIf(torch is None, "torch is not installed")
class Crc32(unittest.TestCase):
    def test_it_is_the_crc32_of_the_bytes(self):
        tensor = torch.arange(1000, dtype=torch.int16)
        self.assertEqual(tensor_bytes.tensor_crc32(tensor), zlib.crc32(tensor.numpy().tobytes()))

    def test_slicing_does_not_change_it(self):
        tensor = torch.arange(1000, dtype=torch.int32)
        whole = tensor_bytes.tensor_crc32(tensor)
        for chunk in (1, 7, 4096, 4000, 4001):
            self.assertEqual(tensor_bytes.tensor_crc32(tensor, chunk=chunk), whole, chunk)

    def test_an_empty_tensor_has_the_crc_of_nothing(self):
        self.assertEqual(tensor_bytes.tensor_crc32(torch.zeros(0, dtype=torch.float32)), zlib.crc32(b""))

    def test_nan_payloads_and_negative_zero_are_told_apart(self):
        """A tensor is compared by its bytes, never by value: -0.0 == 0.0, and a NaN is not equal to itself."""
        positive = torch.tensor([0.0], dtype=torch.float32)
        negative = torch.tensor([-0.0], dtype=torch.float32)
        self.assertTrue(bool((positive == negative).all()))
        self.assertNotEqual(tensor_bytes.tensor_crc32(positive), tensor_bytes.tensor_crc32(negative))
        quiet = torch.tensor([0x7FC00000], dtype=torch.int32).view(torch.float32)
        payload = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
        self.assertNotEqual(tensor_bytes.tensor_crc32(quiet), tensor_bytes.tensor_crc32(payload))
        self.assertEqual(tensor_bytes.tensor_crc32(quiet), tensor_bytes.tensor_crc32(quiet.clone()))

    def test_a_flipped_bit_changes_it(self):
        tensor = torch.arange(100, dtype=torch.int64)
        before = tensor_bytes.tensor_crc32(tensor)
        tensor_bytes.as_bytes(tensor)[3] ^= 1
        self.assertNotEqual(tensor_bytes.tensor_crc32(tensor), before)


if __name__ == "__main__":
    unittest.main()
