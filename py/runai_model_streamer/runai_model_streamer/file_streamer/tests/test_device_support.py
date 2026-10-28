# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.file_streamer.device_support import (
    DeviceSupport,
    _SUPPORTED,
    device_support,
)
from runai_model_streamer.libstreamer import (
    RUNAI_FILE_STREAMER_DEVICE_CPU,
    RUNAI_FILE_STREAMER_DEVICE_CUDA,
    RunaiFileStreamerDeviceType,
)


class TestDeviceSupport(unittest.TestCase):
    """One table says which devices can be read into directly. Adding a vendor is a row here, a value
    in the C API's device enum, and the C++ behind it - nothing else in the Python layer."""

    def test_the_host_is_not_a_device_to_stream_into(self):
        """Not an error: None means read into host memory and copy up, which is what always happened."""
        self.assertIsNone(device_support("cpu"))
        self.assertIsNone(device_support(""))

    def test_a_vendor_that_is_absent_is_simply_unsupported(self):
        self.assertIsNone(device_support("meta"), "an absent vendor falls back, it does not raise")

    def test_cuda_is_supported_and_carries_its_api_type(self):
        support = device_support("cuda:0")
        self.assertIsNotNone(support)
        self.assertEqual(support.api_type, RUNAI_FILE_STREAMER_DEVICE_CUDA)
        self.assertNotEqual(support.api_type, RUNAI_FILE_STREAMER_DEVICE_CPU)

    def test_the_ordinal_does_not_change_the_answer(self):
        self.assertIs(device_support("cuda"), device_support("cuda:3"))

    def test_every_row_supplies_both_operations(self):
        """The point of the record: a vendor cannot be half added. Missing one of these would fail
        later, inside a stream, rather than here."""
        for kind, support in _SUPPORTED.items():
            self.assertIsInstance(support, DeviceSupport, kind)
            self.assertIsInstance(support.api_type, int, kind)
            self.assertTrue(callable(support.new_event), f"{kind}: new_event")

    def test_every_row_names_a_type_the_c_api_serves(self):
        """The one contract between this table and the C ABI. A row with an invented value passes
        every other test here and is then refused by runai_request, with a submission already
        built."""
        for kind, support in _SUPPORTED.items():
            with self.subTest(kind=kind):
                RunaiFileStreamerDeviceType(support.api_type)      # raises if it is not a member

    def test_free_memory_is_not_one_of_them(self):
        """It is a torch question, answered in torch_device for any device torch supports. The
        distributed path asks it about devices this table does not list, so a field here would tie
        that decision to whether the C++ layer can write into the device."""
        self.assertNotIn("free_memory", DeviceSupport._fields)

    def test_no_row_claims_the_host_api_type(self):
        """CPU is the default in runai_request and means "these addresses are host memory". A device
        row claiming it would send device pointers down the host path."""
        for kind, support in _SUPPORTED.items():
            self.assertNotEqual(support.api_type, RUNAI_FILE_STREAMER_DEVICE_CPU, kind)

    def test_a_rocm_build_is_refused_although_torch_calls_it_cuda(self):
        """ROCm reports its devices as "cuda", so the type string cannot tell the vendors apart.

        Without this the AMD device takes the NVIDIA row, and the C++ layer dlopen()s libcuda.so.1 -
        which is not there - after the ring has already been allocated. None is the documented
        answer for a device this layer cannot serve: read into host memory and copy up."""
        with patch.object(torch.version, "hip", "6.2.41134"):
            self.assertIsNone(device_support("cuda:0"),
                              "a ROCm build was handed the NVIDIA row")
            self.assertIsNone(device_support("cuda"))

    def test_an_nvidia_build_still_gets_the_cuda_row(self):
        """The guard must refuse ROCm only. torch.version.hip is None on a CUDA build."""
        with patch.object(torch.version, "hip", None):
            support = device_support("cuda:0")
            self.assertIsNotNone(support, "an NVIDIA build was refused")
            self.assertEqual(support.api_type, RUNAI_FILE_STREAMER_DEVICE_CUDA)

    def test_a_vendor_with_its_own_type_string_needs_no_guard(self):
        """Every other accelerator torch knows names itself - "xpu" for Intel - so the table refuses
        it by not listing it. Pinned here so a future row cannot be added without noticing that ROCm
        is the one vendor that does NOT name itself."""
        with patch.object(torch.version, "hip", None):
            self.assertIsNone(device_support("xpu:0"))
            self.assertIsNone(device_support("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_the_cuda_row_actually_works(self):
        support = device_support("cuda:0")
        event = support.new_event()
        event.record()
        event.synchronize()
        self.assertTrue(event.query())


if __name__ == "__main__":
    unittest.main()
