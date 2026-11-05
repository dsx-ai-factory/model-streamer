# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.torch_device.torch_device import (
    can_measure_device_memory,
    device_free_memory,
)


class TestDeviceFreeMemory(unittest.TestCase):
    """Free memory is a torch question, asked about any device torch supports - not only the ones the
    C++ layer can read into."""

    def test_the_host_cannot_be_measured(self):
        """torch.cpu has no mem_get_info. Not an error: the callers skip or fall back."""
        self.assertFalse(can_measure_device_memory("cpu"))
        self.assertIsNone(device_free_memory("cpu"))

    def test_a_device_type_torch_has_no_module_for(self):
        self.assertFalse(can_measure_device_memory("meta"))
        self.assertIsNone(device_free_memory("meta"))

    def test_the_two_answers_agree(self):
        """can_measure_...() exists for a caller that must decide before reading - every rank enters
        the collective or none does - so it must predict exactly what the read will do."""
        for device in ("cpu", "meta", "cuda", "cuda:0"):
            with self.subTest(device=device):
                self.assertEqual(can_measure_device_memory(device),
                                 device_free_memory(device) is not None)

    def test_a_backend_that_exists_but_is_unavailable_cannot_be_measured(self):
        """torch ships the module whether or not the machine has the hardware, and mem_get_info
        raises there, so having the attribute is not enough."""
        with patch("torch.cuda.is_available", return_value=False):
            self.assertFalse(can_measure_device_memory("cuda:0"))
            self.assertIsNone(device_free_memory("cuda:0"))

    def test_a_malformed_device_string_raises(self):
        """Same rule as everywhere else here: an unmeasurable device is a fallback, a typo is a bug."""
        with self.assertRaises(RuntimeError):
            device_free_memory("bogus")

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_a_real_device_reports_what_torch_reports(self):
        self.assertTrue(can_measure_device_memory("cuda:0"))
        self.assertEqual(device_free_memory("cuda:0"), torch.cuda.mem_get_info("cuda:0")[0])


if __name__ == "__main__":
    unittest.main()
