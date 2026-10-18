import unittest

import torch

from runai_model_streamer.file_streamer.device_support import (
    DeviceSupport,
    _SUPPORTED,
    device_support,
)
from runai_model_streamer.libstreamer import (
    RUNAI_FILE_STREAMER_DEVICE_CPU,
    RUNAI_FILE_STREAMER_DEVICE_CUDA,
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

    def test_every_row_supplies_all_three_operations(self):
        """The point of the record: a vendor cannot be half added. Missing one of these would fail
        later, inside a stream, rather than here."""
        for kind, support in _SUPPORTED.items():
            self.assertIsInstance(support, DeviceSupport, kind)
            self.assertIsInstance(support.api_type, int, kind)
            self.assertTrue(callable(support.free_memory), f"{kind}: free_memory")
            self.assertTrue(callable(support.new_event), f"{kind}: new_event")

    def test_an_absent_vendor_reads_as_zero_rather_than_raising(self):
        """torch.cuda.mem_get_info raises on a host with no GPU, so the row has to guard it. 0 reads as
        "do not use this device" at both call sites. A bad ORDINAL is different and still raises - that
        is a caller bug, and 0 would turn it into a silent fallback."""
        for kind, support in _SUPPORTED.items():
            free = support.free_memory(f"{kind}:0")
            self.assertIsInstance(free, int, kind)
            self.assertGreaterEqual(free, 0, kind)

    def test_no_row_claims_the_host_api_type(self):
        """CPU is the default in runai_request and means "these addresses are host memory". A device
        row claiming it would send device pointers down the host path."""
        for kind, support in _SUPPORTED.items():
            self.assertNotEqual(support.api_type, RUNAI_FILE_STREAMER_DEVICE_CPU, kind)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_the_cuda_row_actually_works(self):
        support = device_support("cuda:0")
        self.assertGreater(support.free_memory("cuda:0"), 0)
        event = support.new_event()
        event.record()
        event.synchronize()
        self.assertTrue(event.query())


if __name__ == "__main__":
    unittest.main()
