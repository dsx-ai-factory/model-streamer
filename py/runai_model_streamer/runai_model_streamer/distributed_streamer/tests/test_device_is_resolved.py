# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""stream_files pins the device to an ordinal.

"cuda" means whichever device is current, and the answer can change. The staging buffers and the
stream a release event is recorded on both read this one value, so they must not be free to drift
onto different devices."""

import unittest

import torch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
)


class DeviceIsResolved(unittest.TestCase):
    def resolved(self, device):
        """What stream_files stores, without the rest of stream_files."""
        streamer = _distributedStreamer.__new__(_distributedStreamer)
        streamer.device = torch.empty(0, device=device).device
        return streamer.device

    def test_the_host_resolves_to_itself(self):
        self.assertEqual(self.resolved("cpu"), torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_a_device_without_an_ordinal_gains_one(self):
        self.assertEqual(self.resolved("cuda").index, torch.cuda.current_device())

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_an_ordinal_the_caller_gave_is_kept(self):
        self.assertEqual(self.resolved("cuda:0").index, 0)

    def test_parsing_alone_would_not_resolve_it(self):
        """The bug this guards: torch.device("cuda") parses and keeps index None."""
        self.assertIsNone(torch.device("cuda").index)


if __name__ == "__main__":
    unittest.main()
