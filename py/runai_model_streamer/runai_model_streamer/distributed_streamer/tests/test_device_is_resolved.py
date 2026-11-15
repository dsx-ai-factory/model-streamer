# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""resolve_device pins a device to an ordinal.

"cuda" means whichever device is current, and the answer can change. stream_files stores one value
that the staging buffers and the stream a release event is recorded on both read, so they must not
be free to drift onto different devices.

Calls the helper the product calls. A test that repeated the expression instead would pass however
the product resolved its device, including not resolving it at all."""

import unittest

import torch

from runai_model_streamer.file_streamer.file_streamer import resolve_device


class DeviceIsResolved(unittest.TestCase):
    def test_the_host_resolves_to_itself(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_a_device_without_an_ordinal_gains_one(self):
        self.assertEqual(resolve_device("cuda").index, torch.cuda.current_device())

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_an_ordinal_the_caller_gave_is_kept(self):
        self.assertEqual(resolve_device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0").index,
                         1 if torch.cuda.device_count() > 1 else 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_parsing_alone_would_not_resolve_it(self):
        """The bug this guards: torch.device("cuda") parses and keeps index None, so a helper that
        only parsed would fail here."""
        self.assertIsNone(torch.device("cuda").index)
        self.assertIsNotNone(resolve_device("cuda").index)


if __name__ == "__main__":
    unittest.main()
