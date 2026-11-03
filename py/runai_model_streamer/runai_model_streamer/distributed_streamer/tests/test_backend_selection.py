# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from runai_model_streamer.distributed_streamer import distributed_streamer
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    DistributedStreamer,
    get_backend_for_device,
    get_backends,
)


def init_string(backend: str, config: str):
    """Patches for a default process group created by init_process_group(backend)."""
    return (
        patch.object(distributed_streamer.dist, "is_initialized", return_value=True),
        patch.object(distributed_streamer.dist, "get_world_size", return_value=2),
        patch.object(distributed_streamer.dist, "get_backend", return_value=backend),
        patch.object(distributed_streamer.dist, "get_backend_config", return_value=config),
        patch.object(distributed_streamer.torch.cuda, "is_available", return_value=False),
    )


class TestBackendLookup(unittest.TestCase):
    """get_backends reads the backend of each device type, whichever way the group was initialized."""

    def lookup(self, backend, config):
        patches = init_string(backend, config)
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return get_backends()

    def test_device_prefixed_strings(self):
        self.assertEqual(
            self.lookup("cuda:nccl,cpu:gloo", "cuda:nccl,cpu:gloo"), {"cuda": "nccl", "cpu": "gloo"})
        self.assertEqual(self.lookup("cuda:nccl", "cuda:nccl"), {"cuda": "nccl"})

    def test_plain_strings(self):
        self.assertEqual(self.lookup("nccl", "cuda:nccl"), {"cuda": "nccl"})
        self.assertEqual(self.lookup("gloo", "cpu:gloo,cuda:gloo"), {"cpu": "gloo", "cuda": "gloo"})

    def test_device_spellings(self):
        self.lookup("cuda:nccl,cpu:gloo", "cuda:nccl,cpu:gloo")
        self.assertEqual(get_backend_for_device("cuda"), "nccl")
        self.assertEqual(get_backend_for_device("cuda:3"), "nccl")
        self.assertEqual(get_backend_for_device("cpu"), "gloo")
        self.assertIsNone(get_backend_for_device("xpu"))


class TestSetIsDistributed(unittest.TestCase):
    """set_is_distributed decides from the backend that serves the device."""

    def decide(self, backend, config, device, enable_dist="auto"):
        patches = init_string(backend, config)
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with patch.dict(os.environ, {"RUNAI_STREAMER_DIST": enable_dist}):
            with DistributedStreamer() as streamer:
                streamer.set_is_distributed(True, device)
                return streamer.is_distributed

    def test_auto_mode_distributes_on_nccl_whichever_way_it_was_initialized(self):
        for backend, config in [("nccl", "cuda:nccl"),
                                ("cuda:nccl", "cuda:nccl"),
                                ("cuda:nccl,cpu:gloo", "cuda:nccl,cpu:gloo")]:
            with self.subTest(backend=backend):
                self.assertTrue(self.decide(backend, config, "cuda"))

    def test_auto_mode_does_not_distribute_on_gloo(self):
        self.assertFalse(self.decide("gloo", "cpu:gloo,cuda:gloo", "cpu"))
        # The CPU backend of a mixed group is gloo as well.
        self.assertFalse(self.decide("cuda:nccl,cpu:gloo", "cuda:nccl,cpu:gloo", "cpu"))

    def test_nccl_does_not_serve_a_cpu_device(self):
        for backend, config in [("nccl", "cuda:nccl"), ("cuda:nccl", "cuda:nccl")]:
            with self.subTest(backend=backend):
                self.assertFalse(self.decide(backend, config, "cpu", enable_dist="1"))

    def test_gloo_does_not_serve_a_cuda_device(self):
        self.assertFalse(self.decide("gloo", "cpu:gloo,cuda:gloo", "cuda", enable_dist="1"))

    def test_forcing_distributed_still_allows_gloo_on_cpu(self):
        self.assertTrue(self.decide("gloo", "cpu:gloo,cuda:gloo", "cpu", enable_dist="1"))
        self.assertTrue(self.decide("cuda:nccl,cpu:gloo", "cuda:nccl,cpu:gloo", "cpu", enable_dist="1"))

    def test_disabled_stays_disabled(self):
        self.assertFalse(self.decide("cuda:nccl,cpu:gloo", "cuda:nccl,cpu:gloo", "cuda", enable_dist="0"))


if __name__ == "__main__":
    unittest.main()
