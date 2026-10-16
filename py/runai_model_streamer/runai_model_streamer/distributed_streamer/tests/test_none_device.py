# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from runai_model_streamer.distributed_streamer.distributed_streamer import DistributedStreamer
from runai_model_streamer.file_streamer.file_streamer import runai_request
from runai_model_streamer.libstreamer.libstreamer import runai_start
from runai_model_streamer import SafetensorsStreamer
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
    RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME,
)

RANGE_SIZE = 8

# Two buffers of one range each against six ranges, so the ring has to recycle.
#
# The pad budget has to go. A slot is buffer_size + direct_block * pads, so under the DEFAULT budget
# a slot would be 512 KiB whatever the limit says, and a recycled slot would write each range at a
# fresh offset inside it. The ring would still recycle, but nothing would ever be overwritten, and
# both checks below would then pass with or without the bug.
TIGHT_RING = {
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "16",
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: "2",
    RUNAI_STREAMER_MAX_PADS_PER_BUFFER_ENV_VAR_NAME: "0",
}


def _check_group_read(path, rank, count, **kwargs):
    previous_override = os.environ.get("RUNAI_STREAMER_PROCESS_GROUP_SIZE")
    with patch("runai_model_streamer.file_streamer.file_streamer.runai_request", wraps=runai_request) as submit, \
            patch("runai_model_streamer.file_streamer.file_streamer.runai_start", wraps=runai_start) as start:
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(path, device="cpu", **kwargs)
            weights = dict(streamer.get_tensors())
            ring = streamer.file_streamer.ring_info()
            if ring is not None:
                assert ring[0] == rank
    torch.testing.assert_close(weights["weight"], torch.arange(6))
    start.assert_called_once_with(process_group_size=count)
    assert submit.call_count >= 2
    assert len({call.args[0].value for call in submit.call_args_list}) == 1
    assert os.environ.get("RUNAI_STREAMER_PROCESS_GROUP_SIZE") == previous_override


def _read_with_groups(rank, rendezvous, path):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8,
                            timeout=timedelta(seconds=90))
    try:
        os.environ.pop("RUNAI_STREAMER_PROCESS_GROUP_SIZE", None)

        # Eight independent readers size metadata for eight, with a single-reader handle still open.
        with SafetensorsStreamer() as single:
            single.stream_file(path, device="cpu", process_group=None)
            torch.testing.assert_close(dict(single.get_tensors())["weight"], torch.arange(6))
            _check_group_read(path, rank, 8)
        assert "RUNAI_STREAMER_PROCESS_GROUP_SIZE" not in os.environ
        dist.barrier()

        if rank == 7:
            with patch.object(dist, "new_group", side_effect=AssertionError("Unexpected discovery")), \
                    patch.object(dist, "all_gather_object", side_effect=AssertionError("Unexpected discovery")):
                _check_group_read(path, rank, 1, process_group=None)
        dist.barrier()

        # Non-contiguous global ranks catch discovery outside the supplied group and rank remapping.
        members = [1, 3, 5, 7]
        group = dist.new_group(ranks=members)
        if rank in members:
            with patch.object(dist, "all_gather_object", wraps=dist.all_gather_object) as discover, \
                    patch.object(dist, "broadcast", wraps=dist.broadcast) as broadcast, \
                    patch.object(dist, "destroy_process_group", wraps=dist.destroy_process_group) as destroy:
                _check_group_read(path, rank, 4, process_group=group, is_distributed=True)
            discovery_group = discover.call_args.kwargs["group"]
            distribution_group = broadcast.call_args.kwargs["group"]
            assert discovery_group is not group
            assert distribution_group is not group
            assert {call.kwargs["group"] for call in destroy.call_args_list} == {
                discovery_group, distribution_group,
            }
            dist.barrier(group=group)
            dist.destroy_process_group(group=group)
        dist.barrier()

        _check_group_read(path, rank, 8, is_distributed=True)
        dist.barrier()

        with patch.dict(os.environ, {"RUNAI_STREAMER_PROCESS_GROUP_SIZE": "4"}):
            _check_group_read(path, rank, 8)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestOwnedFollowsTheDestination(unittest.TestCase):
    """Only a host destination hands its buffers over, and only "cpu" equals "cpu".

    This class covers the entry point where that matters. stream_files decides whether the ring is
    handed over from `owned and device == "cpu"`, so a host written any other way turned `owned=True`
    into a recycled ring under tensors the caller had been promised.

    torch.device("cpu") is the dangerous one: .to() returns the tensor unchanged, so nothing copies
    and the caller is left holding the buffer. The indexed forms copy by accident - torch keeps an
    index the CPU has no use for - which hides the fault rather than fixing it."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write_ranges(self, count: int):
        """A file of `count` ranges, range i holding exactly "rng{i:05d}"."""
        expected = {i: f"rng{i:05d}" for i in range(count)}
        path = os.path.join(self.temp_dir, "none_device.txt")
        with open(path, "w") as handle:
            handle.write("".join(expected[i] for i in range(count)))
        return path, expected

    @patch.dict(os.environ, {"RUNAI_STREAMER_DIST_TIMEOUT": "15", "RUNAI_STREAMER_DIST": "1",
                             "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "4096"})
    def test_reader_counts_are_per_handle_and_preserve_user_overrides(self):
        """Reader groups size metadata, exclude idle peers, and preserve distributed payloads."""
        path = os.path.join(self.temp_dir, "reader_groups.safetensors")
        save_file({"weight": torch.arange(6)}, path)
        rendezvous = "file://" + os.path.join(self.temp_dir, "reader_group_rendezvous")
        mp.spawn(_read_with_groups, args=(rendezvous, path), nprocs=8, join=True)

    @patch.dict(os.environ, TIGHT_RING)
    def test_every_host_spelling_hands_the_ring_over(self):
        for device in [None, "cpu", "cpu:0", torch.device("cpu"), torch.device("cpu", 0)]:
            with self.subTest(device=repr(device)):
                path, expected = self.write_ranges(6)
                chunks = [FileChunks.contiguous(17, path, 0, [RANGE_SIZE] * 6)]

                with DistributedStreamer() as streamer:
                    streamer.stream_files(chunks, None, device=device, is_distributed=False, owned=True)
                    inner = streamer.file_streamer
                    self.assertEqual(inner.device_str, "cpu")
                    self.assertTrue(inner.requests_iterator._pool._owned)
                    held = {index: buffer for _, index, buffer in streamer.get_chunks()}

                # Every range got memory of its own. Under a recycled ring these six collapse to
                # two addresses holding the last two ranges twice over.
                self.assertEqual(len({b.data_ptr() for b in held.values()}), 6)

                self.assertEqual(
                    {i: b.numpy().tobytes().decode("utf-8") for i, b in held.items()},
                    expected,
                )

    @patch.dict(os.environ, TIGHT_RING)
    def test_a_device_destination_keeps_the_ring(self):
        """The other half of the mapping. A device destination yields tensor.to(device), a fresh
        allocation the caller already owns, so the ring stays ours to reuse however owned is set.

        Asserted on the ring, which needs no GPU: stream_files builds it, and nothing reaches CUDA
        until the first .to() in get_chunks."""
        for owned in (True, False):
            with self.subTest(owned=owned):
                path, _ = self.write_ranges(6)
                chunks = [FileChunks.contiguous(17, path, 0, [RANGE_SIZE] * 6)]

                with DistributedStreamer() as streamer:
                    streamer.stream_files(chunks, None, device="cuda", is_distributed=False,
                                          owned=owned)
                    inner = streamer.file_streamer

                    self.assertEqual(inner.device_str, "cuda")
                    self.assertFalse(inner.requests_iterator._pool._owned)

                    # Nothing consumed the submission - get_chunks would need a GPU - and the C layer
                    # must not be left holding destinations.
                    inner.drain_live_submissions()


if __name__ == "__main__":
    unittest.main()
