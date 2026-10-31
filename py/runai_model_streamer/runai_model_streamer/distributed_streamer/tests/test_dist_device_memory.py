# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import unittest
from unittest.mock import MagicMock, call, patch

import torch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    DEFAULT_BUFFER_MIN_BYTESIZE,
    DistributedStreamer,
    _distributedStreamer,
    allocate_dist_staging,
    dist_buffer_bytesize,
    dist_device_bytes_required,
    get_dist_buffer_alignment,
)

MIN_BYTESIZE_ENV = "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE"


class TestDistBufferSize(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_a_small_chunk_is_raised_to_the_default_minimum(self):
        self.assertEqual(dist_buffer_bytesize(2 * 1024 * 1024), DEFAULT_BUFFER_MIN_BYTESIZE)

    @patch.dict(os.environ, {}, clear=True)
    def test_a_chunk_above_the_minimum_is_used_as_it_is(self):
        chunk = 3 * DEFAULT_BUFFER_MIN_BYTESIZE
        self.assertEqual(dist_buffer_bytesize(chunk), chunk)

    @patch.dict(os.environ, {MIN_BYTESIZE_ENV: "0"}, clear=True)
    def test_the_minimum_can_be_turned_off(self):
        self.assertEqual(dist_buffer_bytesize(4096), 4096)


class TestDistDeviceBytesRequired(unittest.TestCase):
    """The check must ask for exactly what allocate_dist_staging() takes, or it guards the wrong number."""

    @patch.dict(os.environ, {MIN_BYTESIZE_ENV: "4096"}, clear=True)
    def test_it_equals_what_allocating_actually_takes(self):
        """Measured against a real allocation rather than restating the formula: a test that repeats the
        arithmetic agrees with it even when the arithmetic counts the wrong number of buffers.

        A staging buffer is handed out as a slice of its over-allocated base, and a slice shares that
        base's storage - so each tensor's own storage is what the device gave up, counted once."""
        chunk = 4096
        tensors = allocate_dist_staging(chunk, torch.device("cpu"))
        allocated = sum(t.untyped_storage().nbytes() for t in tensors)
        self.assertEqual(dist_device_bytes_required(chunk), allocated)

    @patch.dict(os.environ, {MIN_BYTESIZE_ENV: "4096"}, clear=True)
    def test_both_staging_buffers_start_aligned(self):
        data, received, _, _ = allocate_dist_staging(4096, torch.device("cpu"))
        alignment = get_dist_buffer_alignment()
        for name, buffer in (("data", data), ("received", received)):
            self.assertEqual(buffer.data_ptr() % alignment, 0, f"{name} buffer is not aligned")
            self.assertGreaterEqual(buffer.numel(), 4096, f"{name} buffer lost room to its own padding")

    @patch.dict(os.environ, {}, clear=True)
    def test_a_small_chunk_still_needs_two_minimum_buffers(self):
        required = dist_device_bytes_required(2 * 1024 * 1024)
        self.assertGreaterEqual(required, 2 * DEFAULT_BUFFER_MIN_BYTESIZE)


class DistributedJob:
    """A fake job seen from `my_rank`. Every rank reports its own entry of free_by_rank, and a reduction
    over the distribution group covers `group_ranks` - the whole job, or this node."""

    def __init__(self, free_by_rank, my_rank=0, group_ranks=None):
        self.free_by_rank = list(free_by_rank)
        self.my_rank = my_rank
        self.group_ranks = list(range(len(free_by_rank))) if group_ranks is None else list(group_ranks)
        self.group = MagicMock(name="distribution_group")
        self.events = MagicMock()      # records the order of group creation and reduction

    def streamer(self, max_chunk=4096):
        streamer = DistributedStreamer()
        streamer.params.max_chunk = max_chunk
        streamer.distributed_streamer.create_group = self.events.create_group
        streamer.distributed_streamer.release_group = self.events.release_group
        self.events.create_group.return_value = self.group
        return streamer

    @contextlib.contextmanager
    def patched(self):
        real_tensor = torch.tensor

        def broadcast(tensor, src=None, group=None):
            self.events.broadcast(src, group)

        def all_reduce(tensor, op=None, group=None):
            self.events.all_reduce(op, group)
            tensor.fill_(min(self.free_by_rank[rank] for rank in self.group_ranks))

        # Measurable on purpose: this harness exercises the memory DECISION, and the gate in front of
        # it has its own test on a genuinely unmeasurable device. Without this the check never runs on
        # a host with no CUDA, and a test asserting "keeps distributing" passes because nothing was
        # asked rather than because the memory was enough.
        with patch("runai_model_streamer.distributed_streamer.distributed_streamer."
                   "can_measure_device_memory", return_value=True), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=len(self.free_by_rank)), \
                patch("torch.distributed.get_backend", return_value="nccl"), \
                patch("torch.tensor", lambda data, dtype=None, device=None: real_tensor(data, dtype=dtype)), \
                patch("torch.zeros", lambda *args, **kwargs: real_tensor([0])), \
                patch("torch.distributed.get_global_rank", return_value=0), \
                patch("torch.distributed.broadcast", side_effect=broadcast), \
                patch("torch.distributed.all_reduce", side_effect=all_reduce), \
                patch.object(DistributedStreamer, "get_device_free_memory",
                             return_value=self.free_by_rank[self.my_rank]) as own_free:
            self.events.attach_mock(own_free, "read_free_memory")
            yield own_free


class TestOwnFreeMemory(unittest.TestCase):
    """This rank's own reading, before the reduction. It asks torch, so it covers every device torch
    supports - distributing moves torch tensors, and is not limited to what the C++ layer can write
    into."""

    def test_it_reads_through_torch_not_the_streamers_device_table(self):
        with patch("runai_model_streamer.distributed_streamer.distributed_streamer.device_free_memory",
                   return_value=1234) as read:
            self.assertEqual(DistributedStreamer().get_device_free_memory("cuda:3"), 1234)
        read.assert_called_once_with("cuda:3")

    def test_the_check_is_skipped_when_torch_cannot_measure_the_device(self):
        """Skipped, not failed. mem_get_info is missing for several devices torch supports, and
        reading "cannot measure" as "no memory" would fall back from a device that is fine.

        XPU is really unmeasurable here, not mocked into it: torch has the module but no
        mem_get_info, and once it gains one (2.6) is_available() is False on a host without the
        hardware. Either way the device cannot be measured, which is what this exercises."""
        streamer = DistributedStreamer()
        with patch.dict(os.environ, {"RUNAI_STREAMER_DIST": "1"}), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2), \
                patch("torch.distributed.get_backend", return_value="nccl"), \
                patch.object(DistributedStreamer, "get_min_device_free_memory") as reduce_free:
            streamer.set_is_distributed(True, "xpu:0")
        self.assertTrue(streamer.is_distributed, "an unmeasurable device must not stop distributing")
        reduce_free.assert_not_called()


class TestMinFreeMemory(unittest.TestCase):
    """The decision to distribute rests on the smallest free memory of the ranks in the distribution group."""

    def test_it_is_the_minimum_over_the_distribution_group(self):
        job = DistributedJob([50, 40, 90, 70], my_rank=2, group_ranks=[2, 3])
        with job.patched():
            self.assertEqual(job.streamer().get_min_device_free_memory("cuda:0"), 70)

    def test_the_communicator_is_created_before_free_memory_is_read(self):
        """The first collective takes the communicator's device memory, so it must come before the
        reading, or a rank just above the requirement passes and then cannot allocate its buffers."""
        job = DistributedJob([50, 40])
        streamer = job.streamer()
        with job.patched():
            streamer.get_min_device_free_memory("cuda:0")
        self.assertEqual(job.events.mock_calls,
                         [call.create_group(streamer.params),
                          call.broadcast(0, job.group),
                          call.read_free_memory("cuda:0"),
                          call.all_reduce(torch.distributed.ReduceOp.MIN, job.group)])

    def test_this_rank_reports_the_device_it_was_given(self):
        job = DistributedJob([50, 40])
        with job.patched() as own_free:
            job.streamer().get_min_device_free_memory("cuda:3")
        own_free.assert_called_once_with("cuda:3")


class TestDistributedMemoryCheck(unittest.TestCase):
    """set_is_distributed falls back when the smallest rank of the group cannot hold the staging buffers."""

    def setUp(self):
        # Class-wide, so the required size the test computes is the one the check computes.
        patcher = patch.dict(os.environ, {"RUNAI_STREAMER_DIST": "1", MIN_BYTESIZE_ENV: "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.required = dist_device_bytes_required(4096)

    def decide(self, free_by_rank, my_rank=0, group_ranks=None):
        job = DistributedJob(free_by_rank, my_rank=my_rank, group_ranks=group_ranks)
        streamer = job.streamer()
        with job.patched():
            streamer.set_is_distributed(True, "cuda:1")
        return streamer.is_distributed, job

    def test_enough_memory_keeps_distributed_streaming(self):
        distributed, _ = self.decide([self.required, self.required])
        self.assertTrue(distributed)

    def test_one_byte_short_falls_back(self):
        distributed, _ = self.decide([self.required - 1, self.required - 1])
        self.assertFalse(distributed)

    def test_a_rank_with_plenty_falls_back_when_a_peer_in_its_group_is_short(self):
        distributed, _ = self.decide([10 ** 12, self.required - 1], my_rank=0)
        self.assertFalse(distributed)

    def test_every_rank_of_a_group_reaches_the_same_decision(self):
        free = [10 ** 12, self.required - 1]
        self.assertEqual(self.decide(free, my_rank=0)[0], self.decide(free, my_rank=1)[0])

    def test_nodes_may_decide_differently_because_the_group_exists_before_anyone_decides(self):
        free = [10 ** 12, 10 ** 12, 10 ** 12, self.required - 1]        # node 0 is fine, node 1 is short
        node0, job0 = self.decide(free, my_rank=0, group_ranks=[0, 1])
        node1, job1 = self.decide(free, my_rank=2, group_ranks=[2, 3])
        self.assertEqual((node0, node1), (True, False))
        for job in (job0, job1):
            job.events.create_group.assert_called_once()     # created by the rank that falls back, too

    def test_a_rank_that_falls_back_keeps_the_group_it_created(self):
        """It cannot hand the group back on its own. Creating one needs every rank of the JOB, while
        this node decided by itself - so a rank left without a group would ask for a new one while the
        ranks that kept theirs never ask again. __exit__ releases it, where every rank does so
        together."""
        distributed, job = self.decide([self.required - 1, self.required - 1])
        self.assertFalse(distributed)
        job.events.release_group.assert_not_called()

    def test_a_rank_that_keeps_distributing_keeps_its_group(self):
        distributed, job = self.decide([self.required, self.required])
        self.assertTrue(distributed)
        job.events.release_group.assert_not_called()


class TestASecondReadOnTheSameStreamer(unittest.TestCase):
    """What the fallback must not cost the NEXT read.

    dist.new_group is a call every rank of the job has to make, members or not. The memory decision is
    taken per node. Tie the two together and the nodes stop asking for groups the same number of
    times, so the node that asks alone waits for ranks that are never coming.

    Counted rather than asserted on the release, because the count is what torch actually requires:
    the real create_distribution_group runs here, so its cache is part of what is under test."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"RUNAI_STREAMER_DIST": "1", MIN_BYTESIZE_ENV: "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.required = dist_device_bytes_required(4096)

    def two_reads(self, free_by_rank, my_rank, group_ranks, groups_by_ranks):
        """Two reads on one streamer. Returns the decision and the new_group calls each read made."""
        job = DistributedJob(free_by_rank, my_rank=my_rank, group_ranks=group_ranks)
        streamer = DistributedStreamer()
        streamer.params.max_chunk = 4096
        streamer.params.my_global_rank = my_rank
        streamer.params.groups_by_ranks = groups_by_ranks
        with job.patched(), \
                patch("torch.distributed.new_group",
                      side_effect=lambda **kwargs: MagicMock(name="group")) as new_group, \
                patch("torch.distributed.destroy_process_group"):
            streamer.set_is_distributed(True, "cuda:1")
            first = new_group.call_count
            streamer.set_is_distributed(True, "cuda:1")
            second = new_group.call_count - first
        return streamer.is_distributed, first, second

    def test_the_node_that_falls_back_asks_for_no_group_the_other_node_will_not_create(self):
        # node 0 has room, node 1 does not, so the two nodes reach different decisions
        free = [10 ** 12, 10 ** 12, 10 ** 12, self.required - 1]
        nodes = [[0, 1], [2, 3]]
        kept, kept_first, kept_second = self.two_reads(free, 0, [0, 1], nodes)
        fell, fell_first, fell_second = self.two_reads(free, 2, [2, 3], nodes)

        self.assertEqual((kept, fell), (True, False), "the two nodes did not decide differently")
        self.assertEqual(kept_first, fell_first, "the nodes already disagreed on the first read")
        self.assertEqual(
            fell_second, kept_second,
            f"the node that fell back asked for {fell_second} group(s) on the second read while the "
            f"node that kept streaming asked for {kept_second} - it would wait there until the timeout"
        )
        # The number itself, not only that the two agree: both asking again would also "agree", and
        # would still be one job-wide call per read that the cache exists to avoid.
        self.assertEqual(fell_second, 0, "a second read rebuilt groups that were already there")


class TestReleaseGroup(unittest.TestCase):
    def worker(self, group):
        worker = _distributedStreamer(None)
        worker.distribution_group = group
        return worker

    def test_it_destroys_the_group_and_forgets_it(self):
        group = MagicMock(name="group")
        worker = self.worker(group)
        with patch("torch.distributed.destroy_process_group") as destroy:
            worker.release_group()
        destroy.assert_called_once_with(group=group)
        self.assertIsNone(worker.distribution_group)

    def test_with_no_group_it_does_nothing(self):
        worker = self.worker(None)
        with patch("torch.distributed.destroy_process_group") as destroy:
            worker.release_group()
        destroy.assert_not_called()

    def test_exit_after_a_release_has_no_barrier_and_no_second_destroy(self):
        worker = self.worker(MagicMock(name="group"))
        with patch("torch.distributed.destroy_process_group"):
            worker.release_group()
        with patch("torch.distributed.barrier") as barrier, patch("torch.distributed.destroy_process_group") as destroy:
            worker.__exit__(None, None, None)
        barrier.assert_not_called()
        destroy.assert_not_called()

    def test_exit_still_runs_the_barrier_and_destroys_a_live_group(self):
        group = MagicMock(name="group")
        worker = self.worker(group)
        with patch("torch.distributed.barrier") as barrier, patch("torch.distributed.destroy_process_group") as destroy:
            worker.__exit__(None, None, None)
        barrier.assert_called_once_with(group=group)
        destroy.assert_called_once_with(group=group)
        self.assertIsNone(worker.distribution_group)


class TestARankThatCannotMeasure(unittest.TestCase):
    """What a failed memory query must NOT do: leave the reduction.

    The peers are already inside it, and they would stay there until the group timeout - waiting for
    a reading that is never coming. So the rank reduces a sentinel instead, and every rank of the
    group raises on the far side of it, together."""

    def reduce(self, mine, reduced):
        """One rank's trip through get_min_device_free_memory.

        `mine` is what its own query does - a value, or an exception to raise. `reduced` is what the
        group's MIN comes back as."""
        streamer = DistributedStreamer()
        streamer.params.max_chunk = 4096
        streamer.distributed_streamer.create_group = MagicMock(return_value=MagicMock())
        contributed = []
        real_tensor = torch.tensor

        def all_reduce(tensor, op=None, group=None):
            contributed.append(int(tensor.item()))
            tensor.fill_(reduced)

        with patch("torch.tensor", lambda data, dtype=None, device=None: real_tensor(data, dtype=dtype)), \
                patch("torch.zeros", lambda *args, **kwargs: real_tensor([0])), \
                patch("torch.distributed.get_global_rank", return_value=0), \
                patch("torch.distributed.broadcast"), \
                patch("torch.distributed.all_reduce", side_effect=all_reduce) as reduction, \
                patch.object(DistributedStreamer, "get_device_free_memory",
                             side_effect=mine if isinstance(mine, Exception) else None,
                             return_value=None if isinstance(mine, Exception) else mine):
            try:
                result = streamer.get_min_device_free_memory("cuda:0")
            except RuntimeError as error:
                result = error
        return result, contributed, reduction.call_count

    def test_it_reduces_a_sentinel_instead_of_leaving_its_peers_waiting(self):
        result, contributed, reductions = self.reduce(mine=RuntimeError("no mem_get_info"), reduced=-1)
        self.assertEqual(reductions, 1, "the rank left the reduction its peers were waiting in")
        self.assertEqual(contributed, [-1], "it reduced something a real reading could be")
        self.assertIsInstance(result, RuntimeError)

    def test_every_rank_of_the_group_raises_not_only_the_one_that_failed(self):
        """This rank measured its own device without trouble. It learns from the reduction that a
        peer did not, and fails at the same point rather than streaming on alone."""
        result, contributed, _ = self.reduce(mine=1 << 40, reduced=-1)
        self.assertEqual(contributed, [1 << 40], "this rank's own reading did not reach the group")
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("could not read the free memory", str(result))

    def test_a_group_that_could_all_measure_gets_its_smallest_reading(self):
        result, _, _ = self.reduce(mine=1 << 40, reduced=4096)
        self.assertEqual(result, 4096)



if __name__ == "__main__":
    unittest.main()
