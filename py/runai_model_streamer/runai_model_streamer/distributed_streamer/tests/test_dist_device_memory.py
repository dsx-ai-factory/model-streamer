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

        with patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=len(self.free_by_rank)), \
                patch("torch.distributed.get_backend", return_value="nccl"), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.tensor", lambda data, dtype=None, device=None: real_tensor(data, dtype=dtype)), \
                patch("torch.zeros", lambda *args, **kwargs: real_tensor([0])), \
                patch("torch.distributed.get_global_rank", return_value=0), \
                patch("torch.distributed.broadcast", side_effect=broadcast), \
                patch("torch.distributed.all_reduce", side_effect=all_reduce), \
                patch.object(DistributedStreamer, "get_cuda_free_memory",
                             return_value=self.free_by_rank[self.my_rank]) as own_free:
            self.events.attach_mock(own_free, "read_free_memory")
            yield own_free


class TestMinFreeMemory(unittest.TestCase):
    """The decision to distribute rests on the smallest free memory of the ranks in the distribution group."""

    def test_it_is_the_minimum_over_the_distribution_group(self):
        job = DistributedJob([50, 40, 90, 70], my_rank=2, group_ranks=[2, 3])
        with job.patched():
            self.assertEqual(job.streamer().get_min_cuda_free_memory("cuda:0"), 70)

    def test_the_communicator_is_created_before_free_memory_is_read(self):
        """The first collective takes the communicator's device memory, so it must come before the
        reading, or a rank just above the requirement passes and then cannot allocate its buffers."""
        job = DistributedJob([50, 40])
        streamer = job.streamer()
        with job.patched():
            streamer.get_min_cuda_free_memory("cuda:0")
        self.assertEqual(job.events.mock_calls,
                         [call.create_group(streamer.params),
                          call.broadcast(0, job.group),
                          call.read_free_memory("cuda:0"),
                          call.all_reduce(torch.distributed.ReduceOp.MIN, job.group)])

    def test_this_rank_reports_the_device_it_was_given(self):
        job = DistributedJob([50, 40])
        with job.patched() as own_free:
            job.streamer().get_min_cuda_free_memory("cuda:3")
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

    def test_a_rank_that_falls_back_releases_the_group_it_created(self):
        distributed, job = self.decide([self.required - 1, self.required - 1])
        self.assertFalse(distributed)
        job.events.release_group.assert_called_once_with()

    def test_a_rank_that_keeps_distributing_keeps_its_group(self):
        distributed, job = self.decide([self.required, self.required])
        self.assertTrue(distributed)
        job.events.release_group.assert_not_called()


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


if __name__ == "__main__":
    unittest.main()
