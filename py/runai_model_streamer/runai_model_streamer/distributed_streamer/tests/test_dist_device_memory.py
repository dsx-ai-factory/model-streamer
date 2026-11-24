# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import unittest
from unittest.mock import MagicMock, call, patch

import torch

from runai_model_streamer.distributed_streamer import distributed_streamer
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    DEFAULT_BUFFER_MIN_BYTESIZE,
    DIST_BUFFER_ALIGNMENT,
    DistributedStreamer,
    _distributedStreamer,
    _distributedStreamerParams,
    allocate_dist_staging,
    dist_buffer_bytesize,
    dist_device_bytes_required,
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
        for rows in (1, 256, 75944):
            with self.subTest(rows=rows):
                tensors = allocate_dist_staging(chunk, torch.device("cpu"), rows=rows)
                allocated = sum(t.untyped_storage().nbytes() for t in tensors)
                self.assertEqual(dist_device_bytes_required(chunk, rows), allocated)

    @patch.dict(os.environ, {MIN_BYTESIZE_ENV: "4096"}, clear=True)
    def test_both_staging_buffers_start_aligned(self):
        data, received, _, _ = allocate_dist_staging(4096, torch.device("cpu"), rows=256)
        alignment = DIST_BUFFER_ALIGNMENT
        for name, buffer in (("data", data), ("received", received)):
            self.assertEqual(buffer.data_ptr() % alignment, 0, f"{name} buffer is not aligned")
            self.assertGreaterEqual(buffer.numel(), 4096, f"{name} buffer lost room to its own padding")

    @patch.dict(os.environ, {}, clear=True)
    def test_a_small_chunk_still_needs_two_minimum_buffers(self):
        required = dist_device_bytes_required(2 * 1024 * 1024, rows=256)
        self.assertGreaterEqual(required, 2 * DEFAULT_BUFFER_MIN_BYTESIZE)


# set_is_distributed is called here without a request, so the streamer's params carry the height an
# empty one gives. The check and this figure have to come from the same number or the test guards
# nothing.
EMPTY_REQUEST_ROWS = 1


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
                patch("torch.distributed.get_backend_config", return_value="cuda:nccl"), \
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
                patch("torch.distributed.get_backend", return_value="xccl"), \
                patch("torch.distributed.get_backend_config", return_value="xpu:xccl"), \
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

    def test_the_group_carries_one_collective_and_it_is_the_reduction(self):
        """The reduction is the only collective here, so it is the one that creates the communicator.
        The reading therefore does not include that memory, which is why we use half of it."""
        job = DistributedJob([50, 40])
        streamer = job.streamer()
        with job.patched():
            streamer.get_min_device_free_memory("cuda:0")
        self.assertEqual(job.events.mock_calls,
                         [call.create_group(streamer.params),
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
        self.required = dist_device_bytes_required(4096, EMPTY_REQUEST_ROWS)

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


class TestTheAgreedFreeMemoryIsKept(unittest.TestCase):
    """set_is_distributed reduces a free-memory figure; a ring every rank sends from is sized from
    THAT, not from what this rank happens to have free. So it has to survive the decision."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"RUNAI_STREAMER_DIST": "1", MIN_BYTESIZE_ENV: "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.required = dist_device_bytes_required(4096, EMPTY_REQUEST_ROWS)

    def decide(self, free_by_rank, is_distributed=True, device="cuda:1"):
        job = DistributedJob(free_by_rank, my_rank=0)
        streamer = job.streamer()
        with job.patched():
            streamer.set_is_distributed(is_distributed, device)
        return streamer

    def test_the_group_minimum_is_kept(self):
        streamer = self.decide([10 ** 12, self.required + 7])

        self.assertEqual(streamer.params.min_device_free_memory, self.required + 7,
                         "not this rank's own figure, the group's smallest")

    def test_it_is_kept_even_when_the_check_turns_distribution_off(self):
        """The figure is the group's whatever the decision was. A rank that fell back still agreed
        it, and reading it afterwards must not get this rank's own."""
        streamer = self.decide([10 ** 12, self.required - 1])

        self.assertFalse(streamer.is_distributed)
        self.assertEqual(streamer.params.min_device_free_memory, self.required - 1)

    def test_a_stream_that_never_asked_has_no_figure(self):
        self.assertIsNone(self.decide([10 ** 12, 10 ** 12],
                                      is_distributed=False).params.min_device_free_memory)

    def test_a_stale_figure_is_cleared_before_the_next_decision(self):
        """This object is reused across loads. A figure measured before the last model occupied the
        device would size the next ring far too large, and nothing downstream can tell."""
        job = DistributedJob([10 ** 12, 10 ** 12], my_rank=0)
        streamer = job.streamer()
        with job.patched():
            streamer.set_is_distributed(True, "cuda:1")
        self.assertIsNotNone(streamer.params.min_device_free_memory)

        with job.patched():
            streamer.set_is_distributed(False, "cuda:1")
        self.assertIsNone(streamer.params.min_device_free_memory,
                          "the figure from the previous load is still here")


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
        self.required = dist_device_bytes_required(4096, EMPTY_REQUEST_ROWS)

    def two_reads(self, free_by_rank, my_rank, group_ranks, groups_by_ranks):
        """Two reads on one streamer. Returns the decision and the new_group calls each read made."""
        job = DistributedJob(free_by_rank, my_rank=my_rank, group_ranks=group_ranks)
        streamer = DistributedStreamer()
        streamer.params.max_chunk = 4096
        streamer.params.my_global_rank = my_rank
        streamer.params.groups_by_ranks = groups_by_ranks
        # A fresh module-level group cache per simulated rank: in a real job each rank is its own
        # process, so two ranks simulated in one process must not see each other's cache.
        with job.patched(), \
                patch.object(distributed_streamer, "_group_cache", {}), \
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

    def test_it_forgets_the_group_and_keeps_it_for_the_next_stream(self):
        # The group is NOT destroyed on the healthy path: gloo teardown can block in the destructor
        # and corrupt the rank's store client, which is why groups are cached for reuse instead.
        group = MagicMock(name="group")
        worker = self.worker(group)
        with patch("torch.distributed.destroy_process_group") as destroy:
            worker.release_group()
        destroy.assert_not_called()
        self.assertIsNone(worker.distribution_group)

    def test_a_poisoned_group_is_evicted_and_retired(self):
        # The error path: a failed stream may have closed the group's pairs, so the next stream
        # must not inherit it from the cache - and it is retired, never destroyed, because gloo
        # teardown is the blocking hazard the cache exists to avoid.
        group = MagicMock(name="group")
        worker = self.worker(group)
        with patch.object(distributed_streamer, "_group_cache",
                          {("local", ((0, 1),), "test"): [group]}) as cache, \
                patch("torch.distributed.destroy_process_group") as destroy:
            worker.release_group(poisoned=True)
            destroy.assert_not_called()
            self.assertIsNone(worker.distribution_group)
            self.assertNotIn(("local", ((0, 1),), "test"), cache)
            self.assertIn([group], cache[("retired",)])

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

    def test_exit_still_runs_the_barrier_and_keeps_a_live_group(self):
        group = MagicMock(name="group")
        worker = self.worker(group)
        with patch("torch.distributed.barrier") as barrier, patch("torch.distributed.destroy_process_group") as destroy:
            worker.__exit__(None, None, None)
        barrier.assert_called_once_with(group=group)
        destroy.assert_not_called()
        self.assertIsNone(worker.distribution_group)

    def test_exit_with_an_exception_evicts_the_group_without_a_barrier(self):
        group = MagicMock(name="group")
        worker = self.worker(group)
        with patch.object(distributed_streamer, "_group_cache", {("global", 2, "test"): group}) as cache, \
                patch("torch.distributed.barrier") as barrier, \
                patch("torch.distributed.destroy_process_group") as destroy:
            worker.__exit__(RuntimeError, RuntimeError("stream failed"), None)
            barrier.assert_not_called()
            destroy.assert_not_called()
            self.assertIsNone(worker.distribution_group)
            self.assertNotIn(("global", 2, "test"), cache)
            self.assertIn(group, cache[("retired",)])

    def test_a_caught_stream_error_still_poisons_the_group(self):
        # The caller can catch the stream's failure inside the with block, so __exit__ sees no
        # exception - but a timed-out broadcast closed the group's pairs all the same. is_error is
        # the record of that, and it must evict the group and skip the barrier exactly as a raise
        # would: a barrier on closed pairs blocks, and the next load would vote the group healthy.
        group = MagicMock(name="group")
        worker = self.worker(group)
        worker.is_error = True
        with patch.object(distributed_streamer, "_group_cache", {("global", 2, "test"): group}) as cache, \
                patch("torch.distributed.barrier") as barrier, \
                patch("torch.distributed.destroy_process_group") as destroy:
            worker.__exit__(None, None, None)
            barrier.assert_not_called()
            destroy.assert_not_called()
            self.assertIsNone(worker.distribution_group)
            self.assertNotIn(("global", 2, "test"), cache)
            self.assertIn(group, cache[("retired",)])

    def test_a_device_backed_group_is_destroyed_at_release(self):
        # The opposite treatment from gloo, on purpose: a device-backed group holds a communicator
        # on the device, taken before vLLM measures the KV cache, so keeping it would shrink the
        # cache for the life of the server. Its backend was named at creation, so it owns no gloo
        # backend and destroying it is safe - master's behavior, restored.
        group = MagicMock(name="group")
        worker = self.worker(group)
        worker.group_is_cached = False
        with patch("torch.distributed.destroy_process_group") as destroy:
            worker.release_group()
        destroy.assert_called_once_with(group=group)
        self.assertIsNone(worker.distribution_group)

    def test_exit_runs_the_barrier_then_destroys_a_device_backed_group(self):
        group = MagicMock(name="group")
        worker = self.worker(group)
        worker.group_is_cached = False
        with patch("torch.distributed.barrier") as barrier, \
                patch("torch.distributed.destroy_process_group") as destroy:
            worker.__exit__(None, None, None)
        barrier.assert_called_once_with(group=group)
        destroy.assert_called_once_with(group=group)
        self.assertIsNone(worker.distribution_group)


class TestOnlyGlooGroupsAreCached(unittest.TestCase):
    """The backend decides the group's lifecycle: gloo is cached and voted, a device backend is
    created per stream with no cache entry and no vote - the vote would charge every load a
    world-wide all_reduce to learn an answer that is always 'miss'. The backend is always NAMED on
    new_group, so the group never copies the default map, which may hold a gloo backend beside the
    device's."""

    def worker(self, backend, groups_by_ranks=None):
        worker = _distributedStreamer(None)
        worker.distribution_backend = backend
        worker.my_global_rank = 0
        worker.groups_by_ranks = groups_by_ranks or [[0, 1]]
        worker.broadcast_timeout = "test-timeout"
        return worker

    @contextlib.contextmanager
    def job(self, cache):
        with patch.object(distributed_streamer, "_group_cache", cache), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2), \
                patch.dict(os.environ, {"RUNAI_STREAMER_DIST_GLOBAL": "1"}), \
                patch("torch.distributed.new_group",
                      side_effect=lambda **kwargs: MagicMock(name="group")) as new_group, \
                patch("torch.distributed.all_reduce") as vote:
            yield new_group, vote

    def test_the_backend_is_named_on_every_new_group(self):
        worker = self.worker("nccl")
        with self.job({}) as (new_group, _):
            worker.create_distribution_group()
        self.assertEqual(new_group.call_args.kwargs.get("backend"), "nccl")

    def test_an_unnamed_backend_takes_the_default_map(self):
        # None means no caller named one - only then may the group copy the default backend map,
        # and it is cached, because that map may include gloo.
        worker = self.worker(None)
        with self.job({}) as (new_group, _):
            worker.create_distribution_group()
        self.assertNotIn("backend", new_group.call_args.kwargs)
        self.assertTrue(worker.group_is_cached)

    def test_a_gloo_group_is_cached(self):
        cache = {}
        worker = self.worker("gloo")
        with self.job(cache) as (new_group, _):
            group = worker.create_distribution_group()
        self.assertTrue(worker.group_is_cached)
        self.assertIn(group, cache.values())

    def test_a_device_backed_group_is_not_cached_and_not_voted(self):
        cache = {("discovery",): MagicMock(name="discovery")}
        worker = self.worker("nccl")
        with self.job(cache) as (new_group, vote):
            group = worker.create_distribution_group()
        self.assertFalse(worker.group_is_cached)
        self.assertNotIn(group, cache.values())
        vote.assert_not_called()

    def test_a_partial_creation_failure_retires_what_was_built(self):
        # The creator builds several groups; if a later new_group raises, the earlier ones must not
        # lose their last reference while the exception unwinds - their destructors would run, the
        # blocking teardown this module avoids. They are retired instead.
        first = MagicMock(name="first_group")
        cache = {}
        worker = self.worker("gloo", groups_by_ranks=[[0], [1]])
        with patch.object(distributed_streamer, "_group_cache", cache), \
                patch("torch.distributed.new_group",
                      side_effect=[first, RuntimeError("the second creation failed")]):
            with self.assertRaises(RuntimeError):
                worker._new_distribution_groups(worker.groups_by_ranks)
        self.assertIn([first], cache[("retired",)])


class TestAFailedSetupFallsBackWithinTheLoad(unittest.TestCase):
    """A discovery or group-creation failure must not fail the load.

    The failure already ended distribution for the JOB (_lose_distribution) - but vLLM loads once
    per worker, so a raise out of stream_files kills the worker and the server over a stream that
    still has a working plain path. The rank that observes the failure falls back within its own
    load; only a failure that is NOT a recorded loss of distribution - a programming error - may
    still raise."""

    def setUp(self):
        self.dist = MagicMock(name="dist")
        self.dist.is_initialized.return_value = True
        self.dist.get_world_size.return_value = 2
        self.dist.get_rank.return_value = 0
        self.dist.get_backend_config.return_value = "cpu:gloo"
        patches = [
            patch.object(distributed_streamer, "dist", self.dist),
            patch.object(distributed_streamer, "_group_cache", {}),
            patch.object(distributed_streamer, "_topology_cache", None),
            patch.object(distributed_streamer, "_cache_job", None),
            patch.object(distributed_streamer, "_distribution_lost_for_job", object()),
            patch.dict(os.environ, {"RUNAI_STREAMER_DIST": "1"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def streamer(self):
        streamer = DistributedStreamer()
        streamer.file_streamer = MagicMock(name="file_streamer")
        return streamer

    def lose_and_raise(self, *_args, **_kwargs):
        distributed_streamer._lose_distribution()
        raise RuntimeError("the collective failed")

    def test_a_discovery_failure_streams_this_load_plainly(self):
        streamer = self.streamer()
        with patch.object(_distributedStreamerParams, "find_local_ranks",
                          side_effect=self.lose_and_raise):
            streamer.stream_files([], None, "cpu", True)
        self.assertFalse(streamer.is_distributed)
        streamer.file_streamer.stream_files.assert_called_once()

    def test_a_group_creation_failure_streams_this_load_plainly(self):
        streamer = self.streamer()

        def discovered():
            distributed_streamer._topology_cache = (2, 0, [[0, 1]])
            return 2, 0, [[0, 1]]

        with patch.object(_distributedStreamerParams, "find_local_ranks", side_effect=discovered), \
                patch.object(_distributedStreamer, "create_group", side_effect=self.lose_and_raise):
            streamer.stream_files([], None, "cpu", True)
        self.assertFalse(streamer.is_distributed)
        streamer.file_streamer.stream_files.assert_called_once()

    def test_a_failure_that_is_not_a_loss_of_distribution_still_raises(self):
        streamer = self.streamer()
        with patch.object(_distributedStreamerParams, "find_local_ranks",
                          side_effect=RuntimeError("a programming error")):
            with self.assertRaises(RuntimeError):
                streamer.stream_files([], None, "cpu", True)
        streamer.file_streamer.stream_files.assert_not_called()


class TestTheCacheIsVoted(unittest.TestCase):
    """cached_or_created reuses an entry only when EVERY rank still holds one.

    Eviction is rank-local knowledge - in per-node mode a failure on one node never reaches the
    others - and creating a group is a collective, so a rank that recreates alone waits for peers
    that never come. The vote over the kept discovery group is what keeps hit and miss identical."""

    def vote(self, have_mine, group_min):
        """One rank's trip through cached_or_created, with the group's MIN coming back as group_min."""
        key = ("global", 2, "test")
        cache = {("discovery",): MagicMock(name="discovery")}
        old = MagicMock(name="old_group")
        if have_mine:
            cache[key] = old

        def all_reduce(tensor, op=None, group=None):
            tensor.fill_(group_min)

        created = MagicMock(name="created_group")
        with patch.object(distributed_streamer, "_group_cache", cache), \
                patch("torch.distributed.all_reduce", side_effect=all_reduce):
            returned = _distributedStreamer.cached_or_created(key, lambda: created)
        return returned, old, created, cache

    def test_everyone_has_it_so_it_is_reused(self):
        returned, old, created, cache = self.vote(have_mine=True, group_min=1)
        self.assertIs(returned, old)
        self.assertNotIn(("retired",), cache)

    def test_a_peer_evicted_so_everyone_recreates_and_this_ranks_entry_is_retired(self):
        returned, old, created, cache = self.vote(have_mine=True, group_min=0)
        self.assertIs(returned, created)
        self.assertIn(old, cache[("retired",)])

    def test_this_rank_evicted_so_everyone_recreates(self):
        returned, old, created, cache = self.vote(have_mine=False, group_min=0)
        self.assertIs(returned, created)
        self.assertNotIn(("retired",), cache)

    def test_an_instance_reused_across_jobs_rediscovers_the_layout(self):
        # A params object can outlive init_process_group; the layout it discovered belongs to the
        # dead job and must not be served to the new one.
        params = _distributedStreamerParams()
        old_job, new_job = MagicMock(name="old_job"), MagicMock(name="new_job")
        with patch.object(_distributedStreamerParams, "find_local_ranks",
                          side_effect=[(2, 0, [[0, 1]]), (4, 0, [[0, 1, 2, 3]])]) as find, \
                patch.object(distributed_streamer, "_group_cache", {}), \
                patch.object(distributed_streamer, "_topology_cache", None), \
                patch.object(distributed_streamer, "_cache_job", None), \
                patch.dict(os.environ, {"RUNAI_STREAMER_PROCESS_GROUP_SIZE": "ignored"}), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.group") as group_module:
            group_module.WORLD = old_job
            params.discover_local_ranks()
            self.assertEqual(params.num_processes_on_node, 2)
            group_module.WORLD = new_job
            params.discover_local_ranks()
        self.assertEqual(find.call_count, 2, "the dead job's layout was served to the new job")
        self.assertEqual(params.num_processes_on_node, 4)

    def test_a_reused_instance_rediscovers_after_the_module_topology_was_dropped(self):
        # A failed vote retires the discovery group and drops the module topology. A reused
        # instance under the SAME job must not serve its own copy of the layout: that would skip
        # find_local_ranks, never recreate the discovery group, and silently skip later votes.
        params = _distributedStreamerParams()
        job = MagicMock(name="job")
        with patch.object(_distributedStreamerParams, "find_local_ranks",
                          return_value=(2, 0, [[0, 1]])) as find, \
                patch.object(distributed_streamer, "_group_cache", {}), \
                patch.object(distributed_streamer, "_topology_cache", None), \
                patch.object(distributed_streamer, "_cache_job", job), \
                patch.dict(os.environ, {"RUNAI_STREAMER_PROCESS_GROUP_SIZE": "ignored"}), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.group") as group_module:
            group_module.WORLD = job
            params.discover_local_ranks()
            self.assertEqual(find.call_count, 1)

            # The healthy fast path: module topology warm, instance warm - no rediscovery.
            distributed_streamer._topology_cache = (2, 0, [[0, 1]])
            params.discover_local_ranks()
            self.assertEqual(find.call_count, 1, "a warm cache should not rediscover")

            # A failed vote dropped the module topology; the instance must follow.
            distributed_streamer._topology_cache = None
            params.discover_local_ranks()
            self.assertEqual(find.call_count, 2, "the instance served the dropped layout")

    def test_a_group_held_across_jobs_is_not_returned(self):
        # A worker can hold the dead job's distribution group; the new job must get a fresh one.
        worker = _distributedStreamer(None)
        worker.distribution_group = MagicMock(name="old_group")
        old_job, new_job = MagicMock(name="old_job"), MagicMock(name="new_job")
        worker.group_job = old_job
        worker.groups_by_ranks = [[0, 1]]
        worker.my_global_rank = 0
        fresh = MagicMock(name="fresh_group")
        with patch.object(distributed_streamer, "_group_cache", {}), \
                patch.object(distributed_streamer, "_topology_cache", None), \
                patch.object(distributed_streamer, "_cache_job", old_job), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.group") as group_module, \
                patch("torch.distributed.new_group", return_value=fresh), \
                patch("torch.distributed.get_world_size", return_value=2):
            group_module.WORLD = new_job
            got = worker.create_distribution_group()
        self.assertIs(got, fresh)
        self.assertIs(worker.group_job, new_job)

    def test_a_failed_creation_loses_distribution_too(self):
        # After a miss, a rank can enter group creation while a peer's vote failed or answered
        # differently, so a creation failure is possibly asymmetric - and it consumed a
        # group-creation ordinal the ranks can no longer re-agree on. It must flip the same switch
        # a failed vote does; the retired discovery group is the proof _lose_distribution ran.
        key = ("global", 2, "test")
        discovery = MagicMock(name="discovery")
        cache = {("discovery",): discovery}

        def all_reduce(tensor, op=None, group=None):
            tensor.fill_(0)

        def explode():
            raise RuntimeError("new_group timed out")

        with patch.object(distributed_streamer, "_group_cache", cache), \
                patch.object(distributed_streamer, "_topology_cache", (2, 0, [[0, 1]])), \
                patch.object(distributed_streamer, "_distribution_lost_for_job", object()), \
                patch("torch.distributed.all_reduce", side_effect=all_reduce):
            with self.assertRaises(RuntimeError):
                _distributedStreamer.cached_or_created(key, explode)
            self.assertNotIn(("discovery",), cache)
            self.assertIn(discovery, cache[("retired",)])
            self.assertIsNone(distributed_streamer._topology_cache)

    def test_a_failed_vote_loses_distribution_for_the_rest_of_the_job(self):
        """A vote can fail on one rank after succeeding on a peer, and torch consumes its
        group-creation ordinal before constructing a group, so ranks that failed differently can
        never meet in a later new_group. The contract: the failure loses distribution for the
        remainder of the job - no rediscovery, no group creation, loads fall back to plain
        streaming - and a NEW job (a new default group) restores it."""
        key = ("local", ((0, 1),), "timeout")
        job = MagicMock(name="job")
        old_discovery = MagicMock(name="old_discovery")
        old_distribution = [MagicMock(name="old_distribution")]

        fake_dist = MagicMock(name="dist")
        fake_dist.is_initialized.return_value = True
        fake_dist.get_world_size.return_value = 2
        fake_dist.get_rank.return_value = 1
        fake_dist.group.WORLD = job
        fake_dist.all_reduce.side_effect = RuntimeError("vote timed out")

        params = _distributedStreamerParams()
        params.num_processes_on_node = 2      # a reused instance, layout from before the failure
        params.discovered_job = job
        params.my_global_rank = 1
        params.groups_by_ranks = [[0, 1]]

        cache = {("discovery",): old_discovery, key: old_distribution}
        with patch.object(distributed_streamer, "dist", fake_dist), \
                patch.object(distributed_streamer, "_group_cache", cache), \
                patch.object(distributed_streamer, "_topology_cache", (2, 1, [[0, 1]])), \
                patch.object(distributed_streamer, "_cache_job", job), \
                patch.object(distributed_streamer, "_distribution_lost_for_job", object()), \
                patch.dict(os.environ, {"RUNAI_STREAMER_PROCESS_GROUP_SIZE": "ignored"}):
            # The vote fails on this rank (its peer's may have succeeded).
            with self.assertRaises(RuntimeError):
                _distributedStreamer.cached_or_created(key, lambda: MagicMock())
            self.assertNotIn(("discovery",), cache, "the failed vote kept the discovery group")
            self.assertIsNone(distributed_streamer._topology_cache)
            self.assertTrue(distributed_streamer._distribution_is_lost())

            # The next load's discovery creates NO group - the ordinals can no longer be agreed -
            # and answers the conservative lone layout instead, a reused instance included.
            fake_dist.new_group.reset_mock()
            params.discover_local_ranks()
            fake_dist.new_group.assert_not_called()
            self.assertEqual((params.num_processes_on_node, params.groups_by_ranks), (1, [[1]]))

            # A new job - a new default group - is a clean slate.
            fake_dist.group.WORLD = MagicMock(name="new_job")
            self.assertFalse(distributed_streamer._distribution_is_lost())
        self.assertIn(old_discovery, cache[("retired",)])

    def test_a_failed_vote_retires_the_discovery_group_and_the_topology(self):
        # A vote that fails may have closed the discovery group's pairs; keeping it would fail
        # every later vote too. The next load must rediscover, on every rank together.
        key = ("global", 2, "test")
        discovery = MagicMock(name="discovery")
        cache = {("discovery",): discovery, key: MagicMock(name="old_group")}
        with patch.object(distributed_streamer, "_group_cache", cache), \
                patch.object(distributed_streamer, "_topology_cache", (2, 0, [[0, 1]])), \
                patch.object(distributed_streamer, "_distribution_lost_for_job", object()), \
                patch("torch.distributed.all_reduce", side_effect=RuntimeError("vote timed out")):
            with self.assertRaises(RuntimeError):
                _distributedStreamer.cached_or_created(key, lambda: MagicMock())
            self.assertNotIn(("discovery",), cache)
            self.assertIn(discovery, cache[("retired",)])
            self.assertIsNone(distributed_streamer._topology_cache)


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

    def test_the_joint_raise_passes_through_set_is_distributed_and_keeps_the_group(self):
        # The completed reduction reports the failed query on EVERY rank together, so it is not a
        # loss of distribution and must not be swallowed into a per-node fallback: a node that
        # destroyed its held group here would ask for a new one alone on the next read of the same
        # context - its peers reuse their held handles - and stall there until the timeout.
        job = DistributedJob([-1, 1 << 40])
        streamer = job.streamer()
        with job.patched():
            with self.assertRaises(RuntimeError):
                streamer.set_is_distributed(True, "cuda:1")
        job.events.release_group.assert_not_called()

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


class TestTheXcclBackend(unittest.TestCase):
    """Intel's backend, on a host with no XPU. Nothing here needs the hardware: the device is a
    string torch parses, and the backend is whichever one the test says it is."""

    @contextlib.contextmanager
    def decide(self, device):
        """set_is_distributed under xccl, with the memory reading mocked. RUNAI_STREAMER_DIST is
        cleared so the default `auto` runs - a backend allowed only under "1" proves nothing."""
        streamer = DistributedStreamer()
        with patch.dict(os.environ, {}, clear=True), \
                patch("runai_model_streamer.distributed_streamer.distributed_streamer."
                      "can_measure_device_memory", return_value=True), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2), \
                patch("torch.distributed.get_backend", return_value="xccl"), \
                patch("torch.distributed.get_backend_config", return_value="xpu:xccl"), \
                patch.object(DistributedStreamer, "get_min_device_free_memory",
                             return_value=1 << 40) as reduce_free:
            streamer.set_is_distributed(True, device)
            yield streamer, reduce_free

    def test_xccl_on_an_xpu_reaches_the_memory_check(self):
        """Both terms of the gate at once: xccl survives auto mode, and the check runs for it."""
        with self.decide("xpu:0") as (streamer, reduce_free):
            reduce_free.assert_called_once_with("xpu:0")
            self.assertTrue(streamer.is_distributed)

    def test_xccl_on_a_device_that_is_not_an_xpu_falls_back(self):
        """A backend and a device that do not match, as nccl on a CPU already is."""
        with self.decide("cuda:0") as (streamer, reduce_free):
            self.assertFalse(streamer.is_distributed)
            reduce_free.assert_not_called()


if __name__ == "__main__":
    unittest.main()
