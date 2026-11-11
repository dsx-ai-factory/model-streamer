# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import unittest
import hashlib
import os
import sys
import torch
import torch.distributed as dist
from typing import List
import tempfile
import shutil
import pickle
import time
import random
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.distribution_source import (
    RANGE_OFFSET_ALIGNMENT,
)

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    DistributedStreamer,
)
from runai_model_streamer.file_streamer import FileChunks
from runai_model_streamer.file_streamer.requests_iterator import (
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
)

class FakeStream:
    """A stand-in for a device stream, compared by VALUE.

    torch hands out a fresh Stream object per call and defines __eq__, so a fake that is only ever
    one object would let a production comparison written with `is not` pass here and fail on a GPU
    every round."""

    def __init__(self, ident=0):
        self.ident = ident

    def __eq__(self, other):
        return isinstance(other, FakeStream) and other.ident == self.ident

    def __hash__(self):
        return hash(self.ident)

    def synchronize(self):
        pass


class TestDistributedStreamer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

    def setUp(self):
        # No buffer-size override is needed any more. The buffer size is now derived as budget // N, so
        # it scales with the fixture instead of being a fixed 1 GiB minimum that collapsed onto the whole
        # stream and left every test with a single request and a ring that never cycled.
        if self.rank == 0:
            self.temp_dir = tempfile.mkdtemp()
        else:
            self.temp_dir = None
        dir_list = [self.temp_dir]
        dist.broadcast_object_list(dir_list, src=0)
        self.temp_dir = dir_list[0]

    def tearDown(self):
        dist.barrier()
        if self.rank == 0 and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    @contextlib.contextmanager
    def collected_logs(self, name):
        """Every message a logger emits, including none.

        assertLogs FAILS when nothing was logged, and "nothing" is what most ranks must do for a
        line that is said once by the first of them."""
        records = []

        class Collect(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        log = logging.getLogger(name)
        handler = Collect()
        log.addHandler(handler)
        try:
            yield records
        finally:
            log.removeHandler(handler)

    def _prepare_file_requests(self, file_specs: List[dict]) -> List[FileChunks]:
        # rank 0 creates the file requests and broadcasts them to all ranks
        if self.rank == 0:
            file_requests = []
            for i, spec in enumerate(file_specs):
                file_path = os.path.join(self.temp_dir, f"file_{i}.bin")
                content = torch.randint(0, 256, (spec["size"],), dtype=torch.uint8).numpy().tobytes()
                with open(file_path, "wb") as f:
                    f.write(content)
                file_requests.append(FileChunks.contiguous(id=i, path=file_path, sizes=spec["chunks"], offset=0))
            data_bytes = pickle.dumps(file_requests)
            size_tensor = torch.tensor([len(data_bytes)], dtype=torch.long)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long)
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()
        if self.rank == 0:
            buffer = torch.tensor(list(data_bytes), dtype=torch.uint8)
        else:
            buffer = torch.empty(data_size, dtype=torch.uint8)
        dist.broadcast(buffer, src=0)
        if self.rank != 0:
            file_requests = pickle.loads(buffer.numpy().tobytes())
        return file_requests

    def test_1_success_data_correctness(self):
        file_specs = [{"size": 2580, "chunks": [500, 260, 260, 260, 260, 260, 260, 260, 260]}]
        requests = self._prepare_file_requests(file_specs)
        original_data_map = {}
        for req in requests:
            with open(req.path, "rb") as f:
                original_data_map[req.id] = f.read()
        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}

        # BOTH modes. Owning changes what a tensor IS - a clone of its own rather than a view into
        # the staging buffer - so the bytes have to be checked under each, not only under the default.
        for owned in (True, False):
            with self.subTest(owned=owned):
                reconstructed_data_map = {req.id: [None] * len(req.sizes) for req in requests}
                with patch.dict(os.environ, env_vars):
                    with DistributedStreamer() as streamer:
                        streamer.stream_files(requests, None, "cpu", True, owned=owned)
                        for req_id, chunk_idx, data_tensor in streamer.get_chunks():
                            # read here, inside the loop: under owned=False this is the only point
                            # at which the view is guaranteed to hold its own bytes
                            reconstructed_data_map[req_id][chunk_idx] = data_tensor.cpu().numpy().tobytes()
                for req_id, chunks in reconstructed_data_map.items():
                    self.assertEqual(original_data_map[req_id], b"".join(chunks))
        if self.rank == 0:
            print(f"\n✅ Success test verified on all {self.world_size} ranks.")

    def test_1_success_deep_ring_buffer(self):
        # Pins the ring exactly, rather than letting it fall out of whatever the fixture happens to be:
        # a rank with several submissions in flight, recycling buffers between them.
        #
        # RUNAI_STREAMER_MEMORY_LIMIT is a NODE total, so it is divided by the ranks on this host: the
        # limit below is sized for 4 buffers PER RANK (4 x 260 x world_size). The buffer size is not set
        # at all - budget // 4 lands on the chunk size on its own, which is the point of the new rule.
        chunk = 260
        num_chunks = 20
        buffers_per_rank = 4
        file_specs = [{"size": chunk * num_chunks, "chunks": [chunk] * num_chunks}]
        requests = self._prepare_file_requests(file_specs)

        original_data_map = {}
        for req in requests:
            with open(req.path, "rb") as f:
                original_data_map[req.id] = f.read()
        reconstructed_data_map = {req.id: [None] * len(req.sizes) for req in requests}

        env_vars = {
            "RUNAI_STREAMER_DIST": "1",
            "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0",
            RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME: str(buffers_per_rank),
            RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: str(chunk * buffers_per_rank * self.world_size),
        }
        with patch.dict(os.environ, env_vars):
            with DistributedStreamer() as streamer:
                streamer.stream_files(requests, None, "cpu", True)
                for req_id, chunk_idx, data_tensor in streamer.get_chunks():
                    reconstructed_data_map[req_id][chunk_idx] = data_tensor.cpu().numpy().tobytes()

                # the per-rank share really did buy a deep ring, not the floor of 2
                requests_iterator = streamer.distributed_streamer.file_streamer.requests_iterator
                self.assertEqual(requests_iterator.buffer_size, chunk)
                self.assertEqual(requests_iterator.num_buffers, buffers_per_rank)

            for req_id, chunks in reconstructed_data_map.items():
                self.assertEqual(original_data_map[req_id], b"".join(chunks))
            if self.rank == 0:
                print(f"\n✅ Deep ring buffer test verified on all {self.world_size} ranks.")

    def test_1_success_empty_file_list(self):
        requests = []
        env_vars = {"RUNAI_STREAMER_DIST": "1"}
        with patch.dict(os.environ, env_vars):
            with DistributedStreamer() as streamer:
                streamer.stream_files(requests, None, "cpu", True)
                count = 0
                for req_id, chunk_idx, data_tensor in streamer.get_chunks():
                    count += 1
            self.assertEqual(count, 0)
        if self.rank == 0:
            print(f"\n✅ Success empty file list test verified on all {self.world_size} ranks.")

    def test_1_a_rank_short_of_device_memory_keeps_the_slot_size(self):
        """The ring the group agreed will not fit on this rank any more.

        What it may change alone is DEPTH. Changing the slot size would send its peers something
        they cannot receive, and a host ring cannot be broadcast from at all - so the retry is one
        slot of the SAME size, and the stream still has to come out whole."""
        from runai_model_streamer.file_streamer import file_streamer as fs_module
        from runai_model_streamer.file_streamer.device_support import DeviceSupport
        from runai_model_streamer.file_streamer.requests_iterator import (
            RunaiStreamerMemoryLimitException,
        )
        from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA

        class FakeEvent:
            def record(self, stream=None):
                pass

            def query(self):
                return True

            def synchronize(self):
                pass

        support = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeEvent,
                                current_stream=lambda device=None: FakeStream())

        file_specs = [{"size": 6144, "chunks": [2048] * 3}, {"size": 4096, "chunks": [2048] * 2}]
        requests = self._prepare_file_requests(file_specs)
        originals = {}
        for req in requests:
            with open(req.path, "rb") as handle:
                originals[req.id] = handle.read()

        module = "runai_model_streamer.distributed_streamer.distributed_streamer"
        agreed = (2048, 3)
        asked = []
        real_stream_files = fs_module.FileStreamer.stream_files

        handling = []

        def one_device_ring_too_many(self, *args, **kwargs):
            """Refuse the agreed depth once, the way the ring layer refuses a shape it cannot
            allocate, then behave normally."""
            shape = kwargs.get("slot_shape")
            if shape is not None:
                asked.append(shape)
                # Whether an exception is still being handled while this attempt runs. The pool
                # takes every slot up front, and a traceback holds the ones it got, so a retry
                # inside the handler asks for memory on top of the attempt that just failed.
                handling.append(sys.exc_info()[0])
                if len(asked) == 1:
                    # FROM an allocator error, as file_streamer raises it. Only that cause is
                    # retried: the same exception type also carries refusals a shallower ring
                    # cannot answer.
                    try:
                        raise torch.cuda.OutOfMemoryError("out of memory")
                    except torch.cuda.OutOfMemoryError as oom:
                        raise RunaiStreamerMemoryLimitException(
                            "cannot build the ring this stream asks for") from oom
            return real_stream_files(self, *args, **kwargs)

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars), \
             patch.object(fs_module, "device_support", return_value=support), \
             patch.object(fs_module, "device_free_memory", return_value=1 << 30), \
             patch(f"{module}.device_support", return_value=support), \
             patch(f"{module}.agreed_slot_shape", return_value=agreed), \
             patch.object(fs_module.FileStreamer, "stream_files", one_device_ring_too_many):
            with DistributedStreamer() as streamer:
                inner = streamer.distributed_streamer
                streamer.stream_files(requests, None, "cpu", True, owned=True)

                got = {}
                for req_id, chunk_idx, tensor in streamer.get_chunks():
                    got.setdefault(req_id, {})[chunk_idx] = tensor.cpu().numpy().tobytes()

        if not asked:
            self.skipTest("this rank read nothing, so it never built a ring")

        self.assertEqual(asked[0], agreed, "it asks for the agreed shape first")
        self.assertEqual(asked[1], (agreed[0], 1), "then one slot of the SAME size")
        self.assertEqual(inner.slot_shape, (agreed[0], 1), "and says so, for the receive buffer")
        self.assertIsNone(handling[1],
                          "the retry runs after the failed attempt let go of its memory")

        for req in requests:
            rebuilt = b"".join(got[req.id][i] for i in sorted(got[req.id]))
            self.assertEqual(rebuilt, originals[req.id],
                             f"request {req.id} did not reassemble after the retry")

        if self.rank == 0:
            print(f"\n✅ Shallow retry verified on all {self.world_size} ranks.")

    def test_1_a_refusal_that_is_not_an_allocation_failure_is_not_retried(self):
        """The same exception type covers a shape this layer refused outright, and a packing
        failure raised after the submissions are live. A shallower ring answers neither, and
        re-entering stream_files there would replace the real diagnosis with its own guard."""
        from runai_model_streamer.file_streamer import file_streamer as fs_module
        from runai_model_streamer.file_streamer.device_support import DeviceSupport
        from runai_model_streamer.file_streamer.requests_iterator import (
            RunaiStreamerMemoryLimitException,
        )
        from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA

        support = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=object,
                                current_stream=lambda device=None: None)

        requests = self._prepare_file_requests([{"size": 4096, "chunks": [2048, 2048]}])
        module = "runai_model_streamer.distributed_streamer.distributed_streamer"
        asked = []

        def refuse_without_an_allocator_error(self, *args, **kwargs):
            shape = kwargs.get("slot_shape")
            if shape is None:
                raise AssertionError("the direct branch must dictate a shape")
            asked.append(shape)
            # No __cause__: this is the shape pre-check, not the allocator.
            raise RunaiStreamerMemoryLimitException("a slot cannot hold the largest range")

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars), \
             patch.object(fs_module, "device_support", return_value=support), \
             patch(f"{module}.device_support", return_value=support), \
             patch(f"{module}.agreed_slot_shape", return_value=(2048, 3)), \
             patch.object(fs_module.FileStreamer, "stream_files",
                          refuse_without_an_allocator_error):
            with DistributedStreamer() as streamer:
                with self.assertRaises(RunaiStreamerMemoryLimitException):
                    streamer.stream_files(requests, None, "cpu", True, owned=True)

        if not asked:
            self.skipTest("this rank read nothing, so it never built a ring")

        self.assertEqual(asked, [(2048, 3)], "asked once, at the agreed depth, and not again")

    def test_1_a_ring_already_one_slot_deep_is_not_retried(self):
        """There is nowhere shallower to go.

        Retrying the same shape would ask the allocator for what it has just refused, and the second
        refusal is the one the caller would see. Every rank reaches this together, so no peer is
        left waiting."""
        from runai_model_streamer.file_streamer import file_streamer as fs_module
        from runai_model_streamer.file_streamer.device_support import DeviceSupport
        from runai_model_streamer.file_streamer.requests_iterator import (
            RunaiStreamerMemoryLimitException,
        )
        from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA

        support = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=object,
                                current_stream=lambda device=None: None)

        requests = self._prepare_file_requests([{"size": 4096, "chunks": [2048, 2048]}])
        module = "runai_model_streamer.distributed_streamer.distributed_streamer"
        asked = []

        def always_refuse(self, *args, **kwargs):
            shape = kwargs.get("slot_shape")
            if shape is None:
                raise AssertionError("the direct branch must dictate a shape")
            asked.append(shape)
            raise RunaiStreamerMemoryLimitException("cannot build the ring this stream asks for")

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars), \
             patch.object(fs_module, "device_support", return_value=support), \
             patch(f"{module}.device_support", return_value=support), \
             patch(f"{module}.agreed_slot_shape", return_value=(2048, 1)), \
             patch.object(fs_module.FileStreamer, "stream_files", always_refuse):
            with DistributedStreamer() as streamer:
                with self.assertRaises(RunaiStreamerMemoryLimitException):
                    streamer.stream_files(requests, None, "cpu", True, owned=True)

        if not asked:
            self.skipTest("this rank read nothing, so it never built a ring")

        self.assertEqual(asked, [(2048, 1)], "asked once, and not asked again")

        if self.rank == 0:
            print(f"\n✅ No pointless retry verified on all {self.world_size} ranks.")

    def test_1_losing_the_device_ring_is_logged_once(self):
        """A device the library could fill, with no shape the group can send from.

        That combination stages through host memory and used to leave no trace. It is said by ONE
        rank, because every rank reaches it from the same numbers."""
        from runai_model_streamer.file_streamer.device_support import DeviceSupport
        from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA

        class FakeEvent:
            def record(self, stream=None):
                pass

            def query(self):
                return True

            def synchronize(self):
                pass

        support = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeEvent,
                                current_stream=lambda device=None: None)

        requests = self._prepare_file_requests([{"size": 2048, "chunks": [1024, 1024]}])
        module = "runai_model_streamer.distributed_streamer.distributed_streamer"

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars), \
             patch(f"{module}.device_support", return_value=support), \
             patch(f"{module}.agreed_slot_shape", return_value=None), \
             patch(f"{module}.device_ring_budget", return_value=1 << 20):
            with self.collected_logs(module) as logs:
                with DistributedStreamer() as streamer:
                    inner = streamer.distributed_streamer
                    streamer.stream_files(requests, None, "cpu", True)
                    list(streamer.get_chunks())

        self.assertFalse(inner.sends_directly, "the fixture must take the staged path")

        said = [line for line in logs if "no ring fits" in line]
        if inner.original_group_rank == 0:
            self.assertEqual(len(said), 1, "said once, by the first rank")
            self.assertIn("staging through host memory instead", said[0])
        else:
            self.assertEqual(said, [], "and by no other rank")

    def test_1_the_direct_source_is_used_where_the_library_can_fill_the_ring(self):
        for owned in (True, False):
            with self.subTest(owned=owned):
                self._direct_source_round_trip(owned)

    def _direct_source_round_trip(self, owned):
        """The whole 5d switch, end to end, on a host-only runner.

        Two device facts a host-only runner cannot produce are stood in: the vendor row, and the
        slot shape the group agrees from its reduced free memory - that reduction runs only under a
        device backend, so under gloo there is no shape and the switch would always say staged.
        With both in place the library builds a ring it treats as device memory, stream_files is
        asked for a sent-from ring, and the loop broadcasts its slots instead of staging through a
        host ring. The bytes must still come back whole."""
        from runai_model_streamer.file_streamer import file_streamer as fs_module
        from runai_model_streamer.file_streamer.device_support import DeviceSupport
        from runai_model_streamer.libstreamer import RUNAI_FILE_STREAMER_DEVICE_CUDA

        class FakeEvent:
            def record(self, stream=None):
                pass

            def query(self):
                return True

            def synchronize(self):
                pass

        support = DeviceSupport(api_type=RUNAI_FILE_STREAMER_DEVICE_CUDA, new_event=FakeEvent,
                                current_stream=lambda device=None: FakeStream())

        # MORE THAN THE RING HOLDS: 10 ranges of 2048 into 2 slots that take one each, so the
        # slowest rank still gets more ranges than slots and has to recycle. The release and its
        # event run between rounds - with a ring that spans the stream they never would.
        file_specs = [{"size": 12288, "chunks": [2048] * 6},
                      {"size": 8192, "chunks": [2048] * 4}]
        requests = self._prepare_file_requests(file_specs)
        originals = {}
        for req in requests:
            with open(req.path, "rb") as handle:
                originals[req.id] = handle.read()

        module = "runai_model_streamer.distributed_streamer.distributed_streamer"
        sources = []
        real_direct = __import__(module, fromlist=["DirectDistributionSource"]).DirectDistributionSource

        inner_cls = __import__(module, fromlist=["_distributedStreamer"])._distributedStreamer
        real_slot_is_free = inner_cls.slot_is_free
        started = []

        def spy_slot_is_free(self, started_on=None):
            started.append(started_on)
            return real_slot_is_free(self, started_on)

        shape_calls = []

        def record_shape(*a, **k):
            shape_calls.append(a)
            return (2048, 2)

        def record_source(*a, **k):
            source = real_direct(*a, **k)
            sources.append(source)
            return source

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars), \
             patch.object(fs_module, "device_support", return_value=support), \
             patch.object(fs_module, "device_free_memory", return_value=1 << 30), \
             patch(f"{module}.device_support", return_value=support), \
             patch(f"{module}.agreed_slot_shape", side_effect=record_shape), \
             patch.object(inner_cls, "slot_is_free", spy_slot_is_free), \
             patch(f"{module}.DirectDistributionSource", side_effect=record_source), \
             self.collected_logs(module) as logs:
            with DistributedStreamer() as streamer:
                inner = streamer.distributed_streamer
                streamer.stream_files(requests, None, "cpu", True, owned=owned, ring_buffers=5)

                self.assertTrue(inner.sends_directly,
                                "the switch must choose the direct source here")
                # The shape itself is patched in, so asserting it is not None would say nothing.
                # What the production code decides is the CAP it dictates with that shape: the ring
                # packs by bytes, and a slot may hold no more ranges than the metadata has rows.
                from runai_model_streamer.distributed_streamer.distributed_streamer import (
                    MAX_CHUNKS_PER_BATCH,
                )
                self.assertEqual(
                    inner.file_streamer.requests_iterator.max_ranges_per_slot,
                    MAX_CHUNKS_PER_BATCH,
                    "a sent ring must be capped at the rows a peer will read",
                )

                # WHAT THE PRODUCTION CODE PASSES, which patching a return value leaves unexamined.
                # The whole list and not this rank's share - a share is floored by the largest range
                # IN IT, so two ranks would size differently - and the caller's depth, which used to
                # be dropped on this path while the environment variable was still honoured.
                self.assertEqual(len(shape_calls), 1)
                _, whole, depth = shape_calls[0]
                self.assertIs(whole, requests, "the WHOLE file list, never this rank's share")
                self.assertEqual(depth, 5, "the caller's ring_buffers reaches the agreed shape")

                # The ring really is reused, which is what drives the release and its event. A ring
                # that spans the stream never recycles, and then the hardest part of the direct
                # source - handing a slot back and having it refilled - is never exercised. Read
                # HERE: __exit__ frees the ring and leaves requests_iterator None.
                ring = inner.file_streamer.requests_iterator
                mine = sum(len(fc.sizes) for fc in inner.rank_file_chunks_list)
                self.assertGreater(mine, ring.num_buffers,
                                   "this rank needs more ranges than slots, or nothing recycles")

                got = {}
                for req_id, chunk_idx, tensor in streamer.get_chunks():
                    got.setdefault(req_id, {})[chunk_idx] = tensor.cpu().numpy().tobytes()

        # The bytes, not just the count: a wrong slot or a wrong offset still yields the right
        # number of chunks.
        for req in requests:
            rebuilt = b"".join(got[req.id][i] for i in sorted(got[req.id]))
            self.assertEqual(rebuilt, originals[req.id],
                             f"request {req.id} did not reassemble")

        # UNCONDITIONAL. Behind an `if reading_from_storage` this assertion quietly disappears the
        # day the partition gives a rank nothing, and the test still passes.
        self.assertEqual([line for line in logs if "no ring fits" in line], [],
                         "nothing fell back, so nothing should say it did")

        # The guard is only armed if the loop hands it the stream it started on. Passing None would
        # leave slot_is_free unable to see a caller move, and every test of it would still pass.
        if inner.sends_directly and inner.reading_from_storage:
            self.assertTrue(started, "a sending rank releases at least one slot")
            self.assertTrue(all(s is not None for s in started),
                            "the loop must tell slot_is_free which stream it started on")
        self.assertEqual(len(sources), 1 if inner.reading_from_storage else 0,
                         "a rank that reads sends from the ring itself, and one that does not sends"
                         " nothing")

        if self.rank == 0:
            print(f"\n✅ Direct source (owned={owned}) verified on all {self.world_size} ranks.")

    def test_1_the_slot_is_released_with_an_event_where_the_device_has_them(self):
        """The loop marks the end of a round, and a source that refills waits for that mark.

        Gloo on the host has no events, so the path is only reachable here by standing a vendor row
        in. What is being checked is the LOOP: that it records after the broadcast and the clones,
        and hands the mark over with the slot."""
        releases = []

        built = []

        class Watched:
            """The staged source, with every release recorded."""

            def __init__(self, inner):
                built.append(self)
                self._inner = inner

            def next_slot(self):
                return self._inner.next_slot()

            def release(self, filled, done=None):
                releases.append(done)
                self._inner.release(filled, done)

        clones = []

        class FakeEvent:
            def __init__(self):
                self.stream = None
                self.clones_at_record = None

            def record(self, stream=None):
                self.stream = stream
                # WHEN it was recorded, which is the whole point. An event built before the yields
                # would be handed over looking exactly the same.
                self.clones_at_record = len(clones)

        class FakeSupport:
            new_event = FakeEvent

            @staticmethod
            def current_stream(device):
                return f"stream of {device}"

        file_specs = [{"size": 2048, "chunks": [1024, 1024]}]
        requests = self._prepare_file_requests(file_specs)

        module = "runai_model_streamer.distributed_streamer.distributed_streamer"
        real_source = __import__(module, fromlist=["StagedDistributionSource"]).StagedDistributionSource

        inner_cls = __import__(module, fromlist=["_distributedStreamer"])._distributedStreamer
        real_own = inner_cls.own

        def counting_own(buffer):
            clones.append(buffer)
            return real_own(buffer)

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars), \
             patch(f"{module}.device_support", return_value=FakeSupport), \
             patch.object(inner_cls, "own", staticmethod(counting_own)), \
             patch(f"{module}.StagedDistributionSource",
                   side_effect=lambda *a, **k: Watched(real_source(*a, **k))):
            with DistributedStreamer() as streamer:
                inner = streamer.distributed_streamer
                streamer.stream_files(requests, None, "cpu", True, owned=True)
                chunks = list(streamer.get_chunks())

        self.assertEqual(len(chunks), 2, "both chunks reach every rank")

        # Every rank on the distributed path builds a source, even one with an empty share. Without
        # this, a run that never distributed at all would skip below and read as a pass.
        self.assertEqual(len(built), 1, "the distributed path ran on this rank")
        # It is about the STAGED release. device_support is patched above, so the only reason this
        # rank still stages is that gloo never agreed a slot shape - which is worth saying, or the
        # next person reads a confusing failure in a test about events.
        self.assertFalse(inner.sends_directly, "this test is about the staged release path")
        self.assertEqual(len(clones), 2, "both chunks were cloned out of a slot")

        if not releases:
            self.skipTest("this rank read nothing, so it never released a slot")

        for done in releases:
            self.assertIsInstance(done, FakeEvent,
                                  "the loop must hand an event over with the slot")
            self.assertEqual(done.stream, "stream of cpu",
                             "recorded on the stream of THIS streamer's device")
            self.assertEqual(done.clones_at_record, 2,
                             "recorded AFTER the round's clones, not before the loop")

    def test_1_success_one_rank_has_nothing_to_read(self):
        """A rank whose share is empty still receives from its peers.

        Different from an empty FILE LIST, where no rank reads and the broadcast loop never runs.
        One file of one chunk goes to one rank, and total_chunks_to_read is 1 on every rank, so the
        loop does run on the rank that has nothing - as a receiver only."""
        if self.world_size < 2:
            self.skipTest("needs at least two ranks")

        file_specs = [{"size": 100, "chunks": [100]}]
        requests = self._prepare_file_requests(file_specs)
        with open(requests[0].path, "rb") as f:
            original = f.read()

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        received = []
        with patch.dict(os.environ, env_vars):
            with DistributedStreamer() as streamer:
                inner = streamer.distributed_streamer
                streamer.stream_files(requests, None, "cpu", True)

                group = inner.distribution_group
                share_is_empty = len(inner.rank_file_chunks_list) == 0
                self.assertEqual(inner.reading_from_storage, not share_is_empty)

                for req_id, chunk_idx, data_tensor in streamer.get_chunks():
                    received.append((req_id, chunk_idx, data_tensor.cpu().numpy().tobytes()))

        self.assertEqual(len(received), 1, "every rank sees the one chunk, whoever read it")
        self.assertEqual(received[0][0], 0)
        self.assertEqual(received[0][1], 0)
        self.assertEqual(received[0][2], original)

        # The point of the fixture: this really was the empty-share case on all but one rank, not a
        # share that happened to be small.
        #
        # Over the DISTRIBUTION group, which is per node unless RUNAI_STREAMER_DIST_GLOBAL is set.
        # Each group partitions the file on its own, so the world would count one reader per group.
        group_size = dist.get_world_size(group=group)
        empties = [torch.zeros(1, dtype=torch.long) for _ in range(group_size)]
        dist.all_gather(empties, torch.tensor([1 if share_is_empty else 0], dtype=torch.long),
                        group=group)
        self.assertEqual(sum(e.item() for e in empties), group_size - 1,
                         "one rank in this group read the chunk and every other rank had nothing")

        if self.rank == 0:
            print(f"\n✅ Empty-share rank test verified on all {self.world_size} ranks.")

    def test_1_success_random_files(self):
        file_specs = []
        # Use a random number of files (1 to 10)
        num_files = random.randint(1, 10)

        for _ in range(num_files):
            # Use a random file size between 0 and 10 MB
            total_size = random.randint(0, 10 * 1024 * 1024)
            
            # Handle the edge case of a 0-byte file
            if total_size == 0:
                file_specs.append({"size": 0, "chunks": []})
                continue

            # Generate random chunk sizes that sum up to the total file size
            chunks = []
            remaining_size = total_size
            while remaining_size > 0:
                # Set a max chunk size for realism (e.g., 1MB), but don't exceed what's left
                max_chunk_size = min(remaining_size, 1 * 1024 * 1024)
                chunk_size = random.randint(1, max_chunk_size)
                chunks.append(chunk_size)
                remaining_size -= chunk_size
            
            file_specs.append({"size": total_size, "chunks": chunks})
        # --- END OF MODIFIED SECTION ---

        requests = self._prepare_file_requests(file_specs)
        original_data_map = {}
        for req in requests:
            with open(req.path, "rb") as f:
                original_data_map[req.id] = f.read()

        reconstructed_data_map = {req.id: [None] * len(req.sizes) for req in requests}
        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars):
            with DistributedStreamer() as streamer:
                streamer.stream_files(requests, None, "cpu", True)
                for req_id, chunk_idx, data_tensor in streamer.get_chunks():
                    reconstructed_data_map[req_id][chunk_idx] = data_tensor.cpu().numpy().tobytes()

            # Verify that the reconstructed data for each file matches the original
            for req in requests:
                reconstructed_bytes = b"".join(reconstructed_data_map[req.id])
                self.assertEqual(original_data_map[req.id], reconstructed_bytes)

        if self.rank == 0:
            print(f"\n✅ Success random files test verified on all {self.world_size} ranks.")

    def test_1_success_alignment(self):
        # Chunk sizes are random in [1, alignment*10].
        # BUFFER_MIN_BYTESIZE = alignment*20 guarantees at least two max-size chunks fit in
        # one batch: worst case is chunk1=alignment*10 at offset 0 and chunk2=alignment*10
        # at offset=alignment*10, totalling alignment*20 = BUFFER_BYTESIZE exactly.
        # With 1-20 files of 4 chunks each there are enough chunks to guarantee at least
        # one batch where two tensors are packed together.
            
        if self.world_size < 2:
            self.skipTest("Alignment packing test requires at least 2 processes.")

        alignment = RANGE_OFFSET_ALIGNMENT
        num_files = random.randint(1, 20)
        def random_chunks(n):
            return [random.randint(1, alignment * 10) for _ in range(n)]

        chunks_per_file = [random_chunks(4) for _ in range(num_files)]
        file_specs = [
            {"size": sum(c), "chunks": c} for c in chunks_per_file
        ]
        requests = self._prepare_file_requests(file_specs)

        env_vars = {
            "RUNAI_STREAMER_DIST": "1",
            "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": str(alignment * 20),
        }
        # owned=False ON PURPOSE. This alignment describes where tensors sit INSIDE the staging
        # buffers, so that the offsets one rank broadcasts are usable on every other. An owned tensor
        # is a clone with a storage of its own, where that layout does not apply - and forcing the
        # alignment onto it, by slicing an over-allocated buffer, made vLLM fault on a real GPU.
        with patch.dict(os.environ, env_vars):
            with DistributedStreamer() as streamer:
                streamer.stream_files(requests, None, "cpu", True, owned=False)

                for _req_id, _chunk_idx, data_tensor in streamer.get_chunks():
                    ptr = data_tensor.data_ptr()
                    self.assertEqual(
                        ptr % alignment, 0,
                        f"Rank {self.rank}: tensor data_ptr 0x{ptr:x} is not aligned to {alignment} bytes"
                    )

        if self.rank == 0:
            print(f"\n✅ Alignment test verified on all {self.world_size} ranks.")

    def test_1_owned_tensors_survive_later_rounds(self):
        # THE GUARANTEE ITSELF. BUFFER_MIN_BYTESIZE=0 sizes the staging buffer to a single chunk, so
        # every round overwrites the whole of it - a borrowed view would come back holding the bytes
        # of the last round rather than its own.
        if self.world_size < 2:
            self.skipTest("Ownership test requires at least 2 processes.")

        chunk = 260
        num_chunks = 12
        file_specs = [{"size": chunk * num_chunks, "chunks": [chunk] * num_chunks}]
        requests = self._prepare_file_requests(file_specs)

        original = {}
        for req in requests:
            with open(req.path, "rb") as f:
                original[req.id] = f.read()

        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with patch.dict(os.environ, env_vars):
            with DistributedStreamer() as streamer:
                streamer.stream_files(requests, None, "cpu", True)
                # HELD, not read: converting to bytes inside the loop is what a borrowed view survives
                held = {(req_id, idx): tensor
                        for req_id, idx, tensor in streamer.get_chunks()}

        self.assertEqual(len(held), num_chunks)
        for (req_id, idx), tensor in held.items():
            self.assertEqual(
                tensor.cpu().numpy().tobytes(),
                original[req_id][idx * chunk: (idx + 1) * chunk],
                f"Rank {self.rank}: chunk {idx} of file {req_id} changed after the stream ended",
            )
        if self.rank == 0:
            print(f"\n✅ Owned tensors verified on all {self.world_size} ranks.")

    def test_1_owned_tensors_have_independent_storage(self):
        # Per TENSOR, not per buffer. Borrowed, every tensor is a view into one of the two staging
        # buffers, so a whole stream shows at most two storages. Owned, each has its own - otherwise a
        # caller holding one small tensor would pin the buffer it came from.
        if self.world_size < 2:
            self.skipTest("Ownership test requires at least 2 processes.")

        chunk = 260
        num_chunks = 8
        file_specs = [{"size": chunk * num_chunks, "chunks": [chunk] * num_chunks}]
        requests = self._prepare_file_requests(file_specs)
        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}

        def collect(owned):
            with patch.dict(os.environ, env_vars):
                with DistributedStreamer() as streamer:
                    streamer.stream_files(requests, None, "cpu", True, owned=owned)
                    # the TENSORS are kept, not their addresses: a freed storage lets the allocator
                    # hand the same address back, which would make distinctness look like reuse
                    return [tensor for _, _, tensor in streamer.get_chunks()]

        borrowed = collect(False)
        self.assertLessEqual(
            len({tensor.untyped_storage().data_ptr() for tensor in borrowed}), 2,
            "borrowed tensors must alias the staging buffers",
        )

        owned = collect(True)
        addresses = [tensor.untyped_storage().data_ptr() for tensor in owned]
        self.assertEqual(len(set(addresses)), len(addresses),
                         "an owned tensor must not share storage with another")

    def test_1_auto_mode_no_distributed_with_gloo(self):
        """RUNAI_STREAMER_DIST=auto should disable distributed streaming for gloo backend."""
        if self.world_size < 2:
            self.skipTest("This test requires at least 2 processes.")

        if dist.get_backend() != "gloo":
            self.skipTest("This test requires gloo backend to be initialized")

        file_specs = [{"size": 100, "chunks": [100]}]
        requests = self._prepare_file_requests(file_specs)
        original_data_map = {}
        for req in requests:
            with open(req.path, "rb") as f:
                original_data_map[req.id] = f.read()

        with patch.dict(os.environ, {"RUNAI_STREAMER_DIST": "auto"}):
            with DistributedStreamer() as streamer:
                streamer.stream_files(requests, None, "cpu", True)
                self.assertFalse(
                    streamer.is_distributed,
                    "Distributed streaming should be disabled for gloo backend when RUNAI_STREAMER_DIST=auto"
                )
                reconstructed_data_map = {req.id: [None] * len(req.sizes) for req in requests}
                for req_id, chunk_idx, data_tensor in streamer.get_chunks():
                    reconstructed_data_map[req_id][chunk_idx] = data_tensor.cpu().numpy().tobytes()

        for req in requests:
            reconstructed_bytes = b"".join(reconstructed_data_map[req.id])
            self.assertEqual(original_data_map[req.id], reconstructed_bytes)

        if self.rank == 0:
            print(f"\n✅ Auto mode correctly disabled distributed streaming for gloo backend.")


    def _prepare_position_encoded_files(self, files: int, chunks_per_file: int, chunk_size: int):
        """Files whose every chunk holds bytes derived from its own (file, index).

        Identical filler would hide the failure this test exists for: a rank could yield another
        rank's tensor under the wrong identity and the bytes would still compare equal.
        """
        # The property this fixture is FOR, asserted rather than assumed. The previous encoding
        # claimed it in a comment and did not have it, and nothing noticed - a swap between two
        # colliding chunks was indistinguishable from a correct read.
        payloads = {}
        for file_index in range(files):
            for chunk_index in range(chunks_per_file):
                payload = self._expected_bytes(file_index, chunk_index, chunk_size)
                clash = payloads.get(payload)
                assert clash is None, (
                    f"chunks {clash} and {(file_index, chunk_index)} have identical bytes, so a swap "
                    f"between them would pass every assertion in this test")
                payloads[payload] = (file_index, chunk_index)

        if self.rank == 0:
            requests = []
            for file_index in range(files):
                path = os.path.join(self.temp_dir, f"encoded_{file_index}.bin")
                with open(path, "wb") as handle:
                    for chunk_index in range(chunks_per_file):
                        handle.write(self._expected_bytes(file_index, chunk_index, chunk_size))
                requests.append(FileChunks.contiguous(
                    id=file_index, path=path, sizes=[chunk_size] * chunks_per_file, offset=0))
            payload = pickle.dumps(requests)
            size_tensor = torch.tensor([len(payload)], dtype=torch.long)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long)

        dist.broadcast(size_tensor, src=0)
        if self.rank == 0:
            buffer = torch.tensor(list(payload), dtype=torch.uint8)
        else:
            buffer = torch.empty(size_tensor.item(), dtype=torch.uint8)
        dist.broadcast(buffer, src=0)
        if self.rank != 0:
            requests = pickle.loads(buffer.numpy().tobytes())
        return requests

    @staticmethod
    def _expected_bytes(file_index: int, chunk_index: int, size: int) -> bytes:
        # Sensitive to both coordinates and to the position within the chunk, so a swap between any
        # two chunks changes the bytes.
        #
        # HASHED, not a linear combination. This was `(file_index * 37 + chunk_index * 91 + offset)
        # % 256`, and 37*2 + 91*2 is exactly 256 - so (f, c) and (f+2, c+2) produced identical bytes.
        # At files=3, chunks_per_file=8 that was six colliding pairs, (0,0)/(2,2) through (0,5)/(2,7):
        # 18 distinct payloads for 24 chunks. A swap between any of those pairs passed the byte
        # comparison, which is the single failure this test exists to detect.
        #
        # Any linear scheme mod 256 has this problem somewhere; a digest has no such structure to
        # exploit. `+ offset` keeps the within-chunk position sensitivity the linear version had.
        digest = hashlib.blake2b(f"{file_index}:{chunk_index}".encode(), digest_size=32).digest()
        return bytes((digest[offset % 32] + offset) % 256 for offset in range(size))

    def test_1_broadcast_metadata_identifies_tensors_a_rank_never_read(self):
        """A rank must yield the right tensor for chunks it did not read itself.

        This is what makes it safe for a rank to build only its own share of the partition. The
        identity of a received tensor comes from the broadcast METADATA, not from any partition, so a
        rank knows nothing about which tensors the others own - and does not need to.

        The test fails if that ever stops being true. Moving a lookup from the metadata tensor to
        rank_dicts_map would load a model with silently mismatched weights, and nothing else here
        would notice.
        """
        chunk_size = 512
        requests = self._prepare_position_encoded_files(files=3, chunks_per_file=8,
                                                        chunk_size=chunk_size)

        for policy in ("chunks", "files", "spans"):
            env_vars = {
                "RUNAI_STREAMER_DIST": "1",
                "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0",
                "RUNAI_STREAMER_PARTITION_POLICY": policy,
            }
            with patch.dict(os.environ, env_vars):
                with DistributedStreamer() as streamer:
                    streamer.stream_files(requests, None, "cpu", True)

                    # What THIS rank read from storage, before consuming anything.
                    mine = {
                        (origin[0], origin[1])
                        for source_map in streamer.distributed_streamer.rank_dicts_map.values()
                        for origin in source_map.values()
                    }

                    received = {}
                    for req_id, chunk_idx, data in streamer.get_chunks():
                        received[(req_id, chunk_idx)] = data.cpu().numpy().tobytes()

                expected = {
                    (file_index, chunk_index)
                    for file_index in range(len(requests))
                    for chunk_index in range(len(requests[file_index].sizes))
                }

                with self.subTest(policy=policy, rank=self.rank):
                    self.assertEqual(set(received), expected,
                                     "every rank yields the whole model, whoever read it")

                    # The point of the test: chunks this rank never touched.
                    from_others = expected - mine
                    self.assertTrue(
                        from_others,
                        f"rank {self.rank} read everything under {policy}, so this proves nothing")

                    for file_index, chunk_index in sorted(from_others):
                        self.assertEqual(
                            received[(file_index, chunk_index)],
                            self._expected_bytes(file_index, chunk_index, chunk_size),
                            f"tensor ({file_index}, {chunk_index}) arrived under the wrong identity")

        if self.rank == 0:
            print(f"\n✅ Broadcast identity verified on all {self.world_size} ranks.")

    def test_1_lone_reader_does_not_wait_for_idle_peers(self):
        # One rank streams alone while the others wait outside the streamer. A collective in the
        # streamer would wait for peers that never join it, until the timeout.
        requests = self._prepare_file_requests([{"size": 100, "chunks": [40, 60]}])
        with open(requests[0].path, "rb") as f:
            expected = f.read()

        try:
            if self.rank == 0:
                chunks = [None] * len(requests[0].sizes)
                with patch.dict(os.environ, {"RUNAI_STREAMER_DIST_TIMEOUT": "5"}):
                    with DistributedStreamer() as streamer:
                        streamer.stream_files(requests, None, "cpu", False, discover_ranks=False)
                        for _, chunk_idx, data_tensor in streamer.get_chunks():
                            chunks[chunk_idx] = data_tensor.numpy().tobytes()
                self.assertEqual(expected, b"".join(chunks))
        finally:
            # the idle ranks wait here, outside the streamer
            dist.barrier()

    def test_9_failure_on_one_rank(self):
        if self.world_size < 2:
            self.skipTest("This failure test requires at least 2 processes.")
        
        file_specs = [{"size": 100, "chunks": [100]} for _ in range(self.world_size)]
        requests = self._prepare_file_requests(file_specs)
        
        env_vars = {"RUNAI_STREAMER_DIST_TIMEOUT": "5", "RUNAI_STREAMER_DIST": "1"}
        with patch.dict(os.environ, env_vars):
            if self.rank == 1:
                file_to_delete = requests[1].path
                if os.path.exists(file_to_delete):
                    os.remove(file_to_delete)
            
            with self.assertRaises(Exception):
                with DistributedStreamer() as streamer:
                    streamer.stream_files(requests, None, "cpu", True)
                    for _ in streamer.get_chunks():
                        pass
        
    def test_9_timeout_failure(self):
        if self.world_size < 2:
            self.skipTest("This timeout test requires at least 2 processes.")

        file_specs = [{"size": 100, "chunks": [100]} for _ in range(self.world_size)]
        requests = self._prepare_file_requests(file_specs)
        
        timeout_seconds = 1
        env_vars = {"RUNAI_STREAMER_DIST_TIMEOUT": str(timeout_seconds), "RUNAI_STREAMER_DIST": "1"}

        with patch.dict(os.environ, env_vars):
            with self.assertRaises(Exception) as context:
                with DistributedStreamer() as streamer:
                    # All ranks must pass the setup phase
                    streamer.stream_files(requests, None, "cpu", True)
                    
                    if self.rank == 1:
                        time.sleep(timeout_seconds + 2) # Sleep longer than timeout
                    
                    # All ranks try to get chunks. Rank 0 will time out waiting for rank 1
                    # at the internal barrier/broadcast inside get_chunks.
                    for _ in streamer.get_chunks():
                        pass
            error_message = str(context.exception).lower()
            is_expected_error = "timeout" in error_message or "connection closed by peer" in error_message or "timed out" in error_message
            self.assertTrue(
                is_expected_error,
                f"Rank {self.rank} caught an unexpected error: '{context.exception}'"
            )
            print(f"Rank {self.rank}: Correctly caught expected timeout exception.")

if __name__ == '__main__':
    unittest.main()