import unittest
import hashlib
import os
import torch
import torch.distributed as dist
from typing import List
import tempfile
import shutil
import pickle
import time
import random
from unittest.mock import patch

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    DistributedStreamer,
    RUNAI_STREAMER_CUDA_ALIGNMENT_ENV_VAR,
)
from runai_model_streamer.file_streamer import FileChunks
from runai_model_streamer.file_streamer.requests_iterator import (
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RUNAI_STREAMER_RING_BUFFERS_ENV_VAR_NAME,
)

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

        alignment = 512
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
            RUNAI_STREAMER_CUDA_ALIGNMENT_ENV_VAR: str(alignment),
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