import unittest
import torch
import torch.distributed as dist
import os
import shutil
from safetensors import safe_open
from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
)

class TestDistributedSafetensorsStreamer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

    def setUp(self):
        dist.barrier()

    def tearDown(self):
        dist.barrier()

    def test_distributed_safetensors_streamer(self):
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")
        file_path = os.path.join(file_dir, "test.safetensors")
        our = {}
        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with unittest.mock.patch.dict(os.environ, env_vars):
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(file_path, None, "cpu", True)
                for name, tensor in run_sf.get_tensors():
                    our[name] = tensor.clone().detach() # because distributed streamer has internal reusable buffer

        their = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                their[name] = f.get_tensor(name)

        self.assertEqual(len(our.items()), len(their.items()))
        for name, our_tensor in our.items():
            self.assertTrue(our_tensor.is_contiguous())
            self.assertEqual(our_tensor.dtype, their[name].dtype)
            self.assertEqual(our_tensor.shape, their[name].shape)

            res = torch.all(our_tensor.eq(their[name]))

            self.assertTrue(res)


    def test_non_distributed_safetensors_streamer(self):

        rank = dist.get_rank()
        if rank == 0:
            file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_files", "test.safetensors"
            )
        else:
            file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_files", "test_empty.safetensors"
            )

        our = {}
        env_vars = {"RUNAI_STREAMER_DIST": "1", "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0"}
        with unittest.mock.patch.dict(os.environ, env_vars):
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(file_path, None, "cpu", False)
                for name, tensor in run_sf.get_tensors():
                    our[name] = tensor.clone().detach() # because distributed streamer has internal reusable buffer

        if rank != 0:
            self.assertEqual(len(our.items()), 0)

        their = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                their[name] = f.get_tensor(name)

        self.assertEqual(len(our.items()), len(their.items()))
        for name, our_tensor in our.items():
            self.assertTrue(our_tensor.is_contiguous())
            self.assertEqual(our_tensor.dtype, their[name].dtype)
            self.assertEqual(our_tensor.shape, their[name].shape)

            res = torch.all(our_tensor.eq(their[name]))

            self.assertTrue(res)



class TestTensorNamesCrossRankConsistency(unittest.TestCase):
    """tensor_names is filtered LOCALLY, per rank, before DistributedStreamer.stream_files() ever
    runs - nothing today checks every rank was asked for the same set. A mismatch means each rank
    partitions a differently sized workload, which is exactly the shape of a distributed collective
    deadlock: one rank's broadcast loop exits before another's, so ranks still inside dist.broadcast()
    block forever waiting on a peer that already left. RUNAI_STREAMER_DIST_TIMEOUT is set short here
    so a real hang fails loudly (a timeout RuntimeError from the group itself) instead of hanging the
    whole torchrun run.
    """

    # The real names baked into test_files/test.safetensors.
    ALL_NAMES = [
        "tensor1.bfloat16", "tensor1.bool", "tensor1.float16", "tensor1.float32", "tensor1.float64",
        "tensor1.int16", "tensor1.int32", "tensor1.int64", "tensor1.int8", "tensor1.uint8",
    ]

    ENV_VARS = {
        "RUNAI_STREAMER_DIST": "1",
        "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0",
        "RUNAI_STREAMER_DIST_TIMEOUT": "15",
    }

    @classmethod
    def setUpClass(cls):
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

    def setUp(self):
        dist.barrier()

    def tearDown(self):
        dist.barrier()

    def _file_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files", "test.safetensors")

    def test_different_count_across_ranks_raises_not_hangs(self):
        # Rank 0 asks for 5 tensors, rank 1 asks for 3 - a plain size mismatch.
        my_names = set(self.ALL_NAMES[:5]) if self.rank == 0 else set(self.ALL_NAMES[:3])

        with unittest.mock.patch.dict(os.environ, self.ENV_VARS):
            with SafetensorsStreamer() as run_sf:
                with self.assertRaises(Exception):
                    run_sf.stream_file(self._file_path(), None, "cpu", True, tensor_names=my_names)

    def test_same_count_different_membership_across_ranks_raises_not_hangs(self):
        # Both ranks ask for 3 tensors, and share 2 of them - same size, genuinely different sets.
        if self.rank == 0:
            my_names = {self.ALL_NAMES[0], self.ALL_NAMES[1], self.ALL_NAMES[2]}
        else:
            my_names = {self.ALL_NAMES[0], self.ALL_NAMES[1], self.ALL_NAMES[5]}

        with unittest.mock.patch.dict(os.environ, self.ENV_VARS):
            with SafetensorsStreamer() as run_sf:
                with self.assertRaises(Exception):
                    run_sf.stream_file(self._file_path(), None, "cpu", True, tensor_names=my_names)

    def test_same_set_different_order_and_collection_type_succeeds(self):
        # Same 3 names on both ranks, but as different collection types in different orders - the
        # doc is explicit that tensor_names is "treated as a set", so this must succeed identically
        # to the same-order case, not be flagged as a mismatch.
        names = [self.ALL_NAMES[2], self.ALL_NAMES[0], self.ALL_NAMES[1]]
        my_names = names if self.rank == 0 else set(reversed(names))

        with unittest.mock.patch.dict(os.environ, self.ENV_VARS):
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(self._file_path(), None, "cpu", True, tensor_names=my_names)
                our = {}
                for name, tensor in run_sf.get_tensors():
                    our[name] = tensor.clone().detach()

        self.assertEqual(set(our.keys()), set(names))


if __name__ == "__main__":
    unittest.main()
