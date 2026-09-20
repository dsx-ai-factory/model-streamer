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



class TestTensorNamesDistributedFiltering(unittest.TestCase):
    """tensor_names is filtered LOCALLY, per rank - there is no cross-rank check that every rank
    was asked for the same set (see usage.md: callers must ensure this themselves, or risk a hang
    bounded by RUNAI_STREAMER_DIST_TIMEOUT). This class only covers same-selection behavior.
    """

    # The real names baked into test_files/test.safetensors.
    ALL_NAMES = [
        "tensor1.bfloat16", "tensor1.bool", "tensor1.float16", "tensor1.float32", "tensor1.float64",
        "tensor1.int16", "tensor1.int32", "tensor1.int64", "tensor1.int8", "tensor1.uint8",
    ]

    ENV_VARS = {
        "RUNAI_STREAMER_DIST": "1",
        "RUNAI_STREAMER_DIST_BUFFER_MIN_BYTESIZE": "0",
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
