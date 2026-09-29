# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import torch
import os
import struct
import json
import tempfile
import shutil
import humanize
from safetensors import safe_open
from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
)

# Constants for binary generation
HEADER_SIZE_FORMAT = "<Q"  # Little-endian unsigned long long (8 bytes)

class TestSafetensorsStreamer(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for generating corrupted files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Cleanup after tests
        shutil.rmtree(self.test_dir)

    def create_corrupted_safetensors(self, filename, header_len_val, header_content, tensor_data=b""):
        """
        Helper to craft raw safetensors files byte-by-byte.
        
        Args:
            header_len_val: The integer value to put in the first 8 bytes.
            header_content: String or bytes to put in the header body.
            tensor_data: Bytes to append after the header.
        """
        filepath = os.path.join(self.test_dir, filename)
        
        # Ensure header content is bytes
        if isinstance(header_content, str):
            header_content = header_content.encode('utf-8')
            
        with open(filepath, "wb") as f:
            # 1. Write the 8-byte length
            f.write(struct.pack(HEADER_SIZE_FORMAT, header_len_val))
            
            # 2. Write the JSON (or garbage) header
            f.write(header_content)
            
            # 3. Write the tensor data
            f.write(tensor_data)
        
        return filepath

    def ring_records(self, logs):
        return [r for r in logs.records if "CPU ring" in r.getMessage()]

    def test_the_ring_is_reported_once_per_load_with_the_rank(self):
        """One INFO line per load, carrying the rank.

        Both halves matter. ONCE: stream_files runs three times per load - twice for the safetensors
        metadata - so reporting where the ring is BUILT gives three lines, two of them describing an
        8 byte read. WITH THE RANK: every rank builds its own ring, so without it a distributed load
        emits N indistinguishable copies, which is why the line lives at the session boundary rather
        than inside FilesRequestsIteratorWithBuffer.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "test_files", "test.safetensors")
        if not os.path.exists(file_path):
            self.skipTest(f"Original test file not found at {file_path}")

        with self.assertLogs("runai_model_streamer", level="INFO") as logs:
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(file_path, None, "cpu")
                for _name, _tensor in run_sf.get_tensors():
                    pass

        records = self.ring_records(logs)
        self.assertEqual(len(records), 1, f"expected one ring line, got {[r.getMessage() for r in records]}")
        self.assertIn("Rank 0", records[0].getMessage())

        # ...and it describes the MODEL read, not one of the metadata reads: the payload it reports is
        # the file's tensor bytes. Asserting the size rather than the absence of a metadata-sized string
        # on purpose - `assertNotIn("8 Bytes", ...)` looks like it says that but is a substring of
        # "198 Bytes", so it fails on any fixture whose size happens to end in 8.
        with safe_open(file_path, framework="pt", device="cpu") as f:
            total = sum(f.get_tensor(name).nbytes for name in f.keys())
        self.assertIn(humanize.naturalsize(total, binary=True), records[0].getMessage())

        # the per-request line still exists for debugging - it is just below INFO now, so raising the
        # level brings back one line per stream_files (the two metadata reads plus the model read)
        with self.assertLogs("runai_model_streamer", level="DEBUG") as logs:
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(file_path, None, "cpu")
                for _name, _tensor in run_sf.get_tensors():
                    pass
        self.assertGreater(len(self.ring_records(logs)), 1)

    def test_ring_info_is_none_before_streaming(self):
        # a rank whose share is empty never builds an iterator, and list_files-only use never does
        # either; the reporter must treat that as normal rather than raising
        from runai_model_streamer.distributed_streamer import DistributedStreamer

        with DistributedStreamer() as streamer:
            self.assertIsNone(streamer.ring_info())

    def test_ring_info_is_none_for_a_rank_whose_partition_is_empty(self):
        """A rank that read no model data must report no ring - not the metadata read's ring.

        The FileStreamer is shared with the safetensors metadata reads, so requests_iterator is ALWAYS
        set by the time the model read starts. A rank whose share of the partition is empty returns from
        _distributedStreamer.stream_files before calling it again, so reading the iterator on demand
        answers with the 8 byte metadata ring - which printed against the model's byte total looks
        exactly like a ring clamped to a single buffer, the failure this report exists to expose.
        """
        from unittest.mock import patch
        from runai_model_streamer.distributed_streamer import DistributedStreamer
        from runai_model_streamer.file_streamer import FileChunks

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "test_files", "test.safetensors")
        if not os.path.exists(file_path):
            self.skipTest(f"Original test file not found at {file_path}")
        requests = [FileChunks.contiguous(0, file_path, 0, [os.path.getsize(file_path)])]

        with DistributedStreamer() as streamer:
            # A real read first, standing in for the metadata reads: it leaves an iterator behind.
            streamer.stream_files(requests, None, "cpu", False)
            for _chunk in streamer.get_chunks():
                pass
            self.assertIsNotNone(streamer.ring_info())

            # Now the model read on the empty rank. Both patches reproduce distributed_streamer.py
            # exactly: is_distributed True (the fallback checks need a real process group otherwise),
            # and the empty-partition early return, which sets reading_from_storage False and returns
            # without touching the FileStreamer.
            def empty_partition(*_args, **_kwargs):
                streamer.distributed_streamer.reading_from_storage = False

            with patch.object(
                streamer, "set_is_distributed",
                side_effect=lambda *_a: setattr(streamer, "is_distributed", True),
            ), patch.object(
                streamer.distributed_streamer, "stream_files", side_effect=empty_partition
            ):
                streamer.stream_files(requests, None, "cpu", True)

            self.assertIsNone(streamer.ring_info())

    def test_valid_file(self):
        # Assuming test_files exists relative to the script location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "test_files", "test.safetensors")
        
        if not os.path.exists(file_path):
            self.skipTest(f"Original test file not found at {file_path}")

        our = {}
        with SafetensorsStreamer() as run_sf:
            run_sf.stream_file(file_path, None, "cpu")
            for name, tensor in run_sf.get_tensors():
                # clone: the yielded tensor is a VIEW into a ring buffer that is recycled once
                # the generator advances, so anything compared after the loop must own its data
                our[name] = tensor.clone()

        their = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                their[name] = f.get_tensor(name)

        self.assertEqual(len(our), len(their))
        for name, our_tensor in our.items():
            self.assertTrue(our_tensor.is_contiguous())
            self.assertEqual(our_tensor.dtype, their[name].dtype)
            self.assertEqual(our_tensor.shape, their[name].shape)
            self.assertTrue(torch.all(our_tensor.eq(their[name])))

    # -------------------------------------------------------------------------
    # CORRUPTION TESTS
    # -------------------------------------------------------------------------

    def test_header_too_large(self):
        """Test the '18 Exabyte' crash scenario (MAX_HEADER_SIZE check)."""
        # Create a file claiming its header is 101 MB (just over the 100MB limit)
        huge_size = (100 * 1024 * 1024) + 1
        path = self.create_corrupted_safetensors("too_large.st", huge_size, b"{}")

        # 1. Validate our Streamer
        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "exceeds limit"):
                streamer.stream_file(path, None, "cpu")

        # 2. Validate HF safetensors library behavior
        # Should raise an exception because the header length is suspiciously large 
        # or the file is smaller than claimed header.
        with self.assertRaises(Exception, msg="HF safetensors should raise on oversized header"):
            with safe_open(path, framework="pt", device="cpu") as f:
                pass

    def test_invalid_json(self):
        """Test catching broken JSON syntax."""
        # A header that cuts off before closing brace
        bad_json = '{"test": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]' 
        path = self.create_corrupted_safetensors("bad_json.st", len(bad_json), bad_json)

        # 1. Validate our Streamer
        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                streamer.stream_file(path, None, "cpu")

        # 2. Validate HF safetensors library behavior
        with self.assertRaises(Exception, msg="HF safetensors should raise on invalid JSON"):
            with safe_open(path, framework="pt", device="cpu") as f:
                pass

    def test_payload_inconsistency_shape_mismatch(self):
        """
        Test logic where Shape * Dtype size != Offset Length.
        This prevents reading garbage data or segfaults.
        """
        # Tensor claims to be F32 (4 bytes) * 10 elements = 40 bytes.
        # But offsets only reserve 4 bytes [0, 4].
        header_dict = {
            "test_tensor": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [0, 4] 
            }
        }
        json_str = json.dumps(header_dict)
        # Provide 4 bytes of dummy data
        path = self.create_corrupted_safetensors("mismatch.st", len(json_str), json_str, b"\x00"*4)

        # 1. Validate our Streamer
        with SafetensorsStreamer() as streamer:
            # Expect the payload consistency check to fire
            with self.assertRaisesRegex(ValueError, "Shape claims 40 bytes.*but offsets reserve 4"):
                streamer.stream_file(path, None, "cpu")

        # 2. Validate HF safetensors library behavior
        # HF safetensors validates that (end - start) matches the byte size of shape * dtype
        with self.assertRaises(Exception, msg="HF safetensors should raise on shape/offset mismatch"):
            with safe_open(path, framework="pt", device="cpu") as f:
                pass

    def test_overlapping_tensors(self):
        """Test logic ensuring tensors do not overlap in memory."""
        # Tensor A: [0, 10]
        # Tensor B: [5, 15] (Starts inside A)
        header_dict = {
            "A": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
            "B": {"dtype": "U8", "shape": [10], "data_offsets": [5, 15]}
        }
        json_str = json.dumps(header_dict)
        path = self.create_corrupted_safetensors("overlap.st", len(json_str), json_str, b"\x00"*20)

        # 1. Validate our Streamer
        with SafetensorsStreamer() as streamer:
            # Note: Dictionary ordering in python < 3.7 might make A or B come first.
            # Our code sorts by offset, so A (start=0) comes before B (start=5).
            # The error should trigger when checking B.
            with self.assertRaisesRegex(ValueError, "overlaps with next tensor"):
                streamer.stream_file(path, None, "cpu")

        # 2. Validate HF safetensors library behavior
        # The safetensors spec strictly forbids overlapping tensors.
        with self.assertRaises(Exception, msg="HF safetensors should raise on overlapping tensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                pass

    def test_unknown_dtype(self):
        """Test handling of unsupported data types."""
        header_dict = {
            "test_tensor": {
                "dtype": "ALIEN_TYPE_128",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        }
        json_str = json.dumps(header_dict)
        path = self.create_corrupted_safetensors("bad_dtype.st", len(json_str), json_str, b"\x00"*4)

        # 1. Validate our Streamer
        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "Unsupported dtype.*ALIEN_TYPE_128"):
                streamer.stream_file(path, None, "cpu")

        # 2. Validate HF safetensors library behavior
        with self.assertRaises(Exception, msg="HF safetensors should raise on unknown dtype"):
            with safe_open(path, framework="pt", device="cpu") as f:
                pass

    def test_truncated_file_header(self):
        """Test a file that ends inside the header."""
        # Claim header is 100 bytes, but file only has 10 bytes total
        path = self.create_corrupted_safetensors("truncated.st", 100, b"short")
        
        # 1. Validate our Streamer
        # Depending on implementation, this might raise JSON error or Truncated error.
        # But it MUST raise something.
        with SafetensorsStreamer() as streamer:
            with self.assertRaises(ValueError):
                streamer.stream_file(path, None, "cpu")

        # 2. Validate HF safetensors library behavior
        with self.assertRaises(Exception, msg="HF safetensors should raise on truncated header"):
            with safe_open(path, framework="pt", device="cpu") as f:
                pass

    def test_truncated_tensor_data(self):
        """
        Test a valid header with missing/truncated tensor data.
        Verifies that create_torch_tensor raises ValueError if buffer size mismatch.
        """
        # Header claims 100 bytes of data (U8 x 100)
        header_dict = {
            "test_tensor": {
                "dtype": "U8",
                "shape": [100],
                "data_offsets": [0, 100]
            }
        }
        json_str = json.dumps(header_dict)
        
        # We only provide 10 bytes of actual data instead of 100
        truncated_data = b"\x00" * 10
        path = self.create_corrupted_safetensors("truncated_body.st", len(json_str), json_str, truncated_data)

        # 1. Validate our Streamer
        with SafetensorsStreamer() as streamer:
            # stream_file reads the header (which is fine)
            streamer.stream_file(path, None, "cpu")
            
            # get_tensors attempts to read the body. 
            # If truncation occurs, the buffer check in create_torch_tensor MUST fail.
            with self.assertRaises(ValueError):
                for _ in streamer.get_tensors():
                    pass

        # 2. Validate HF safetensors library behavior
        # HF safetensors validates file size against header claims on open/mmap.
        with self.assertRaises(Exception, msg="HF safetensors should raise on truncated data"):
            with safe_open(path, framework="pt", device="cpu") as f:
                # If it doesn't fail on open, try to access data
                for k in f.keys():
                    f.get_tensor(k)

    def test_holes_between_tensors(self):
        """
        Test logic ensuring gaps/holes between tensors are not allowed.
        This verifies we throw exceptions for non-contiguous tensor storage.
        """
        # Tensor A: [0, 10] (10 bytes)
        # Gap: [10, 20] (10 bytes of unused data)
        # Tensor B: [20, 30] (10 bytes)
        header_dict = {
            "A": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
            "B": {"dtype": "U8", "shape": [10], "data_offsets": [20, 30]}
        }
        json_str = json.dumps(header_dict)
        
        # Total data needed: 30 bytes (up to end of B)
        # We fill it with zeros. The gap bytes (index 10-19) are just unused.
        tensor_data = b"\x00" * 30
        
        path = self.create_corrupted_safetensors("holes.st", len(json_str), json_str, tensor_data)

        with SafetensorsStreamer() as streamer:
            with self.assertRaises(ValueError):
                streamer.stream_file(path, None, "cpu")
 
        with self.assertRaises(Exception):
            with safe_open(path, framework="pt", device="cpu") as f:
                pass

    def test_unsorted_header_offsets_is_allowed(self):
        """
        Test that the order of keys in the JSON header does NOT have to match the 
        order of data in the file. The implementation should sort them by offset 
        internally before processing.
        """
        # Physically: "First" is at [0, 10], "Second" is at [10, 20].
        # In JSON: We put "Second" before "First" to verify sorting logic works.
        header_dict = {
            "Second": {"dtype": "U8", "shape": [10], "data_offsets": [10, 20]},
            "First":  {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
        }
        
        # Data: 20 bytes total. 
        # First 10 bytes = 1, Second 10 bytes = 2
        tensor_data = (b"\x01" * 10) + (b"\x02" * 10)
        
        json_str = json.dumps(header_dict)
        path = self.create_corrupted_safetensors("unsorted.st", len(json_str), json_str, tensor_data)

        # 1. Validate our Streamer (Should Succeed)
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(path, None, "cpu")
            tensors = {}
            for name, tensor in streamer.get_tensors():
                # clone: the yielded tensor is a VIEW into a ring buffer that is recycled once
                # the generator advances, so anything compared after the loop must own its data
                tensors[name] = tensor.clone()
            
            self.assertIn("First", tensors)
            self.assertIn("Second", tensors)
            
            # Verify data content to ensure we read the correct offsets
            # "First" should be all 1s
            self.assertTrue(torch.all(tensors["First"].eq(1)))
            # "Second" should be all 2s
            self.assertTrue(torch.all(tensors["Second"].eq(2)))

        # 2. Validate HF safetensors library behavior (Should Succeed)
        with safe_open(path, framework="pt", device="cpu") as f:
            tensors = {}
            tFirst = f.get_tensor("First")
            tSecond = f.get_tensor("Second")
            
            self.assertTrue(torch.all(tFirst.eq(1)))
            self.assertTrue(torch.all(tSecond.eq(2)))

if __name__ == "__main__":
    unittest.main()