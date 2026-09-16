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

            # Now the model read on the empty rank. All three patches reproduce distributed_streamer.py
            # exactly: is_distributed True (the fallback checks need a real process group otherwise),
            # the cross-rank tensor_names check skipped (it also needs a real process group - this
            # test is about ring reporting, not that check), and the empty-partition early return,
            # which sets reading_from_storage False and returns without touching the FileStreamer.
            def empty_partition(*_args, **_kwargs):
                streamer.distributed_streamer.reading_from_storage = False

            with patch.object(
                streamer, "set_is_distributed",
                side_effect=lambda *_a: setattr(streamer, "is_distributed", True),
            ), patch.object(
                streamer, "_assert_tensor_names_match_across_ranks",
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
        Test a valid header with missing/truncated tensor data - the file is physically
        shorter than the header declares.

        Now caught eagerly, at stream_file() time (a whole-file length check: header's declared
        total data bytes vs the file's actual size on disk), not lazily whenever get_tensors()
        happens to reach the short tensor. Local filesystem only - object storage size probing
        is not wired up here.
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

        # 1. Validate our Streamer - stream_file() itself must now reject this, before any
        # tensor is actually read.
        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "truncated"):
                streamer.stream_file(path, None, "cpu")

        # 2. Validate HF safetensors library behavior
        # HF safetensors validates file size against header claims on open/mmap.
        with self.assertRaises(Exception, msg="HF safetensors should raise on truncated data"):
            with safe_open(path, framework="pt", device="cpu") as f:
                # If it doesn't fail on open, try to access data
                for k in f.keys():
                    f.get_tensor(k)

    def test_file_longer_than_header_declares_raises(self):
        """The mirror image of truncation: extra trailing bytes the header never accounted for.

        Not a truncation, not an overlap, not a gap - the header is perfectly self-consistent and
        every declared tensor reads fine. But bytes exist on disk that no tensor claims, which
        means the file does not match its own header - worth rejecting up front rather than
        silently ignoring however many bytes got appended.
        """
        header_dict = {
            "test_tensor": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
        }
        json_str = json.dumps(header_dict)
        # Header declares exactly 10 bytes of data; write 25 - 15 bytes of unexplained trailing
        # garbage past what the header says the file should contain.
        bloated_data = b"\x00" * 25
        path = self.create_corrupted_safetensors("bloated.st", len(json_str), json_str, bloated_data)

        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "extra"):
                streamer.stream_file(path, None, "cpu")

    def test_gap_before_the_first_tensor_raises(self):
        """The pairwise gap/overlap check only ever compares CONSECUTIVE tensors to each other -
        it never checks that the first tensor (by sorted offset) starts exactly at the data
        section's beginning. A header claiming the first tensor starts at byte 5, when the data
        section actually starts at byte 0, leaves 5 unaccounted-for bytes that only a
        sum-of-declared-sizes length check catches.
        """
        header_dict = {
            "test_tensor": {"dtype": "U8", "shape": [10], "data_offsets": [5, 15]},
        }
        json_str = json.dumps(header_dict)
        # Physically: 5 bytes of untracked padding, then the 10 declared bytes.
        data = (b"\xff" * 5) + (b"\x00" * 10)
        path = self.create_corrupted_safetensors("leading_gap.st", len(json_str), json_str, data)

        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "extra"):
                streamer.stream_file(path, None, "cpu")

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

    # -------------------------------------------------------------------------
    # TENSOR_NAMES FILTER TESTS
    # -------------------------------------------------------------------------

    def test_tensor_names_none_preserves_default_behavior(self):
        """tensor_names=None must be byte-for-byte identical to today's default (no filter)."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "test_files", "test.safetensors")
        if not os.path.exists(file_path):
            self.skipTest(f"Original test file not found at {file_path}")

        our = {}
        with SafetensorsStreamer() as run_sf:
            run_sf.stream_file(file_path, None, "cpu", tensor_names=None)
            for name, tensor in run_sf.get_tensors():
                our[name] = tensor.clone()

        with safe_open(file_path, framework="pt", device="cpu") as f:
            their_names = set(f.keys())

        self.assertEqual(set(our.keys()), their_names)

    def test_tensor_names_filters_to_requested_subset(self):
        """Three tensors laid back to back on disk - A, B, C, 10 bytes each, filled with the
        bytes 1, 2, 3 respectively:

            byte offset:   0        10       20       30
                           |--A: 1s--|--B: 2s--|--C: 3s--|

        Request only A and C - B, in the MIDDLE, is skipped. This is deliberately NOT the first or
        last tensor: the remaining A+C bytes are no longer back to back on disk, which is exactly
        the layout that would break a naive "walk sizes cumulatively from the start" offset
        calculation (see FileChunks.contiguous). If the filter's offset math were wrong, this test
        would read C's tensor starting 10 bytes too early - i.e. it would come back full of 2s
        (B's data) instead of 3s.
        """
        header_dict = {
            "A": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
            "B": {"dtype": "U8", "shape": [10], "data_offsets": [10, 20]},
            "C": {"dtype": "U8", "shape": [10], "data_offsets": [20, 30]},
        }
        json_str = json.dumps(header_dict)
        tensor_data = (b"\x01" * 10) + (b"\x02" * 10) + (b"\x03" * 10)  # A=1s, B=2s, C=3s
        path = self.create_corrupted_safetensors("subset.st", len(json_str), json_str, tensor_data)

        with SafetensorsStreamer() as streamer:
            streamer.stream_file(path, None, "cpu", tensor_names={"A", "C"})
            tensors = {}
            for name, tensor in streamer.get_tensors():
                tensors[name] = tensor.clone()

        # B was never requested - it must not appear at all, not even with wrong/empty data.
        self.assertNotIn("B", tensors)
        self.assertEqual(set(tensors.keys()), {"A", "C"})

        # And the two we DID request must have their own, correct bytes - not each other's,
        # and not B's.
        self.assertTrue(torch.all(tensors["A"].eq(1)))
        self.assertTrue(torch.all(tensors["C"].eq(3)))

    def test_tensor_names_empty_set_raises(self):
        """An explicitly empty tensor_names must raise - None means 'no filter', set() means
        'load nothing', which is never useful and almost certainly a caller bug."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "test_files", "test.safetensors")
        if not os.path.exists(file_path):
            self.skipTest(f"Original test file not found at {file_path}")

        with SafetensorsStreamer() as streamer:
            with self.assertRaises(ValueError):
                streamer.stream_file(file_path, None, "cpu", tensor_names=set())

    def test_tensor_names_unknown_name_raises(self):
        """A name not present in the checkpoint must raise, naming the problem, not silently
        return fewer tensors than requested."""
        header_dict = {
            "A": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
        }
        json_str = json.dumps(header_dict)
        path = self.create_corrupted_safetensors("unknown_name.st", len(json_str), json_str, b"\x01" * 10)

        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "does_not_exist"):
                streamer.stream_file(path, None, "cpu", tensor_names={"A", "does_not_exist"})

    def test_tensor_names_does_not_bypass_header_validation_for_excluded_tensor(self):
        """A corrupted tensor elsewhere in the header must still fail the load, even when
        tensor_names never asks for it - one corruption undermines trust in the whole offset
        table, not just the tensor it touches. See test_overlapping_tensors for the base case."""
        header_dict = {
            "A": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
            "B": {"dtype": "U8", "shape": [10], "data_offsets": [10, 20]},
            "C": {"dtype": "U8", "shape": [10], "data_offsets": [15, 25]},
        }
        json_str = json.dumps(header_dict)
        path = self.create_corrupted_safetensors("filtered_overlap.st", len(json_str), json_str, b"\x00" * 25)

        with SafetensorsStreamer() as streamer:
            # Only A is requested - B and C (the overlapping pair) are excluded - but the load
            # must still fail, because the header as a whole is corrupted.
            with self.assertRaisesRegex(ValueError, "overlaps with next tensor"):
                streamer.stream_file(path, None, "cpu", tensor_names={"A"})

    def test_tensor_names_excluding_a_truncated_tensor_no_longer_avoids_its_error(self):
        """Superseded by the whole-file length check added in test_file_longer_than_header_declares_raises
        / test_truncated_tensor_data: truncation is now caught eagerly, for the whole file, before
        tensor_names filtering ever runs - same "validate everything regardless of filter" policy as
        test_tensor_names_does_not_bypass_header_validation_for_excluded_tensor, just extended from
        internal header self-consistency to physical file length. Excluding the truncated tensor no
        longer saves the load, on purpose."""
        header_dict = {
            "A": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
            "B": {"dtype": "U8", "shape": [100], "data_offsets": [10, 110]},
        }
        json_str = json.dumps(header_dict)
        # A's 10 bytes are all present; B claims 100 bytes but only 10 more are on disk.
        tensor_data = (b"\x01" * 10) + (b"\x00" * 10)
        path = self.create_corrupted_safetensors("filtered_truncated.st", len(json_str), json_str, tensor_data)

        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "truncated"):
                streamer.stream_file(path, None, "cpu", tensor_names={"A"})


if __name__ == "__main__":
    unittest.main()