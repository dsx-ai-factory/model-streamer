# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import torch
import os
import tempfile
import shutil
from pathlib import Path
from safetensors import safe_open
from unittest.mock import patch

# Import the class to be tested
from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
    pull_files,
    list_safetensors,
)

from runai_model_streamer.safetensors_streamer.streamer_mock import StreamerPatcher

class TestSafetensorsStreamerMock(unittest.TestCase):

    def setUp(self):
        """Find the test file path once for all tests."""
        self.file_name = "test.safetensors"
        self.file_dir = "test_files"
        self.local_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_file_path = os.path.join(self.local_dir, self.file_dir, self.file_name)
        
        # Make sure the test file exists before running any test
        if not os.path.exists(self.local_file_path):
            raise FileNotFoundError(
                f"Test file 'test.safetensors' not found in {self.local_dir}. "
                "Please add it to run the tests."
            )

    # We patch `SafetensorsStreamer` *where it is imported*, which is
    # in this local test module. `__name__` resolves to the module name.
    @patch(__name__ + '.SafetensorsStreamer')
    def test_safetensors_streamer_S3_MOCK(self, mock_streamer_class):
        """
        Mocked test: Verifies streaming from a FAKE S3 path
        is correctly patched and reads the local file.
        """
        
        # 1. Create a fake S3 path
        fake_s3_path = f"s3://my-fake-bucket/{self.file_dir}/{self.file_name}"

        # 2. Initialize the Patcher, pointing to the *directory*
        patcher = StreamerPatcher(local_bucket_path=self.local_dir)
        
        # 3. Connect the patch to the patcher
        # When `SafetensorsStreamer()` is called, use the patcher's factory
        mock_streamer_class.side_effect = patcher.create_mock_streamer

        # 4. Run streamer logic, passing the FAKE S3 PATH
        our = {}
        with SafetensorsStreamer() as run_sf: # This call is now mocked
            # Pass the *fake* path here. The patcher will rewrite it.
            run_sf.stream_file(fake_s3_path, None, "cpu")
            for name, tensor in run_sf.get_tensors():
                # clone: the yielded tensor is a VIEW into a ring buffer that is recycled once
                # the generator advances, so anything compared after the loop must own its data
                our[name] = tensor.clone()

        # 5. Load baseline from the REAL LOCAL PATH
        their = {}
        with safe_open(self.local_file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                their[name] = f.get_tensor(name)

        # 6. Assertions 
        self.assertEqual(len(our.items()), len(their.items()))
        self.assertGreater(len(our.items()), 0, "No tensors loaded via mock")
        for name, our_tensor in our.items():
            self.assertTrue(our_tensor.is_contiguous())
            self.assertEqual(our_tensor.dtype, their[name].dtype)
            self.assertEqual(our_tensor.shape, their[name].shape)
            res = torch.all(our_tensor.eq(their[name]))
            self.assertTrue(res)

    @patch(__name__ + '.SafetensorsStreamer')
    def test_safetensors_streamer_GS_MOCK(self, mock_streamer_class):
        """
        Mocked test: Verifies streaming from a FAKE GS path
        is correctly patched and reads the local file.
        """
        
        # 1. Create a fake GS path
        fake_gs_path = f"gs://my-fake-bucket/{self.file_dir}/{self.file_name}"

        # 2. Initialize the Patcher
        patcher = StreamerPatcher(local_bucket_path=self.local_dir)
        
        # 3. Connect the patch to the patcher
        mock_streamer_class.side_effect = patcher.create_mock_streamer

        # 4. Run streamer logic, passing the FAKE GS PATH
        our = {}
        with SafetensorsStreamer() as run_sf:
            run_sf.stream_file(fake_gs_path, None, "cpu")
            for name, tensor in run_sf.get_tensors():
                # clone: the yielded tensor is a VIEW into a ring buffer that is recycled once
                # the generator advances, so anything compared after the loop must own its data
                our[name] = tensor.clone()

        # 5. Load baseline from the REAL LOCAL PATH
        their = {}
        with safe_open(self.local_file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                their[name] = f.get_tensor(name)

        # 6. Assertions
        self.assertEqual(len(our.items()), len(their.items()))
        self.assertGreater(len(our.items()), 0, "No tensors loaded via mock")
        for name, their_tensor in their.items():
            self.assertTrue(name in our, f"Tensor {name} missing from mocked run")
            self.assertTrue(torch.all(our[name].eq(their_tensor)))

    @patch(__name__ + '.list_safetensors')
    def test_list_safetensors_S3_MOCK(self, mock_list_safetensors):
        """
        Mocked test: Verifies list_safetensors shim logic.
        """

        fake_s3_path = f"s3://my-fake-bucket/{self.file_dir}"
        patcher = StreamerPatcher(local_bucket_path=self.local_dir)
        
        # Connect the patch to the patcher's shim method
        mock_list_safetensors.side_effect = patcher.shim_list_safetensors
        
        # Call the (now-mocked) function
        listed_files = list_safetensors(fake_s3_path)
        
        # Assertions
        self.assertIsInstance(listed_files, list)
        self.assertEqual(len(listed_files), 2, "No files listed by shim")
        self.assertTrue(
            any(f.endswith(self.file_name) for f in listed_files),
            f"{self.file_name} not found in listed files: {listed_files}"
        )

        expected_mock_prefix = "s3://my-fake-bucket/"
        for f in listed_files:
            self.assertTrue(
                f.startswith(expected_mock_prefix),
                f"File path '{f}' does not start with the expected mock prefix '{expected_mock_prefix}'"
            )

    @patch(__name__ + '.list_safetensors')
    def test_list_safetensors_GS_MOCK(self, mock_list_safetensors):
        """
        Mocked test: Verifies list_safetensors shim logic.
        """
        
        fake_gs_path = f"gs://my-fake-bucket/{self.file_dir}"
        patcher = StreamerPatcher(local_bucket_path=self.local_dir)
        
        # Connect the patch to the patcher's shim method
        mock_list_safetensors.side_effect = patcher.shim_list_safetensors
        
        # Call the (now-mocked) function
        listed_files = list_safetensors(fake_gs_path)
        
        # Assertions
        self.assertIsInstance(listed_files, list)
        self.assertGreater(len(listed_files), 0, "No files listed by shim")
        self.assertTrue(
            any(f.endswith(self.file_name) for f in listed_files),
            f"{self.file_name} not found in listed files: {listed_files}"
        )

        expected_mock_prefix = "gs://my-fake-bucket/"
        for f in listed_files:
            self.assertTrue(
                f.startswith(expected_mock_prefix),
                f"File path '{f}' does not start with the expected mock prefix '{expected_mock_prefix}'"
            )

    @patch(__name__ + '.pull_files')
    def test_pull_files_S3_MOCK(self, mock_pull_files):
        """
        Mocked test: Verifies pull_files shim logic.
        """
        
        fake_s3_path = f"s3://my-fake-bucket/{self.file_dir}"
        dest_dir = tempfile.mkdtemp()

        try:
            patcher = StreamerPatcher(local_bucket_path=self.local_dir)
            
            # Connect the patch to the patcher's shim method
            mock_pull_files.side_effect = patcher.shim_pull_files
            
            # Call the (now-mocked) function
            pull_files(
                model_path=fake_s3_path,
                dst=dest_dir,
                allow_pattern=["*.safetensors"], # Test the filtering
                ignore_pattern=None
            )
            
            # Assertions
            expected_file_path = os.path.join(dest_dir, self.file_name)
            self.assertTrue(
                os.path.exists(expected_file_path),
                f"File was not 'pulled' to {expected_file_path}"
            )
            
        finally:
            shutil.rmtree(dest_dir)

    @patch(__name__ + '.pull_files')
    def test_pull_files_GS_MOCK(self, mock_pull_files):
        """
        Mocked test: Verifies pull_files shim logic with a GS path.
        """
        
        fake_gs_path = f"gs://my-fake-bucket/{self.file_dir}"
        dest_dir = tempfile.mkdtemp()

        try:
            patcher = StreamerPatcher(local_bucket_path=self.local_dir)
            mock_pull_files.side_effect = patcher.shim_pull_files
            
            pull_files(
                model_path=fake_gs_path,
                dst=dest_dir,
                allow_pattern=["*.safetensors"],
                ignore_pattern=None
            )
            
            expected_file_path = os.path.join(dest_dir, self.file_name)
            self.assertTrue(
                os.path.exists(expected_file_path),
                f"File was not 'pulled' to {expected_file_path}"
            )
            
        finally:
            shutil.rmtree(dest_dir)

    @patch(__name__ + '.pull_files')
    def test_pull_files_GS_MOCK_No_Slash(self, mock_pull_files):
        """
        Mocked test: Verifies pull_files shim logic with a GS path.
        """
        
        fake_gs_path = f"gs://my-fake-bucket/{self.file_dir}"
        dest_dir = tempfile.mkdtemp()

        try:
            patcher = StreamerPatcher(local_bucket_path=self.local_dir)
            mock_pull_files.side_effect = patcher.shim_pull_files
            
            pull_files(
                model_path=fake_gs_path,
                dst=dest_dir,
                allow_pattern=["*.safetensors"],
                ignore_pattern=None
            )
            
            expected_file_path = os.path.join(dest_dir, self.file_name)
            self.assertTrue(
                os.path.exists(expected_file_path),
                f"File was not 'pulled' to {expected_file_path}"
            )
            
        finally:
            shutil.rmtree(dest_dir)

    @patch(__name__ + '.pull_files')
    def test_pull_files_GS_MOCK_With_Allow_Pattern(self, mock_pull_files):
        """
        Mocked test: Verifies pull_files shim logic with a GS path.
        """
        
        fake_gs_path = f"gs://my-fake-bucket/{self.file_dir}"
        dest_dir = tempfile.mkdtemp()

        try:
            patcher = StreamerPatcher(local_bucket_path=self.local_dir)
            mock_pull_files.side_effect = patcher.shim_pull_files
            
            pull_files(
                model_path=fake_gs_path,
                dst=dest_dir,
                allow_pattern=["*.bin"],
                ignore_pattern=None
            )
            
            expected_file_path = os.path.join(dest_dir, "test.bin")
            self.assertFalse(
                os.path.exists(expected_file_path),
                f"File was 'pulled' to {expected_file_path}"
            )
            
        finally:
            shutil.rmtree(dest_dir)

    @patch(__name__ + '.pull_files')
    def test_pull_files_GS_MOCK_With_Ignore_Pattern(self, mock_pull_files):
        """
        Mocked test: Verifies pull_files shim logic with a GS path.
        """
        
        fake_gs_path = f"gs://my-fake-bucket/{self.file_dir}"
        dest_dir = tempfile.mkdtemp()

        try:
            patcher = StreamerPatcher(local_bucket_path=self.local_dir)
            mock_pull_files.side_effect = patcher.shim_pull_files
            
            pull_files(
                model_path=fake_gs_path,
                dst=dest_dir,
                allow_pattern=["*.safetensors"],
                ignore_pattern=["test_empty.*"]
            )
            
            expected_file_path = os.path.join(dest_dir, self.file_name)
            self.assertTrue(
                os.path.exists(expected_file_path),
                f"File was not 'pulled' to {expected_file_path}"
            )

            expected_ignored_file_path = os.path.join(dest_dir, "test_empty.safetensors")
            self.assertFalse(
                os.path.exists(expected_ignored_file_path),
                f"File was 'pulled' to {expected_ignored_file_path}"
            )
            
        finally:
            shutil.rmtree(dest_dir)

# --- test shim_pull_files ---

class TestShimPullFiles(unittest.TestCase):
    def setUp(self):
        """
        Sets up a source directory acting as the 'cloud' bucket and a destination.
        """
        # Create a temporary directory for the test run
        self.test_dir_obj = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir_obj.name)
        
        self.mock_cloud_storage = self.test_path / "my-bucket"
        self.destination = self.test_path / "download_dest"
        
        self.mock_cloud_storage.mkdir()
        self.destination.mkdir()

        # Initialize the patcher pointing to our mock cloud bucket
        self.patcher = StreamerPatcher(local_bucket_path=str(self.mock_cloud_storage))


    def tearDown(self):
        # Clean up temporary directory
        self.test_dir_obj.cleanup()

    def test_recursive_copy_success(self):
        """
        CRITICAL TEST: This verifies that the Stage 2 bug is fixed.
        If the 'dirs[:] = ...' bug existed, 'subdir/deep_file.txt' would fail to copy.
        """
        # Setup "Cloud" structure: s3://my-bucket/model/
        bucket_path = self.mock_cloud_storage / "model"
        bucket_path.mkdir(parents=True)
        
        # Create files
        (bucket_path / "root_file.txt").write_text("content")
        (bucket_path / "subdir").mkdir()
        (bucket_path / "subdir" / "deep_file.txt").write_text("content")

        # Action
        self.patcher.shim_pull_files("s3://my-bucket/model", str(self.destination))

        # Assert
        self.assertTrue((self.destination / "root_file.txt").exists())
        self.assertTrue((self.destination / "subdir").exists())
        self.assertTrue((self.destination / "subdir" / "deep_file.txt").exists())

    def test_allow_patterns(self):
        """Verifies that allow_pattern restricts downloads."""
        bucket_path = self.mock_cloud_storage / "model"
        bucket_path.mkdir(parents=True)
        
        (bucket_path / "model.bin").write_text("binary")
        (bucket_path / "config.json").write_text("json")
        (bucket_path / "readme.txt").write_text("text")

        # Action: Only allow .json and .bin
        self.patcher.shim_pull_files(
               "s3://my-bucket/model", 
            str(self.destination), 
            allow_pattern=["*.json", "*.bin"]
        )

        # Assert
        self.assertTrue((self.destination / "model.bin").exists())
        self.assertTrue((self.destination / "config.json").exists())
        self.assertFalse((self.destination / "readme.txt").exists())

    def test_ignore_patterns(self):
        """Verifies that ignore_pattern excludes files."""
        bucket_path = self.mock_cloud_storage / "model"
        bucket_path.mkdir(parents=True)
        
        (bucket_path / "important.data").write_text("data")
        (bucket_path / "junk.tmp").write_text("junk")
        (bucket_path / ".hidden").write_text("hidden")

        # Action: Ignore .tmp and hidden files
        self.patcher.shim_pull_files(
            "gs://my-bucket/model", 
            str(self.destination), 
            ignore_pattern=["*.tmp", ".*"]
        )

        # Assert
        self.assertTrue((self.destination / "important.data").exists())
        self.assertFalse((self.destination / "junk.tmp").exists())
        self.assertFalse((self.destination / ".hidden").exists())

    def test_local_path_raises_error(self):
        """Verifies that providing a local path instead of s3/gs raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as excinfo:
            self.patcher.shim_pull_files("/local/path/on/disk", str(self.destination))
        
        self.assertIn("not implemented for file system paths", str(excinfo.exception))

    def test_no_matching_files(self):
        """Verifies behavior when source exists but filters exclude everything."""
        bucket_path = self.mock_cloud_storage / "model"
        bucket_path.mkdir(parents=True)
        (bucket_path / "file.txt").write_text("text")

        # Action: Allow only .csv (which doesn't exist)
        self.patcher.shim_pull_files(
            "s3://my-bucket/model", 
            str(self.destination), 
            allow_pattern=["*.csv"]
        )

        # Assert: Destination should be empty
        # list(generator) evaluates it; boolean check works on the list
        self.assertFalse(list(self.destination.glob("**/*")))

if __name__ == "__main__":
    unittest.main()