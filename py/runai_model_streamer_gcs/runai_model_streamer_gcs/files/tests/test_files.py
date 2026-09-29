# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import runai_model_streamer_gcs.files.files as files


class TestFiles(unittest.TestCase):
    def test_filter_allow(self):
        res = files._filter_allow(
            ["test_file1.txt1", "test_file2.txt2", "test_file3.txt3"],
            patterns=["*.txt2"]
        )
        self.assertEqual(res, ["test_file2.txt2"])

    def test_filter_allow_full_path(self):
        res = files._filter_allow(
            ["test_file1.txt1", "dir/test_file2.txt2", "test_file3.txt3"],
            patterns=["*.txt2"]
        )
        self.assertEqual(res, ["dir/test_file2.txt2"])

    def test_filter_ignore(self):
        res = files._filter_ignore(
            ["test_file1.txt1", "test_file2.txt2", "test_file3.txt3"],
            patterns=["*.txt2"]
        )
        self.assertEqual(res, ["test_file1.txt1", "test_file3.txt3"])

    def test_removeprefix(self):
        res = files.removeprefix("test_prefix_string", "test_prefix_")
        self.assertEqual(res, "string")

    def test_removeprefix_no(self):
        res = files.removeprefix("test_prefix_string", "test_suffix_")
        self.assertEqual(res, "test_prefix_string")


class TestSafeDestinationPath(unittest.TestCase):
    def setUp(self):
        self.dst = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dst, ignore_errors=True)

    def test_allows_valid_nested_path(self):
        result = files._safe_destination_path(
            self.dst, "models/llama/", "models/llama/subdir/config.json")
        self.assertEqual(
            result, os.path.realpath(os.path.join(self.dst, "subdir/config.json")))

    def test_allows_valid_top_level_path(self):
        result = files._safe_destination_path(
            self.dst, "models/llama/", "models/llama/config.json")
        self.assertEqual(
            result, os.path.realpath(os.path.join(self.dst, "config.json")))

    def test_rejects_traversal_object_name(self):
        with self.assertRaises(ValueError):
            files._safe_destination_path(
                self.dst, "models/llama/",
                "models/llama/../../../../etc/cron.d/malicious")

    def test_rejects_traversal_object_name_not_matching_base_dir(self):
        with self.assertRaises(ValueError):
            files._safe_destination_path(
                self.dst, "models/llama/", "../../etc/passwd")

    def test_rejects_sibling_prefix_collision(self):
        # base_dir "models/llama" (no trailing slash) is a plain string prefix
        # of "models/llama_backup/...", but it's a different, unrelated key.
        with self.assertRaises(ValueError):
            files._safe_destination_path(
                self.dst, "models/llama", "models/llama_backup/weights.bin")

    def test_rejects_object_name_unrelated_to_base_dir(self):
        with self.assertRaises(ValueError):
            files._safe_destination_path(
                self.dst, "models/llama/", "other/path/secret.txt")

    def test_allows_top_level_key_when_base_dir_is_empty(self):
        # base_dir is "" when pulling from the root of a bucket (no prefix).
        result = files._safe_destination_path(self.dst, "", "config.json")
        self.assertEqual(result, os.path.realpath(os.path.join(self.dst, "config.json")))

    def test_allows_leading_slash_object_key_when_base_dir_is_empty(self):
        # A key with a literal leading "/" is unusual but valid; os.path.join
        # would otherwise treat it as absolute and discard dst entirely.
        result = files._safe_destination_path(self.dst, "", "/config.json")
        self.assertEqual(result, os.path.realpath(os.path.join(self.dst, "config.json")))

    def test_allows_double_slash_at_prefix_boundary(self):
        # A doubled "/" right where base_dir ends leaves a leading "/" on the
        # remainder after the prefix is stripped off.
        result = files._safe_destination_path(
            self.dst, "models/llama/", "models/llama//config.json")
        self.assertEqual(result, os.path.realpath(os.path.join(self.dst, "config.json")))


class TestPullFilesTraversalRegression(unittest.TestCase):
    """Exercises the real pull_files() loop end-to-end, with only the SDK
    boundary (_create_client/list_files/download_to_filename) mocked, so
    this keeps failing if a future change stops wiring
    _safe_destination_path into the download loop.

    Not run against a real backend: fake-gcs-server itself builds its local
    storage path from the raw object key without sanitizing it, so a blob
    named with ".." segments gets written outside its own data directory
    at upload time -- before pull_files() is ever involved -- making a
    real-emulator reproduction indistinguishable from the emulator's own
    (unrelated) bug.
    """
    def setUp(self):
        self.dst = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dst, ignore_errors=True)

    @patch("runai_model_streamer_gcs.files.files._create_client")
    @patch("runai_model_streamer_gcs.files.files.list_files")
    def test_pull_files_rejects_traversal_key_and_writes_nothing(
            self, mock_list_files, mock_create_client):
        mock_gcs = MagicMock()
        mock_create_client.return_value = mock_gcs
        malicious_key = "models/llama/" + ("../" * 12) + "tmp/pwned"
        mock_list_files.return_value = ("bucket", "models/llama/", [malicious_key])

        with self.assertRaises(ValueError):
            files.pull_files("gs://bucket/models/llama/", self.dst)

        mock_gcs.get_bucket.return_value.blob.return_value.download_to_filename.assert_not_called()
        self.assertEqual(os.listdir(self.dst), [])

    @patch("runai_model_streamer_gcs.files.files._create_client")
    @patch("runai_model_streamer_gcs.files.files.list_files")
    def test_pull_files_downloads_valid_nested_key(
            self, mock_list_files, mock_create_client):
        mock_gcs = MagicMock()
        mock_create_client.return_value = mock_gcs
        mock_list_files.return_value = (
            "bucket", "models/llama/", ["models/llama/subdir/config.json"])

        def fake_download_to_filename(destination_file):
            with open(destination_file, "wb") as f:
                f.write(b"content")
        mock_gcs.get_bucket.return_value.blob.return_value.download_to_filename.side_effect = (
            fake_download_to_filename)

        files.pull_files("gs://bucket/models/llama/", self.dst)

        self.assertTrue(os.path.exists(os.path.join(self.dst, "subdir", "config.json")))

    @patch("runai_model_streamer_gcs.files.files._create_client")
    @patch("runai_model_streamer_gcs.files.files.list_files")
    def test_pull_files_downloads_deeply_nested_key(
            self, mock_list_files, mock_create_client):
        mock_gcs = MagicMock()
        mock_create_client.return_value = mock_gcs
        mock_list_files.return_value = (
            "bucket", "models/llama/", ["models/llama/a/b/c/deep.safetensors"])

        def fake_download_to_filename(destination_file):
            with open(destination_file, "wb") as f:
                f.write(b"content")
        mock_gcs.get_bucket.return_value.blob.return_value.download_to_filename.side_effect = (
            fake_download_to_filename)

        files.pull_files("gs://bucket/models/llama/", self.dst)

        self.assertTrue(os.path.exists(os.path.join(self.dst, "a", "b", "c", "deep.safetensors")))


if __name__ == "__main__":
    unittest.main()
