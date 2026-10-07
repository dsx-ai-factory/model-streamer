# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from botocore import UNSIGNED

import runai_model_streamer_s3.files.files as files


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


def _env_without_unsigned():
    env = os.environ.copy()
    env.pop(files.RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR, None)
    return env


class TestBuildClientConfig(unittest.TestCase):
    def test_unsigned_enabled_when_one(self):
        with patch.dict(os.environ, {files.RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "1"}):
            config = files._build_client_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.signature_version, UNSIGNED)

    def test_unsigned_disabled_when_zero(self):
        with patch.dict(os.environ, {files.RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "0"}):
            config = files._build_client_config()
        sig = config.signature_version if config else None
        self.assertNotEqual(sig, UNSIGNED)

    def test_unsigned_disabled_when_absent(self):
        with patch.dict(os.environ, _env_without_unsigned(), clear=True):
            config = files._build_client_config()
        sig = config.signature_version if config else None
        self.assertNotEqual(sig, UNSIGNED)


class TestBuildS3Client(unittest.TestCase):
    @patch("runai_model_streamer_s3.files.files.boto3")
    @patch("runai_model_streamer_s3.files.files.get_credentials")
    def test_credentials_used_when_unsigned_disabled(self, mock_get_credentials, mock_boto3):
        mock_session = MagicMock()
        mock_get_credentials.return_value = (mock_session, MagicMock())
        with patch.dict(os.environ, {files.RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "0"}):
            files._build_s3_client(None)
        mock_get_credentials.assert_called_once()
        mock_session.client.assert_called_once()
        mock_boto3.client.assert_not_called()

    @patch("runai_model_streamer_s3.files.files.boto3")
    @patch("runai_model_streamer_s3.files.files.get_credentials")
    def test_credentials_used_when_unsigned_absent(self, mock_get_credentials, mock_boto3):
        mock_session = MagicMock()
        mock_get_credentials.return_value = (mock_session, MagicMock())
        with patch.dict(os.environ, _env_without_unsigned(), clear=True):
            files._build_s3_client(None)
        mock_get_credentials.assert_called_once()
        mock_session.client.assert_called_once()
        mock_boto3.client.assert_not_called()

    @patch("runai_model_streamer_s3.files.files.boto3")
    @patch("runai_model_streamer_s3.files.files.get_credentials")
    def test_credentials_not_used_when_unsigned_enabled(self, mock_get_credentials, mock_boto3):
        mock_get_credentials.return_value = (None, MagicMock())
        with patch.dict(os.environ, {files.RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "1"}):
            files._build_s3_client(None)
        mock_get_credentials.assert_called_once()
        mock_boto3.client.assert_called_once()


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
    boundary (_build_s3_client/list_files/download_file) mocked, so this
    keeps failing if a future change stops wiring _safe_destination_path
    into the download loop.

    Not run against a real backend: S3-compatible servers commonly reject ".."
    in object keys at PutObject time, so a real upload can rarely carry a
    traversal key to reproduce this with. This test stays mock-only
    regardless of which backend the other S3 integration tests use.
    """
    def setUp(self):
        self.dst = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dst, ignore_errors=True)

    @patch("runai_model_streamer_s3.files.files._build_s3_client")
    @patch("runai_model_streamer_s3.files.files.list_files")
    def test_pull_files_rejects_traversal_key_and_writes_nothing(
            self, mock_list_files, mock_build_client):
        mock_s3 = MagicMock()
        mock_build_client.return_value = mock_s3
        malicious_key = "models/llama/" + ("../" * 12) + "tmp/pwned"
        mock_list_files.return_value = ("bucket", "models/llama/", [malicious_key])

        with self.assertRaises(ValueError):
            files.pull_files("s3://bucket/models/llama/", self.dst)

        mock_s3.download_file.assert_not_called()
        self.assertEqual(os.listdir(self.dst), [])

    @patch("runai_model_streamer_s3.files.files._build_s3_client")
    @patch("runai_model_streamer_s3.files.files.list_files")
    def test_pull_files_downloads_valid_nested_key(
            self, mock_list_files, mock_build_client):
        mock_s3 = MagicMock()
        mock_build_client.return_value = mock_s3
        mock_list_files.return_value = (
            "bucket", "models/llama/", ["models/llama/subdir/config.json"])

        def fake_download_file(bucket, key, destination_file):
            with open(destination_file, "wb") as f:
                f.write(b"content")
        mock_s3.download_file.side_effect = fake_download_file

        files.pull_files("s3://bucket/models/llama/", self.dst)

        self.assertTrue(os.path.exists(os.path.join(self.dst, "subdir", "config.json")))

    @patch("runai_model_streamer_s3.files.files._build_s3_client")
    @patch("runai_model_streamer_s3.files.files.list_files")
    def test_pull_files_downloads_deeply_nested_key(
            self, mock_list_files, mock_build_client):
        mock_s3 = MagicMock()
        mock_build_client.return_value = mock_s3
        mock_list_files.return_value = (
            "bucket", "models/llama/", ["models/llama/a/b/c/deep.safetensors"])

        def fake_download_file(bucket, key, destination_file):
            with open(destination_file, "wb") as f:
                f.write(b"content")
        mock_s3.download_file.side_effect = fake_download_file

        files.pull_files("s3://bucket/models/llama/", self.dst)

        self.assertTrue(os.path.exists(os.path.join(self.dst, "a", "b", "c", "deep.safetensors")))


if __name__ == "__main__":
    unittest.main()
