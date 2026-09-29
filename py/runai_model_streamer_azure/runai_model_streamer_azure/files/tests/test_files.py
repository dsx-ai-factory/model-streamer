# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import runai_model_streamer_azure.files.files as files
from azure.storage.blob import BlobProperties, BlobServiceClient, ContainerClient


class TestFiles(unittest.TestCase):
    def test_filter_allow(self):
        res = files._filter_allow(
            ["test_file1.txt1", "test_file2.txt2", "test_file3.txt3"],
            ["*.txt2"]
        )
        self.assertEqual(res, ["test_file2.txt2"])

    def test_filter_allow_full_path(self):
        res = files._filter_allow(
            ["test_file1.txt1", "dir/test_file2.txt2", "test_file3.txt3"],
            ["*.txt2"]
        )
        self.assertEqual(res, ["dir/test_file2.txt2"])

    def test_filter_ignore(self):
        res = files._filter_ignore(
            ["test_file1.txt1", "test_file2.txt2", "test_file3.txt3"],
            ["*.txt2"]
        )
        self.assertEqual(res, ["test_file1.txt1", "test_file3.txt3"])

    def test_removeprefix(self):
        res = files.removeprefix("test_prefix_string", "test_prefix_")
        self.assertEqual(res, "string")

    def test_removeprefix_no(self):
        res = files.removeprefix("test_prefix_string", "test_suffix_")
        self.assertEqual(res, "test_prefix_string")


class TestListFiles(unittest.TestCase):

    def make_blob(self, name, size=10, metadata=None):
        blob = MagicMock(spec=BlobProperties)
        blob.name = name
        blob.size = size
        blob.metadata = metadata
        return blob
    
    def setUp(self):
        self.mock_container_client = MagicMock(spec=ContainerClient)
        self.mock_blob_client = MagicMock(spec=BlobServiceClient)
        self.mock_blob_client.get_container_client.return_value = (
            self.mock_container_client
        )

    def test_listfiles_recursive(self):
        # list_blobs and walk_blobs return an Iterable so list is fine
        test_blobs = [
            self.make_blob("file1.txt"),
            self.make_blob("dir1/test.txt"),
            self.make_blob("adls-dir1", size=0, metadata={"hdi_isfolder": "true"}),  # ADLS directory stub
            self.make_blob("dir2/", size=0, metadata=None),
            self.make_blob("empty-blob", size=0),
            self.make_blob("adls-caps", size=0, metadata={"Hdi_isfolder": "true"}),  
            self.make_blob("empty-file", size=0, metadata={"Hdi_isfolder": "false"}),
        
        ]

        self.mock_container_client.list_blobs.return_value = test_blobs
        _, _, result = files.list_files(self.mock_blob_client, "az://container/", recursive=True)
        self.assertEqual(result, ["file1.txt", "dir1/test.txt", "empty-blob", "empty-file"])
    
    def test_listfiles_non_recursive(self):
        test_blobs = [
            self.make_blob("file1.txt"),
            self.make_blob("adls-dir1", size=0, metadata={"hdi_isfolder": "true"}), 
            self.make_blob("dir2/"),
            self.make_blob("empty-blob", size=0),
            self.make_blob("adls-caps", size=0, metadata={"Hdi_isfolder": "true"}),  
            self.make_blob("empty-file", size=0, metadata={"Hdi_isfolder": "false"}),
        ]

        self.mock_container_client.walk_blobs.return_value = test_blobs
        _, _, result = files.list_files(self.mock_blob_client, "az://container/", )
        self.assertEqual(result, ["file1.txt", "empty-blob", "empty-file"])

    def test_listfiles_with_allow_pattern(self):
        blobs = [
            self.make_blob("models/weights/config.json"),
            self.make_blob("models/weights/model.safetensors"),
            self.make_blob("models/README")
        ]
       
        self.mock_container_client.list_blobs.return_value = blobs
        _, _, result = files.list_files(
            self.mock_blob_client, "az://container/", allow_pattern=["*.safetensors"], recursive=True
        )
        self.assertEqual(result,["models/weights/model.safetensors"])

    def test_listfiles_with_ignore_pattern(self):
        blobs = [
            self.make_blob("models/weights/config.json"),
            self.make_blob("models/weights/model.safetensors"),
            self.make_blob("models/README")
        ]
        self.mock_container_client.list_blobs.return_value = blobs
        _, _, result = files.list_files(
            self.mock_blob_client, "az://container/", ignore_pattern=["*.safetensors"], recursive=True
        )
        self.assertEqual(result, ["models/weights/config.json", "models/README"])


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
        # base_dir is "" when pulling from the root of a container (no prefix).
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
    boundary (_create_client/list_files/download_blob) mocked, so this
    keeps failing if a future change stops wiring _safe_destination_path
    into the download loop -- including the Azure-specific detail that the
    destination file is open()'d for write before the download itself.
    """
    def setUp(self):
        self.dst = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dst, ignore_errors=True)

    @patch("runai_model_streamer_azure.files.files._create_client")
    @patch("runai_model_streamer_azure.files.files.list_files")
    def test_pull_files_rejects_traversal_key_and_writes_nothing(
            self, mock_list_files, mock_create_client):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        malicious_key = "models/llama/" + ("../" * 12) + "tmp/pwned"
        mock_list_files.return_value = ("container", "models/llama/", [malicious_key])

        with self.assertRaises(ValueError):
            files.pull_files("az://container/models/llama/", self.dst)

        mock_client.get_container_client.return_value.get_blob_client.assert_not_called()
        self.assertEqual(os.listdir(self.dst), [])

    @patch("runai_model_streamer_azure.files.files._create_client")
    @patch("runai_model_streamer_azure.files.files.list_files")
    def test_pull_files_downloads_valid_nested_key(
            self, mock_list_files, mock_create_client):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_list_files.return_value = (
            "container", "models/llama/", ["models/llama/subdir/config.json"])
        mock_client.get_container_client.return_value.get_blob_client \
            .return_value.download_blob.return_value.readall.return_value = b"content"

        files.pull_files("az://container/models/llama/", self.dst)

        self.assertTrue(os.path.exists(os.path.join(self.dst, "subdir", "config.json")))

    @patch("runai_model_streamer_azure.files.files._create_client")
    @patch("runai_model_streamer_azure.files.files.list_files")
    def test_pull_files_downloads_deeply_nested_key(
            self, mock_list_files, mock_create_client):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_list_files.return_value = (
            "container", "models/llama/", ["models/llama/a/b/c/deep.safetensors"])
        mock_client.get_container_client.return_value.get_blob_client \
            .return_value.download_blob.return_value.readall.return_value = b"content"

        files.pull_files("az://container/models/llama/", self.dst)

        self.assertTrue(os.path.exists(os.path.join(self.dst, "a", "b", "c", "deep.safetensors")))


if __name__ == "__main__":
    unittest.main()
