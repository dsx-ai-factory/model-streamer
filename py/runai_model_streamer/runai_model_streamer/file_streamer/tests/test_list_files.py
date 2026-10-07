# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.libstreamer.libstreamer import runai_list_files, runai_start, runai_end
from runai_model_streamer.s3_utils.s3_utils import S3Credentials


def _list_files_stub(entries):
    """Returns a side_effect for mock runai_list_files that fires the callback with given (path, size) pairs."""
    def stub(streamer, prefix, callback, is_recursive=True, allow_patterns=None, ignore_patterns=None):
        for path, size in entries:
            callback(path, size)
    return stub


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

class TestListFilesFilesystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write(self, rel_path, content=b"x"):
        full = os.path.join(self.temp_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(content)
        return full

    def test_basic_listing(self):
        self._write("a.txt", b"hello")
        self._write("b.safetensors", b"world!")
        results = FileStreamer().list_files(self.temp_dir)
        paths = {r[0] for r in results}
        self.assertIn(os.path.join(self.temp_dir, "a.txt"), paths)
        self.assertIn(os.path.join(self.temp_dir, "b.safetensors"), paths)

    def test_returns_correct_sizes(self):
        self._write("sized.bin", b"12345")
        results = FileStreamer().list_files(self.temp_dir)
        by_path = {r[0]: r[1] for r in results}
        self.assertEqual(by_path[os.path.join(self.temp_dir, "sized.bin")], 5)

    def test_recursive(self):
        self._write("root.txt", b"r")
        self._write("sub/nested.txt", b"n")
        nested = os.path.join(self.temp_dir, "sub", "nested.txt")

        recursive_paths = {r[0] for r in FileStreamer().list_files(self.temp_dir, is_recursive=True)}
        non_recursive_paths = {r[0] for r in FileStreamer().list_files(self.temp_dir, is_recursive=False)}

        self.assertIn(nested, recursive_paths)
        self.assertNotIn(nested, non_recursive_paths)

    def test_allow_pattern(self):
        self._write("model.safetensors", b"m")
        self._write("config.json", b"c")
        results = FileStreamer().list_files(self.temp_dir, allow_patterns=["*.safetensors"])
        paths = {r[0] for r in results}
        self.assertTrue(all(p.endswith(".safetensors") for p in paths))
        self.assertFalse(any(p.endswith(".json") for p in paths))

    def test_ignore_pattern(self):
        self._write("model.safetensors", b"m")
        self._write("config.json", b"c")
        results = FileStreamer().list_files(self.temp_dir, ignore_patterns=["*.json"])
        paths = {r[0] for r in results}
        self.assertFalse(any(p.endswith(".json") for p in paths))
        self.assertTrue(any(p.endswith(".safetensors") for p in paths))

    def test_empty_directory(self):
        results = FileStreamer().list_files(self.temp_dir)
        self.assertEqual(results, [])

    def test_nonexistent_path_raises(self):
        with self.assertRaises(ValueError):
            FileStreamer().list_files(os.path.join(self.temp_dir, "does_not_exist"))

    def test_callback_exception_is_reraised(self):
        # an exception raised inside the callback must propagate out of
        # runai_list_files rather than be swallowed by the ctypes boundary
        self._write("a.txt", b"hello")

        class BoomError(Exception):
            pass

        def raising_callback(path, size):
            raise BoomError("callback failed")

        streamer = runai_start()
        try:
            with self.assertRaises(BoomError):
                runai_list_files(streamer, self.temp_dir, raising_callback, is_recursive=True)
        finally:
            runai_end(streamer)


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

class TestListFilesS3(unittest.TestCase):
    PREFIX = "s3://my-bucket/models/"
    ENTRIES = [
        ("s3://my-bucket/models/model.safetensors", 1024),
        ("s3://my-bucket/models/config.json", 256),
    ]

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_basic_listing(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub(self.ENTRIES)
        results = FileStreamer().list_files(self.PREFIX)
        self.assertEqual(sorted(results), sorted(self.ENTRIES))

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_is_recursive_passed(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX, is_recursive=False)
        self.assertEqual(mock_rlf.call_args.kwargs["is_recursive"], False)

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_allow_patterns_passed(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX, allow_patterns=["*.safetensors"])
        self.assertEqual(mock_rlf.call_args.kwargs["allow_patterns"], ["*.safetensors"])

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_ignore_patterns_passed(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX, ignore_patterns=["*.json"])
        self.assertEqual(mock_rlf.call_args.kwargs["ignore_patterns"], ["*.json"])

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_set_credentials")
    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_list_files_does_not_set_credentials(self, mock_rlf, mock_set_creds):
        # list_files gets no credentials, so nothing is applied - it lists using the default provider chain
        mock_rlf.side_effect = _list_files_stub(self.ENTRIES)
        results = FileStreamer().list_files(self.PREFIX)
        mock_set_creds.assert_not_called()
        self.assertEqual(sorted(results), sorted(self.ENTRIES))

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_set_credentials")
    @patch("runai_model_streamer.file_streamer.file_streamer.s3_credentials_module")
    def test_handle_object_store_resolves_and_applies_credentials(self, mock_s3_mod, mock_set_creds):
        # handle_object_store resolves credentials via the credentials module (the boto3-resolved values under
        # RUNAI_STREAMER_NO_BOTO3_SESSION=0) and applies exactly those to the streamer, once
        resolved = S3Credentials(
            access_key_id="RESOLVED_AKID",
            secret_access_key="RESOLVED_SECRET",
            session_token="RESOLVED_TOKEN",
            region_name="us-west-2",
            endpoint="https://s3.example.com",
        )
        mock_s3_mod.get_credentials.return_value = (MagicMock(), resolved)

        fs = FileStreamer()
        sentinel_streamer = object()
        fs.handle_object_store("s3://bucket/key", sentinel_streamer, S3Credentials(access_key_id="INPUT"))

        mock_s3_mod.get_credentials.assert_called_once()
        mock_set_creds.assert_called_once()
        applied_streamer = mock_set_creds.call_args.args[0]
        applied_creds = mock_set_creds.call_args.args[1]
        self.assertIs(applied_streamer, sentinel_streamer)
        self.assertEqual(applied_creds.access_key_id, "RESOLVED_AKID")
        self.assertEqual(applied_creds.secret_access_key, "RESOLVED_SECRET")
        self.assertEqual(applied_creds.session_token, "RESOLVED_TOKEN")
        self.assertEqual(applied_creds.region_name, "us-west-2")
        self.assertEqual(applied_creds.endpoint, "https://s3.example.com")

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_set_credentials")
    @patch("runai_model_streamer.file_streamer.file_streamer.s3_credentials_module")
    def test_credentials_applied_to_each_distinct_streamer(self, mock_s3_mod, mock_set_creds):
        # Resolution happens once, but application happens once PER C++ handle: a second temporary streamer
        # (as an out-of-context list_files creates) must still get its own runai_set_credentials, even though
        # the resolution flag (s3_session) is already set from the first handle.
        resolved = S3Credentials(access_key_id="AKID", secret_access_key="SECRET")
        mock_s3_mod.get_credentials.return_value = (MagicMock(), resolved)

        fs = FileStreamer()
        streamer_1 = object()
        streamer_2 = object()
        fs.handle_object_store("s3://bucket/a", streamer_1, None)
        fs.handle_object_store("s3://bucket/b", streamer_2, None)

        # resolved only once (s3_session cached), but applied to both distinct handles
        mock_s3_mod.get_credentials.assert_called_once()
        self.assertEqual(mock_set_creds.call_count, 2)
        applied_streamers = [call.args[0] for call in mock_set_creds.call_args_list]
        self.assertEqual(applied_streamers, [streamer_1, streamer_2])

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_set_credentials")
    @patch("runai_model_streamer.file_streamer.file_streamer.s3_credentials_module")
    def test_credentials_applied_once_per_streamer(self, mock_s3_mod, mock_set_creds):
        # Repeated calls with the SAME handle (e.g. the per-file loop in stream_files) apply credentials only
        # once, not once per file.
        resolved = S3Credentials(access_key_id="AKID", secret_access_key="SECRET")
        mock_s3_mod.get_credentials.return_value = (MagicMock(), resolved)

        fs = FileStreamer()
        streamer = object()
        fs.handle_object_store("s3://bucket/a", streamer, None)
        fs.handle_object_store("s3://bucket/b", streamer, None)

        mock_set_creds.assert_called_once()


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------

class TestListFilesGCS(unittest.TestCase):
    PREFIX = "gs://my-bucket/models/"
    ENTRIES = [
        ("gs://my-bucket/models/model.safetensors", 2048),
        ("gs://my-bucket/models/config.json", 512),
    ]

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_basic_listing(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub(self.ENTRIES)
        results = FileStreamer().list_files(self.PREFIX)
        self.assertEqual(sorted(results), sorted(self.ENTRIES))

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_is_recursive_passed(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX, is_recursive=False)
        self.assertEqual(mock_rlf.call_args.kwargs["is_recursive"], False)

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_allow_patterns_passed(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX, allow_patterns=["*.safetensors"])
        self.assertEqual(mock_rlf.call_args.kwargs["allow_patterns"], ["*.safetensors"])

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_no_credentials_for_gcs(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX)
        # credentials are streamer-scoped now; list_files never passes a per-call params/credentials arg
        self.assertNotIn("params", mock_rlf.call_args.kwargs)


# ---------------------------------------------------------------------------
# Azure
# ---------------------------------------------------------------------------

class TestListFilesAzure(unittest.TestCase):
    PREFIX = "az://my-container/models/"
    ENTRIES = [
        ("az://my-container/models/model.safetensors", 4096),
        ("az://my-container/models/config.json", 128),
    ]

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_basic_listing(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub(self.ENTRIES)
        results = FileStreamer().list_files(self.PREFIX)
        self.assertEqual(sorted(results), sorted(self.ENTRIES))

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_is_recursive_passed(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX, is_recursive=False)
        self.assertEqual(mock_rlf.call_args.kwargs["is_recursive"], False)

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_ignore_patterns_passed(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX, ignore_patterns=["*.json"])
        self.assertEqual(mock_rlf.call_args.kwargs["ignore_patterns"], ["*.json"])

    @patch("runai_model_streamer.file_streamer.file_streamer.runai_list_files")
    def test_no_credentials_for_azure(self, mock_rlf):
        mock_rlf.side_effect = _list_files_stub([])
        FileStreamer().list_files(self.PREFIX)
        # credentials are streamer-scoped now; list_files never passes a per-call params/credentials arg
        self.assertNotIn("params", mock_rlf.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
