# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import glob
import shutil
import fnmatch
from pathlib import Path
from typing import List, Optional, Iterator, Any
from urllib.parse import urlparse
import logging
import torch

# --- Import REAL implementations ---
from runai_model_streamer import (
    SafetensorsStreamer as OriginalSafetensorsStreamer,
    list_safetensors as original_list_safetensors,
)
from runai_model_streamer.s3_utils.s3_utils import (
    is_s3_path,
    is_gs_path,
    S3Credentials,
    filter_allow,
    filter_ignore,
    removeprefix,
)

logger = logging.getLogger(__name__)

class StreamerPatcher:
    """
    Generates mock objects for runai_model_streamer that redirect
    S3/GS paths to a local filesystem path.
    """

    def __init__(self, local_bucket_path: str):
        # The bucket part in the uri would be replaced by the local path
        # e.g. gs://mybucket/path/to/file  would be replaced by local_path/path/to/file
        self.local_path = local_bucket_path 
        logger.debug(f"[Patcher] Initialized with local path: {self.local_path} instead of object storage bucket")

    def convert_remote_path_to_local_path(self, path: str) -> str:
        """Helper to convert a remote path to a local path."""        
        # The bucket part in the uri would be replaced by the local path
        # e.g. gs://mybucket/path/to/file  would be replaced by local_path/path/to/file
        if not self.is_remote_path(path):
            return path

        # 1. Parse the full path as a URL
        parsed_uri = urlparse(path)

        # For "s3://mybucket/path/to/object":
        # parsed_uri.path will be "/path/to/object"

        # We just want the path. We strip the leading "/"
        # so os.path.join doesn't treat it as an absolute path.
        object_path = parsed_uri.path.lstrip('/')

        converted_path = os.path.join(self.local_path, object_path)
        logger.debug(f"[RunAI Streamer][SHIM] Converted '{path}' to '{converted_path}'")
        return converted_path

    def convert_local_path_to_mocked_remote_path(self, local_path_result: str, original_remote_path: str) -> str:
        """
        Converts a local path result (from list_safetensors) back to a mocked remote URI.

        It takes the path segment relative to self.local_path and prepends the
        original scheme (s3://bucket/) to it.
        """
        # 1. Determine the path segment relative to the patcher's local_path
        if not local_path_result.startswith(self.local_path):
            logger.warning(f"Local path result '{local_path_result}' does not start with '{self.local_path}'. Returning original.")
            return local_path_result

        # Get the path component that represents the object storage key
        relative_path = os.path.relpath(local_path_result, self.local_path)

        # 2. Extract the scheme and netloc (bucket name) from the original remote path
        parsed_original = urlparse(original_remote_path)
        scheme_and_netloc = f"{parsed_original.scheme}://{parsed_original.netloc}"

        # 3. Construct the final mocked remote URI
        # Ensure forward slashes for URI key and a single path separator
        remote_uri = f"{scheme_and_netloc}/{relative_path.replace(os.path.sep, '/')}"

        return remote_uri


    def is_remote_path(self, path: str) -> bool:            
        """Helper to check if a path is S3 or GS."""
        return is_s3_path(path) or is_gs_path(path)
        

    # === Shim for list_safetensors ===
    def shim_list_safetensors(self, path: str, s3_credentials: Optional[S3Credentials] = None) -> List[str]:
        logger.debug(f"[RunAI Streamer][SHIM] list_safetensors is called with path: {path}")
        rewritten_path = self.convert_remote_path_to_local_path(path)
        local_paths = original_list_safetensors(rewritten_path, s3_credentials)
        remote_paths = [
            self.convert_local_path_to_mocked_remote_path(p, path)
            for p in local_paths
        ]

        return remote_paths


    # === Shim for pull_files ===
    def shim_pull_files(self, model_path: str,
                          dst: str,
                          allow_pattern: Optional[List[str]] = None,
                          ignore_pattern: Optional[List[str]] = None,
                          s3_credentials: Optional[S3Credentials] = None) -> None:
        logger.debug(f"[RunAI Streamer][SHIM] pull_files is called with path: {model_path}")
        
        # 1. Check if it's a path we're mocking (S3 or GS)
        if not self.is_remote_path(model_path):
            # Match the original's behavior for non-remote paths
            raise NotImplementedError("pull files is not implemented for file system paths")
        
        source_dir = self.convert_remote_path_to_local_path(model_path)
        logger.debug(f"[RunAI Streamer][SHIM] pull_files: Simulating download from '{model_path}' (using local: '{source_dir}') to '{dst}'")

        # 2. Simulate list_objects_v2 by walking the local dir
        all_local_files_relative = []
        for root, _, files in os.walk(source_dir, topdown=True):           
            for file in files:
                full_path = os.path.join(root, file)
                # Get the path relative to the model root
                relative_path = os.path.relpath(full_path, source_dir)
                # S3/GS keys use forward slashes
                all_local_files_relative.append(relative_path.replace(os.path.sep, '/'))

        # 3. --- REUSE ORIGINAL FILTERING LOGIC ---
        # This is the exact filtering logic from both S3 and GCS `list_files`
        
        logger.debug(f"[RunAI Streamer][SHIM] pull_files: Filtering {len(all_local_files_relative)} local files...")
        
        # paths = _filter_ignore(paths, ["*/"])
        filtered_paths = filter_ignore(all_local_files_relative, ["*/"])
        
        # if allow_pattern is not None: paths = _filter_allow(paths, allow_pattern)
        if allow_pattern:
            filtered_paths = filter_allow(filtered_paths, allow_pattern)
            
        # if ignore_pattern is not None: paths = _filter_ignore(paths, ignore_pattern)
        if ignore_pattern:
            filtered_paths = filter_ignore(filtered_paths, ignore_pattern)

        if not filtered_paths:
            logger.debug("[RunAI Streamer][SHIM] pull_files: No files matched patterns.")
            return

        logger.debug(f"[RunAI Streamer][SHIM] pull_files: Matched {len(filtered_paths)} files.")

        # 4. Simulate the copy loop from `pull_files`
        # The `base_dir` in S3/GS-land is the prefix. In our case, the
        # `base_dir` is effectively empty string, because our
        # `filtered_paths` are already relative to the root.
        
        base_dir = "" # Our local paths are already relative
        
        for file in filtered_paths:
            # S3/GS logic: destination_file = os.path.join(dst, removeprefix(file, base_dir).lstrip("/"))
            relative_file = removeprefix(file, base_dir).lstrip("/")
            
            # Source path on local disk
            source_file_path = os.path.join(source_dir, relative_file.replace('/', os.path.sep))
            
            # Destination path
            destination_file = os.path.join(dst, relative_file.replace('/', os.path.sep))

            # Create parent directory in destination
            local_dir = Path(destination_file).parent
            os.makedirs(local_dir, exist_ok=True)
            
            # Copy the file
            logger.debug(f"[RunAI Streamer][SHIM] pull_files: Copying {source_file_path} to {destination_file}")
            shutil.copy(source_file_path, destination_file)

    # === Shim for SafetensorsStreamer ===
       
    def create_mock_streamer(self, *args, **kwargs):
        return self.MockSafetensorsStreamer(self)

    class MockSafetensorsStreamer:
        def __init__(self, patcher: StreamerPatcher):
            self.patcher = patcher
            self.original_streamer = OriginalSafetensorsStreamer()
            self.files_to_tensors_metadata = {}

        def __enter__(self) -> "MockSafetensorsStreamer":
            self.original_streamer.__enter__()
            return self

        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
            return self.original_streamer.__exit__(exc_type, exc_value, traceback)

        def stream_file(self, path: str, s3_credentials: Optional[S3Credentials] = None,
                          device: Optional[str] = "cpu", is_distributed: bool = False) -> None:
            logger.debug(f"[RunAI Streamer][SHIM] stream_file is called with path: {path}")
            self.files_to_tensors_metadata = {}
            rewritten_path = self.patcher.convert_remote_path_to_local_path(path)
            res = self.original_streamer.stream_file(
                rewritten_path, s3_credentials, device, is_distributed
            )
            self.files_to_tensors_metadata = self.original_streamer.files_to_tensors_metadata
            return res

        def stream_files(self, paths: List[str], s3_credentials: Optional[S3Credentials] = None,
                           device: Optional[str] = "cpu", is_distributed: bool = False) -> None:
            logger.debug(f"[RunAI Streamer][SHIM] stream_files is called")
            rewritten_paths = [self.patcher.convert_remote_path_to_local_path(p) for p in paths]
            return self.original_streamer.stream_files(
                rewritten_paths, s3_credentials, device, is_distributed
            )

        def get_tensors(self) -> Iterator[torch.tensor]:
            logger.debug(f"[RunAI Streamer][SHIM] get_tensors is called")
            return self.original_streamer.get_tensors()