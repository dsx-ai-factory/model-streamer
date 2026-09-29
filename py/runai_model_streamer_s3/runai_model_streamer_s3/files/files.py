# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Tuple
from runai_model_streamer_s3.credentials.credentials import get_credentials, S3Credentials, RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR
import fnmatch
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import posixpath


def _build_client_config() -> Optional[Config]:
    config_kwargs = {}
    if os.getenv("RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING", "1") == "0":
        config_kwargs["s3"] = {"addressing_style": "path"}
    if os.getenv(RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR, "0") == "1":
        config_kwargs["signature_version"] = UNSIGNED
    return Config(**config_kwargs) if config_kwargs else None

def _build_s3_client(credentials: Optional[S3Credentials]):
    session, _ = get_credentials(credentials)
    client_config = _build_client_config()
    if session is None:
        return boto3.client("s3", config=client_config)
    return session.client("s3", config=client_config)

def glob(path: str, allow_pattern: Optional[List[str]] = None, credentials: Optional[S3Credentials] = None) -> List[str]:
    s3 = _build_s3_client(credentials)
    if not path.endswith("/"):
        path = f"{path}/"
    bucket_name, _, keys = list_files(s3,
                                       path=path,
                                       allow_pattern=allow_pattern)
    return [f"s3://{bucket_name}/{key}" for key in keys]

def pull_files(model_path: str,
                dst: str,
                allow_pattern: Optional[List[str]] = None,
                ignore_pattern: Optional[List[str]] = None,
                credentials: Optional[S3Credentials] = None,) -> None:
    s3 = _build_s3_client(credentials)

    if not model_path.endswith("/"):
        model_path = model_path + "/"

    bucket_name, base_dir, files = list_files(s3, model_path,
                                                allow_pattern,
                                                ignore_pattern,
                                                recursive = True)
    if len(files) == 0:
        return

    for file in files:
        destination_file = _safe_destination_path(dst, base_dir, file)
        local_dir = Path(destination_file).parent
        os.makedirs(local_dir, exist_ok=True)
        s3.download_file(bucket_name, file, destination_file)

def list_files(
        s3,
        path: str,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
        recursive: bool = False
) -> Tuple[str, str, List[str]]:
    parts = removeprefix(path, 's3://').split('/')
    bucket_name = parts[0]
    
    # Reconstruct the prefix
    prefix = '/'.join(parts[1:]).rstrip('/')

    if prefix:
        # This ensures a trailing slash without double-slashing
        prefix = posixpath.join(prefix, '')

    paginator = s3.get_paginator('list_objects_v2')

    op_parameters = {
        'Bucket': bucket_name,
        'Prefix': prefix,
    }

    if not recursive:
        # delimiter='/' so list is not recursive  
        op_parameters['Delimiter'] = '/'

    paths = []
    for page in paginator.paginate(**op_parameters):
        # Contents is a list of files (no folders)
        if 'Contents' in page:
            for obj in page['Contents']:
                paths.append(obj['Key'])

    # Filter logic remains the same
    paths = _filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths

def _filter_allow(paths: List[str], patterns: List[str]) -> List[str]:
    return [
        path for path in paths if any(
            fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]

def _filter_ignore(paths: List[str], patterns: List[str]) -> List[str]:
    return [
        path for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]

def _safe_destination_path(dst: str, base_dir: str, file: str) -> str:
    prefix = base_dir if base_dir == "" or base_dir.endswith("/") else base_dir + "/"
    if not file.startswith(prefix):
        raise ValueError(f"object key {file!r} does not start with expected prefix {prefix!r}")
    relative = file[len(prefix):].lstrip("/")
    dst_real = os.path.realpath(dst)
    destination_file = os.path.realpath(os.path.join(dst_real, relative))
    if os.path.commonpath([dst_real, destination_file]) != dst_real:
        raise ValueError(f"refusing to write outside destination directory: {file!r}")
    return destination_file

def removeprefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix):]
    return s