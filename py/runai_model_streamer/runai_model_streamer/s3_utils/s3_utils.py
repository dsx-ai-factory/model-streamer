# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List
import re
import os
import importlib
import fnmatch

GCS_PROTOCOL_PREFIX = "gs://"
S3_PROTOCOL_PREFIX = "s3://"
AZURE_PROTOCOL_PREFIX = "az://"
DEFAULT_GCS_ENDPOINT_URL = "https://storage.googleapis.com"
AWS_ENDPOINT_URL_ENV = "AWS_ENDPOINT_URL"
AWS_EC2_METADATA_DISABLED_ENV = "AWS_EC2_METADATA_DISABLED"
DEFAULT_AWS_EC2_METADATA_DISABLED = "true"

def get_s3_credentials_module():
    s3_module_name = "runai_model_streamer_s3"
    s3_credentials_module_name = "runai_model_streamer_s3.credentials.credentials"
    
    return get_module(s3_module_name, s3_credentials_module_name)

def get_s3_files_module():
    s3_module_name = "runai_model_streamer_s3"
    s3_files_module_name = "runai_model_streamer_s3.files.files"
    
    return get_module(s3_module_name, s3_files_module_name)

def get_gcs_files_module():
    gcs_module_name = "runai_model_streamer_gcs"
    gcs_files_module_name = "runai_model_streamer_gcs.files.files"

    return get_module(gcs_module_name, gcs_files_module_name)

def get_azure_files_module():
    azure_module_name = "runai_model_streamer_azure"
    azure_files_module_name = "runai_model_streamer_azure.files.files"

    return get_module(azure_module_name, azure_files_module_name)

def get_module(main_module: str, module_name: str):
    # Check if the main module exists first
    if importlib.util.find_spec(main_module) is None:
        return None

    # Now check if the credentials module exists
    if importlib.util.find_spec(module_name) is None:
        return None

    # Import and return the credentials module
    return importlib.import_module(module_name)

class S3Credentials:
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.region_name = region_name
        self.endpoint = endpoint

def is_s3_path(path: str) -> bool:
    """
    Checks if the given string is an S3 path.

    :param path: The string to check.
    :return: True if it's an S3 path, False otherwise.
    """
    return path.startswith(S3_PROTOCOL_PREFIX)

def is_gs_path(path: str) -> bool:
    """
    Checks if the given string is a GCS path.

    :param path: The string to check.
    :return: True if it's an GCS path, False otherwise.
    """
    return path.startswith(GCS_PROTOCOL_PREFIX)

def is_azure_path(path: str) -> bool:
    """
    Checks if the given string is an Azure path.

    :param path: The string to check.
    :return: True if it's an Azure path, False otherwise.
    """
    return path.startswith(AZURE_PROTOCOL_PREFIX)

def s3_glob(path: str, allow_pattern: Optional[List[str]] = None, s3_credentials : Optional[S3Credentials] = None) -> List[str]:
    """
    Glob for S3 paths.

    :param path: The S3 path to glob.
    :param allow_pattern: Optional list of patterns to allow.
    :param s3_credentials: Optional S3 credentials.
    :return: List of matching S3 paths.
    """
    s3_files_module = get_s3_files_module()
    if s3_files_module is None:
        raise ImportError("S3 files module not found. Please install the required package.")
    return s3_files_module.glob(path, allow_pattern, s3_credentials)

def s3_pull_files(model_path: str,
                dst: str,
                allow_pattern: Optional[List[str]] = None,
                ignore_pattern: Optional[List[str]] = None,
                s3_credentials : Optional[S3Credentials] = None,) -> None:
    s3_files_module = get_s3_files_module()
    if s3_files_module is None:
        raise ImportError("S3 files module not found. Please install the required package.")
    return s3_files_module.pull_files(model_path, dst, allow_pattern, ignore_pattern, s3_credentials)

def gcs_glob(path: str, allow_pattern: Optional[List[str]] = None) -> List[str]:
    """
    Glob for GCS paths.

    :param path: The GCS path to glob.
    :param allow_pattern: Optional list of patterns to allow.
    :return: List of matching GCS paths.
    """
    gcs_files_module = get_gcs_files_module()
    if gcs_files_module is None:
        raise ImportError("GCS files module not found. Please install the required package.")
    return gcs_files_module.glob(path, allow_pattern)

def gcs_pull_files(model_path: str,
                dst: str,
                allow_pattern: Optional[List[str]] = None,
                ignore_pattern: Optional[List[str]] = None,
                s3_credentials : Optional[S3Credentials] = None,) -> None:
    gcs_files_module = get_gcs_files_module()
    if gcs_files_module is None:
        raise ImportError("GCS files module not found. Please install the required package.")
    return gcs_files_module.pull_files(model_path, dst, allow_pattern, ignore_pattern)

def azure_glob(path: str, allow_pattern: Optional[List[str]] = None) -> List[str]:
    """
    Glob for Azure paths.

    :param path: The Azure path to glob.
    :param allow_pattern: Optional list of patterns to allow.
    :return: List of matching Azure paths.
    """
    azure_files_module = get_azure_files_module()
    if azure_files_module is None:
        raise ImportError("Azure files module not found. Please install the required package.")
    return azure_files_module.glob(path, allow_pattern)

def azure_pull_files(model_path: str,
                dst: str,
                allow_pattern: Optional[List[str]] = None,
                ignore_pattern: Optional[List[str]] = None) -> None:
    azure_files_module = get_azure_files_module()
    if azure_files_module is None:
        raise ImportError("Azure files module not found. Please install the required package.")
    return azure_files_module.pull_files(model_path, dst, allow_pattern, ignore_pattern)

def filter_allow(paths: List[str], patterns: List[str]) -> List[str]:
    return [
        path for path in paths if any(
            fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]

def filter_ignore(paths: List[str], patterns: List[str]) -> List[str]:
    return [
        path for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]

def removeprefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix):]
    return s
