# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Tuple
from runai_model_streamer_azure.credentials.credentials import AzureCredentials, get_credentials

import fnmatch
import os
import posixpath
from pathlib import Path

from azure.storage.blob import BlobServiceClient, BlobProperties


# Application ID for telemetry (prepended to User-Agent)
# Reference: https://azure.github.io/azure-sdk/general_azurecore.html#user-agent-format
_USER_AGENT = "azpartner-runai"


def _create_client(credentials: Optional[AzureCredentials] = None) -> BlobServiceClient:
    """
    Creates an Azure BlobServiceClient.

    Authentication priority:
    1. Connection string (AZURE_STORAGE_CONNECTION_STRING) - for local testing with Azurite
    2. SAS token (AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_SAS_TOKEN)
    3. Storage account key (AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY)
    4. DefaultAzureCredential with account URL - for production

    Args:
        credentials: Optional AzureCredentials object

    Returns:
        BlobServiceClient instance
    """
    if credentials is None:
        credentials = get_credentials()

    # Use connection string if available (for Azurite/local testing)
    if credentials.connection_string:
        return BlobServiceClient.from_connection_string(
            credentials.connection_string,
            user_agent=_USER_AGENT
        )

    # Use account name + SAS token if available
    if credentials.sas_token and credentials.account_name:
        token = credentials.sas_token.lstrip("?")
        account_url = f"https://{credentials.account_name}.{credentials.endpoint_suffix}"
        return BlobServiceClient(
            account_url=account_url,
            credential=token,
            user_agent=_USER_AGENT
        )

    # Use account name + account key if available (StorageSharedKeyCredential)
    if credentials.account_key and credentials.account_name:
        account_url = f"https://{credentials.account_name}.{credentials.endpoint_suffix}"
        return BlobServiceClient(
            account_url=account_url,
            credential=credentials.account_key,
            user_agent=_USER_AGENT
        )

    # Use account name + DefaultAzureCredential (for production)
    account_url = f"https://{credentials.account_name}.{credentials.endpoint_suffix}"
    return BlobServiceClient(
        account_url=account_url,
        credential=credentials.credential,
        user_agent=_USER_AGENT
    )


def glob(path: str, allow_pattern: Optional[List[str]] = None, credentials: Optional[AzureCredentials] = None) -> List[str]:
    """
    List files in Azure Blob Storage matching the given pattern.
    
    Args:
        path: Azure blob path in format "az://container/prefix"
        allow_pattern: Optional list of glob patterns to include
        credentials: Optional AzureCredentials object
        
    Returns:
        List of full Azure blob paths
    """
    client = _create_client(credentials)

    if not path.endswith("/"):
        path = f"{path}/"
    
    # glob is non-recursive - only list files in the given directory
    container_name, _, keys = list_files(client, path, allow_pattern, recursive=False)
    return [f"az://{container_name}/{key}" for key in keys]


def pull_files(
    model_path: str,
    dst: str,
    allow_pattern: Optional[List[str]] = None,
    ignore_pattern: Optional[List[str]] = None,
    credentials: Optional[AzureCredentials] = None
) -> None:
    """
    Download files from Azure Blob Storage to local directory.
    
    Args:
        model_path: Azure blob path in format "az://container/prefix"
        dst: Local destination directory
        allow_pattern: Optional list of glob patterns to include
        ignore_pattern: Optional list of glob patterns to exclude
        credentials: Optional AzureCredentials object
    """
    client = _create_client(credentials)

    if not model_path.endswith("/"):
        model_path = model_path + "/"

    # pull_files is recursive - download all files including subdirectories
    container_name, base_dir, files = list_files(
        client, model_path, allow_pattern, ignore_pattern, recursive=True
    )
    
    if len(files) == 0:
        return

    container_client = client.get_container_client(container_name)

    for file in files:
        destination_file = _safe_destination_path(dst, base_dir, file)
        local_dir = Path(destination_file).parent
        os.makedirs(local_dir, exist_ok=True)
        
        blob_client = container_client.get_blob_client(file)
        with open(destination_file, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())


def list_files(
    client: BlobServiceClient,
    path: str,
    allow_pattern: Optional[List[str]] = None,
    ignore_pattern: Optional[List[str]] = None,
    recursive: bool = False
) -> Tuple[str, str, List[str]]:
    """
    List files in Azure Blob Storage at the given path.
    
    Args:
        client: BlobServiceClient instance
        path: Azure blob path
        allow_pattern: Optional list of glob patterns to include
        ignore_pattern: Optional list of glob patterns to exclude
        recursive: If True, list files in subdirectories. If False, only list files directly in the path.

    Returns:
        Tuple of (container_name, prefix, list_of_blob_names)
    """
    # Parse az://container/prefix format
    path = removeprefix(path, 'az://')
    parts = path.split('/', 1)
    container_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # Reconstruct the prefix
    prefix = prefix.rstrip('/')
    
    if prefix:
        # This ensures a trailing slash without double-slashing
        prefix = posixpath.join(prefix, '')

    container_client = client.get_container_client(container_name)
    
    paths = []
    if recursive:
        # Recursive: list all blobs under the prefix
        blob_items = container_client.list_blobs(name_starts_with=prefix, include=["metadata"])
        for item in blob_items:
            if hasattr(item, 'name') and not _is_adls_directory(item):
                paths.append(item.name)
    else:
        # Non-recursive: use walk_blobs with delimiter to only get blobs at this level
        # This is more efficient as Azure service handles the filtering
        blob_items = container_client.walk_blobs(name_starts_with=prefix, delimiter='/', include=["metadata"])
        for item in blob_items:
            # walk_blobs returns BlobProperties for blobs and BlobPrefix for directories
            # We only want blobs (files), not prefixes (directories)

            if hasattr(item, 'name') and not item.name.endswith('/') and not _is_adls_directory(item):
                paths.append(item.name)

    # Filter out directories (blobs ending with /)
    paths = _filter_ignore(paths, ["*/"])
    
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return container_name, prefix, paths


def _filter_allow(paths: List[str], patterns: List[str]) -> List[str]:
    return [
        path for path in paths if any(
            fnmatch.fnmatch(path, pattern) for pattern in patterns
        )
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

def _is_adls_directory(blob: BlobProperties) -> bool:
    # When listing against ADLS, we might hit directory stubs (hdi_isfolder=true) that don't have a trailing slash
    # So, we use metadata to filter them out
    return (
        blob.size == 0
        and blob.metadata is not None
        and (blob.metadata.get("hdi_isfolder") == "true" 
            or blob.metadata.get("Hdi_isfolder") == "true" # Sometimes, the service returns this metadata key capitalized 
            )
    )
