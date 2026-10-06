# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


class AzureCredentials:
    """
    Azure Blob Storage credentials configuration.

    Authentication methods (checked in this order):
    1. Connection string: Set AZURE_STORAGE_CONNECTION_STRING
    2. SAS token: Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_SAS_TOKEN
    3. Storage account key: Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY
    4. DefaultAzureCredential: Set AZURE_STORAGE_ACCOUNT_NAME (uses Managed Identity, Azure CLI, etc.)

    If values are not provided explicitly, they are loaded from environment variables:
    - AZURE_STORAGE_CONNECTION_STRING
    - AZURE_STORAGE_ACCOUNT_NAME
    - AZURE_STORAGE_ACCOUNT_KEY
    - AZURE_STORAGE_SAS_TOKEN
    """

    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        sas_token: Optional[str] = None,
        connection_string: Optional[str] = None,
        endpoint_suffix: Optional[str] = None,
        credential: Optional[DefaultAzureCredential] = None
    ):
        self.connection_string = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.account_name = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        self.sas_token = sas_token or os.environ.get("AZURE_STORAGE_SAS_TOKEN")
        self.endpoint_suffix = endpoint_suffix or os.environ.get("AZURE_STORAGE_ENDPOINT_SUFFIX", "blob.core.windows.net")
        if credential is None and not self.connection_string and not self.account_key and not self.sas_token:
            credential = DefaultAzureCredential()
        self.credential = credential
        self._validate()

    def _validate(self) -> None:
        """Validates that sufficient credentials are available to create a client."""
        if not self.connection_string and not self.account_name:
            raise ValueError(
                "Azure credentials required. Set AZURE_STORAGE_CONNECTION_STRING for local testing, "
                "or AZURE_STORAGE_ACCOUNT_NAME for production with DefaultAzureCredential."
            )


def get_credentials() -> AzureCredentials:
    """
    Creates Azure credentials from environment variables.

    Returns:
        AzureCredentials object with credentials loaded from environment
        
    Raises:
        ValueError: If neither connection string nor account name is available
    """
    return AzureCredentials()
