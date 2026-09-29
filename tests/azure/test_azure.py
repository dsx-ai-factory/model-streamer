# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import time

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ServiceRequestError, ResourceNotFoundError

from tests.cases.interface import ObjectStoreBackend
from tests.cases.testcases import compatibility_test_cases, list_files_test_cases


class AzuriteServer(ObjectStoreBackend):
    """A helper class to interact with Azurite (Azure Storage emulator) test server."""
    def __init__(self):
        # Use connection string for Azurite (local emulator)
        # Azurite requires authentication - it doesn't support anonymous access
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING environment variable must be set for Azure tests")
        self.client = BlobServiceClient.from_connection_string(connection_string)

    def wait_for_startup(self, timeout=30):
        """Wait for the Azurite server to become available."""
        print("Waiting for Azurite server to be up and running.")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to list containers to check connectivity
                containers = self.client.list_containers(results_per_page=1)
                next(iter(containers), None)  # Consume one item from the iterator
                print("Azurite server is up and running.")
                return
            except (ServiceRequestError, ResourceNotFoundError, StopIteration):
                time.sleep(0.5)
        raise TimeoutError(f"Azurite server failed to start within {timeout} seconds.")

    def upload_file(self, container_name, directory, file_path):
        """Uploads a local file to the specified Azure container and directory."""
        container_client = self.client.get_container_client(container_name)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
        
        blob_name = os.path.join(directory, os.path.basename(file_path)).replace("\\", "/")
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)


TestAzureCompatibility = compatibility_test_cases(
    backend_class=AzuriteServer,
    scheme="az",
    bucket_name=os.getenv("AZURE_CONTAINER")
)

TestAzureListFiles = list_files_test_cases(
    backend_class=AzuriteServer,
    scheme="az",
    bucket_name=os.getenv("AZURE_CONTAINER")
)

if __name__ == "__main__":
    unittest.main()