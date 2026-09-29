# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import time

from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from google.api_core import exceptions as gcs_exceptions

from tests.cases.interface import ObjectStoreBackend
from tests.cases.testcases import compatibility_test_cases, list_files_test_cases

class FakeGCSServer(ObjectStoreBackend):
    """A helper class to interact with a GCS-compatible test server."""
    def __init__(self):
        self.url = os.getenv("CLOUD_STORAGE_EMULATOR_ENDPOINT")
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        # Use anonymous client as authentication is implicit
        self.client = storage.Client(
            client_options={'api_endpoint': self.url},
            project=project,
            credentials=AnonymousCredentials(),
        )

    def wait_for_startup(self, timeout=30):
        """Wait for the fake GCS server to become available."""
        print("Waiting for FakeGCS server to be up and running.")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Any simple operation to check connectivity will do.
                list(self.client.list_buckets(max_results=1))
                print("FakeGCS server is up and running.")
                return
            except gcs_exceptions.ServiceUnavailable:
                time.sleep(0.5)
        raise TimeoutError(f"FakeGCS server failed to start within {timeout} seconds.")

    def upload_file(self, bucket_name, directory, file_path):
        """Uploads a local file to the specified GCS bucket and directory."""
        bucket = self.client.bucket(bucket_name)
        blob_name = os.path.join(directory, os.path.basename(file_path)).replace("\\", "/")
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)


TestGCSCompatibility = compatibility_test_cases(
    backend_class = FakeGCSServer,
    scheme = "gs",
    bucket_name = os.getenv("GCS_BUCKET")
)

TestGCSListFiles = list_files_test_cases(
    backend_class = FakeGCSServer,
    scheme = "gs",
    bucket_name = os.getenv("GCS_BUCKET")
)

if __name__ == "__main__":
    unittest.main()
