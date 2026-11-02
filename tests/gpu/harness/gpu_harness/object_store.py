# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The object storage that the environment provides (the `seaweedfs` service of tests/gpu/compose.yaml, started with the
`object_storage` profile), for a test that needs objects: to put some there, to read them back, and to look at what the
streamer left. The streamer reaches the same service by itself, through the environment (AWS_ENDPOINT_URL and the keys),
which is also where this takes its settings from:

    AWS_ENDPOINT_URL         the S3 endpoint of the service (required)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY      its keys
    GPU_OBJECT_BUCKET        the bucket that the service creates (default test-bucket)

Needs boto3, which the object_storage suite installs; it is imported when the first request is made, so that importing
this does not need it.
"""

import os
import time
import uuid

DEFAULT_BUCKET = "test-bucket"


class ObjectStoreNotAvailable(Exception):
    """The environment has no object storage: say how to get one, instead of a connection error from deep inside boto3."""


class ObjectStore:
    def __init__(self, endpoint, bucket=DEFAULT_BUCKET, access_key=None, secret_key=None, region="us-east-1"):
        self.endpoint = endpoint
        self.bucket = bucket
        self._settings = {"endpoint_url": endpoint, "aws_access_key_id": access_key, "aws_secret_access_key": secret_key,
                          "region_name": region}
        self._client = None

    @classmethod
    def from_environment(cls):
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if not endpoint:
            raise ObjectStoreNotAvailable(
                "AWS_ENDPOINT_URL is not set: this environment has no object storage. Start the service with the "
                "object_storage profile of tests/gpu/compose.yaml (tests/gpu/run.sh --suites object_storage does)")
        return cls(endpoint, os.environ.get("GPU_OBJECT_BUCKET", DEFAULT_BUCKET), os.environ.get("AWS_ACCESS_KEY_ID"),
                   os.environ.get("AWS_SECRET_ACCESS_KEY"))

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("s3", **self._settings)
        return self._client

    def wait_until_ready(self, seconds=60):
        """Wait until the service answers for the bucket: a container that is up is not a service that is ready."""
        deadline = time.monotonic() + seconds
        last = None
        while time.monotonic() < deadline:
            try:
                self.client.head_bucket(Bucket=self.bucket)
                return
            except Exception as error:  # not up yet: a refused connection, a bucket that is not created yet, ...
                last = error
                time.sleep(1)
        raise ObjectStoreNotAvailable("%s (bucket %s) was not ready in %d s: %s" % (self.endpoint, self.bucket, seconds, last))

    def uri(self, key):
        return "s3://%s/%s" % (self.bucket, key)

    def put_bytes(self, key, data):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def put_file(self, key, path):
        self.client.upload_file(path, self.bucket, key)

    def get_bytes(self, key):
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def keys(self, prefix=""):
        found = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            found.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(found)

    def delete_prefix(self, prefix):
        for key in self.keys(prefix):
            self.client.delete_object(Bucket=self.bucket, Key=key)

    def new_prefix(self, name="test"):
        """A prefix of a test's own, so that tests do not see each other's objects and a run leaves nothing that the
        next one can mistake for its own."""
        return "gpu-tests/%s-%s/" % (name, uuid.uuid4().hex[:12])
