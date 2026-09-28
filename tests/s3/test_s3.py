import json
import shutil
import tempfile
import unittest
import os
import time
import boto3
from unittest.mock import patch

from botocore.exceptions import NoCredentialsError, ClientError, EndpointConnectionError
from safetensors.torch import safe_open

from tests.cases.interface import ObjectStoreBackend
from tests.cases.testcases import compatibility_test_cases, list_files_test_cases
from tests.safetensors.generator import create_random_safetensors
from tests.safetensors.comparison import tensor_maps_are_equal
from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
    list_safetensors,
    pull_files,
)
RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR = "RUNAI_STREAMER_S3_UNSIGNED"


class SeaweedFSServer(ObjectStoreBackend):
    def __init__(self):
        self.url = os.getenv("AWS_ENDPOINT_URL")
        self.key = os.getenv("AWS_ACCESS_KEY_ID")
        self.password = os.getenv("AWS_SECRET_ACCESS_KEY")

    def wait_for_startup(self, timeout=30):
        print("Waiting for SeaweedFS server to be up and running.")
        start_time = time.time()
        s3_client = boto3.client(
            "s3",
            endpoint_url=self.url,
            aws_access_key_id=self.key,
            aws_secret_access_key=self.password
        )
        while time.time() - start_time < timeout:
            try:
                s3_client.list_buckets()
                print("SeaweedFS server is up and running.")
                return
            except (ClientError, NoCredentialsError, EndpointConnectionError):
                time.sleep(0.5)
        raise TimeoutError(f"SeaweedFS server failed to start within {timeout} seconds.")

    def upload_file(self, bucket, directory, file):
        s3_client = boto3.client(
            's3',
            endpoint_url=self.url,
            aws_access_key_id=self.key,
            aws_secret_access_key=self.password
        )
        s3_client.upload_file(file, bucket, os.path.join(directory, os.path.basename(file)))

TestS3ompatibility = compatibility_test_cases(
    backend_class = SeaweedFSServer,
    scheme = "s3",
    bucket_name = os.getenv("AWS_BUCKET")
)

TestS3ListFiles = list_files_test_cases(
    backend_class = SeaweedFSServer,
    scheme = "s3",
    bucket_name = os.getenv("AWS_BUCKET")
)


class TestS3UnsignedPublicBucket(unittest.TestCase):
    PUBLIC_BUCKET = "public-test-bucket"

    @classmethod
    def setUpClass(cls):
        cls.server = SeaweedFSServer()
        cls.server.wait_for_startup()
        cls.temp_dir = tempfile.mkdtemp()

        s3_admin = boto3.client(
            "s3",
            endpoint_url=cls.server.url,
            aws_access_key_id=cls.server.key,
            aws_secret_access_key=cls.server.password,
        )
        try:
            s3_admin.create_bucket(Bucket=cls.PUBLIC_BUCKET)
        except ClientError as e:
            if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise

        s3_admin.put_bucket_policy(
            Bucket=cls.PUBLIC_BUCKET,
            Policy=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{cls.PUBLIC_BUCKET}",
                        f"arn:aws:s3:::{cls.PUBLIC_BUCKET}/*"
                    ]
                }]
            })
        )

        cls.file_path = create_random_safetensors(cls.temp_dir)
        cls.server.upload_file(cls.PUBLIC_BUCKET, "", cls.file_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_list_safetensors_unsigned(self):
        with patch.dict(os.environ, {RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "1"}):
            result = list_safetensors(f"s3://{self.PUBLIC_BUCKET}/")
        self.assertIn(f"s3://{self.PUBLIC_BUCKET}/model.safetensors", result)

    def test_pull_files_unsigned(self):
        pull_dir = tempfile.mkdtemp()
        try:
            with patch.dict(os.environ, {RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "1"}):
                pull_files(f"s3://{self.PUBLIC_BUCKET}/", pull_dir, allow_pattern=["*.safetensors"])
            self.assertIn("model.safetensors", os.listdir(pull_dir))
        finally:
            shutil.rmtree(pull_dir)

    def test_stream_file_unsigned(self):
        our = {}
        with patch.dict(os.environ, {RUNAI_STREAMER_S3_UNSIGNED_ENV_VAR: "1"}):
            with SafetensorsStreamer() as streamer:
                streamer.stream_file(f"s3://{self.PUBLIC_BUCKET}/model.safetensors", None, "cpu")
                for name, tensor in streamer.get_tensors():
                    # clone: the yielded tensor is a VIEW into a ring buffer that is recycled
                    # once the generator advances, so a comparison after the loop must own it
                    our[name] = tensor.clone()

        their = {}
        with safe_open(self.file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                their[name] = f.get_tensor(name)

        equal, message = tensor_maps_are_equal(our, their)
        if not equal:
            self.fail(f"Tensor mismatch: {message}")


if __name__ == "__main__":
    unittest.main()
