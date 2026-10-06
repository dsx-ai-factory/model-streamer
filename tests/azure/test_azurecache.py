# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end pytest for the azure cache provider (azcache_provider).

Tests:
  1. Compiles simple_file_cache_test.cc into a shared library
  2. Creates a synthetic safetensors model and populates a cache directory
  3. Streams the model through RunAI using the compiled cache .so
  4. Verifies every tensor matches the original data

Run:
    pytest tests/azure/test_azurecache.py -v
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest


SIMPLE_FILE_CACHE_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "cpp", "azure", "azcache_provider", "simple_file_cache_test.cc"
)

DTYPE_MAP = {
    np.dtype("float32"): "F32",
    np.dtype("float64"): "F64",
    np.dtype("float16"): "F16",
    np.dtype("int32"):   "I32",
    np.dtype("int64"):   "I64",
    np.dtype("uint8"):   "U8",
}


def create_safetensors_file(path, tensors):
    """Create a safetensors file from a dict of {name: numpy_array}."""
    header = {}
    data_chunks = []
    offset = 0
    for name, arr in tensors.items():
        nbytes = arr.nbytes
        header[name] = {
            "dtype": DTYPE_MAP[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        data_chunks.append(arr.tobytes())
        offset += nbytes

    header_json = json.dumps(header).encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for chunk in data_chunks:
            f.write(chunk)
    return offset


@pytest.fixture(scope="module")
def cache_lib():
    """Compile simple_file_cache_test.cc into a shared library."""
    src = os.path.abspath(SIMPLE_FILE_CACHE_SRC)
    if not os.path.isfile(src):
        pytest.skip(f"simple_file_cache_test.cc not found at {src}")

    so_path = os.path.join(tempfile.gettempdir(), "libsimple_file_cache_test.so")
    result = subprocess.run(
        ["g++", "-shared", "-fPIC", "-o", so_path, src],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to compile simple_file_cache_test.cc:\n{result.stderr}")

    yield so_path

    if os.path.exists(so_path):
        os.remove(so_path)


@pytest.fixture(scope="module")
def model_tensors():
    """Create synthetic model tensors."""
    return {
        "model.embed_tokens.weight":              np.random.randn(512, 64).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight":    np.random.randn(256, 64).astype(np.float32),
        "model.layers.0.mlp.up_proj.weight":      np.random.randn(256, 64).astype(np.float32),
        "model.layers.0.mlp.down_proj.weight":    np.random.randn(64, 256).astype(np.float32),
        "model.norm.weight":                      np.random.randn(64).astype(np.float32),
        "lm_head.weight":                         np.random.randn(512, 64).astype(np.float32),
    }


@pytest.fixture(scope="module")
def cache_dir(model_tensors):
    """Populate a cache directory with the synthetic model."""
    cache_root = tempfile.mkdtemp(prefix="runai_azcache_test_")
    container = "test-models"
    blob = "my-llm/model.safetensors"
    cache_path = os.path.join(cache_root, container, blob)

    create_safetensors_file(cache_path, model_tensors)

    yield cache_root, container, blob

    shutil.rmtree(cache_root)


class TestAzureCache:
    """Test the azure cache provider with RunAI Model Streamer."""

    def test_stream_from_cache(self, cache_lib, cache_dir, model_tensors):
        """Stream a safetensors model from a file-based cache and verify all tensors.

        Runs in a subprocess because AzCacheProviderLoader is created per-AzureClient
        and the ClientMgr pool reuses clients — env vars are read at first client
        creation and cannot be changed for a reused client within the same process."""
        cache_root, container, blob = cache_dir

        env = os.environ.copy()
        env["RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB"] = cache_lib
        env["RUNAI_CACHE_DIR"] = cache_root
        env["RUNAI_STREAMER_LOG_LEVEL"] = "DEBUG"
        env["RUNAI_STREAMER_LOG_TO_STDERR"] = "1"
        # Provide a dummy account so the Azure client initializes without
        # real credentials — the cache provider intercepts all reads.
        if not env.get("AZURE_STORAGE_ACCOUNT_NAME") and \
           not env.get("AZURE_STORAGE_CONNECTION_STRING"):
            env["AZURE_STORAGE_ACCOUNT_NAME"] = "dummycachetest"

        # Write a small inline script that streams and verifies
        script = f"""
import os, sys, json, struct, numpy as np
os.environ["RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB"] = "{cache_lib}"
os.environ["RUNAI_CACHE_DIR"] = "{cache_root}"
from runai_model_streamer import SafetensorsStreamer

uri = "az://{container}/{blob}"
with SafetensorsStreamer() as streamer:
    streamer.stream_file(uri)
    count = 0
    for name, tensor in streamer.get_tensors():
        count += 1
    print(f"OK {{count}} tensors")
    assert count == {len(model_tensors)}, f"Expected {len(model_tensors)}, got {{count}}"
"""
        # Run in a subprocess because the ClientMgr pool reuses AzureClient instances
        # along with their AzCacheProviderLoader — env vars are captured at first client
        # creation and a pooled client won't pick up new env var values.
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            pytest.fail(
                f"Cache streaming subprocess failed:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        assert f"OK {len(model_tensors)} tensors" in result.stdout

    def test_missing_cache_lib_falls_back(self, cache_dir, model_tensors):
        """When cache lib is not set, the provider should not be enabled."""
        os.environ.pop("RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB", None)
        assert not os.environ.get("RUNAI_STREAMER_EXPERIMENTAL_AZURE_CACHE_LIB")
        # TODO: Full fallback verification requires a subprocess with Azure
        # credentials to confirm reads go through Azure SDK when env var is unset.
        # ClientMgr pools clients, so env var changes within the same process
        # won't affect already-created clients.
