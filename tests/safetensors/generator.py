# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import string
import random
from safetensors.torch import save_file

# --- Constants ---
MIN_NUM_FILES, MAX_NUM_FILES = 1, 10
MIN_NUM_TENSORS, MAX_NUM_TENSORS = 1, 20
MAX_NUM_TENSORS_SMALL = 5
MIN_TENSOR_NAME_LEN, MAX_TENSOR_NAME_LEN = 5, 20

def _can_safetensors_save(dtype):
    """
    Check if the current safetensors version can save this dtype.

    This is a capability-probing function that tests whether a dtype can be used.
    It catches all exceptions broadly because:
    - torch.zeros() may raise RuntimeError for experimental types that exist as
      attributes but don't support tensor creation
    - save_file() may raise various exceptions for unsupported types
    - We want to gracefully handle any failure and return False
    """
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as f:
            test_tensor = torch.zeros((1,), dtype=dtype)
            save_file({"test": test_tensor}, f.name)
        return True
    except Exception:
        # Broadly catch all exceptions since this is capability probing.
        # Any failure means "this dtype is not usable", which is what we need to know.
        return False

# All types our streamer supports (for reading files)
TYPE_TO_STR = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
    torch.uint8: "U8", torch.bool: "BOOL",
    torch.complex64: "C64",
}

# Add unsigned types if available (PyTorch >= 2.3.0)
if hasattr(torch, "uint16"):
    TYPE_TO_STR[torch.uint16] = "U16"
if hasattr(torch, "uint32"):
    TYPE_TO_STR[torch.uint32] = "U32"
if hasattr(torch, "uint64"):
    TYPE_TO_STR[torch.uint64] = "U64"

# Add experimental float8 types if available (PyTorch >= 2.1)
if hasattr(torch, "float8_e4m3fn"):
    TYPE_TO_STR[torch.float8_e4m3fn] = "F8_E4M3"
if hasattr(torch, "float8_e5m2"):
    TYPE_TO_STR[torch.float8_e5m2] = "F8_E5M2"
if hasattr(torch, "float8_e8m0fnu"):
    TYPE_TO_STR[torch.float8_e8m0fnu] = "F8_E8M0"

# Add experimental float4 types if available (future PyTorch)
if hasattr(torch, "float4_e2m1fn_x2"):
    TYPE_TO_STR[torch.float4_e2m1fn_x2] = "F4"

# Lazy initialization to avoid disk I/O on every import
_SAFETENSORS_SUPPORTED_TYPES = None

def get_safetensors_supported_types():
    """
    Get types supported by both PyTorch and safetensors.

    Lazily initializes on first call to avoid disk I/O during module import.
    This is important because _can_safetensors_save() creates temporary files
    for each dtype, and the test suite imports this module multiple times.
    """
    global _SAFETENSORS_SUPPORTED_TYPES
    if _SAFETENSORS_SUPPORTED_TYPES is None:
        _SAFETENSORS_SUPPORTED_TYPES = {}
        for dtype, dtype_str in TYPE_TO_STR.items():
            if _can_safetensors_save(dtype):
                _SAFETENSORS_SUPPORTED_TYPES[dtype] = dtype_str
    return _SAFETENSORS_SUPPORTED_TYPES

def get_dtype_str(dtype):
    """
    Convert a PyTorch dtype to its safetensors string representation.

    Raises ValueError if dtype is not in TYPE_TO_STR, indicating a bug in type detection.
    """
    if dtype in TYPE_TO_STR:
        return TYPE_TO_STR[dtype]

    # If we reach here, it means a dtype was used that wasn't properly registered
    # This indicates a bug in the type detection logic
    raise ValueError(
        f"Unsupported dtype {dtype}. This dtype should have been added to TYPE_TO_STR "
        f"or filtered out by get_safetensors_supported_types(). This is a bug in generator.py."
    )

def random_name():
    characters = string.ascii_letters + string.digits + "_"
    return "tensor_" + "".join(random.choice(characters) for _ in range(random.randint(5, 15)))

def generate_random_data(min_tensors, max_tensors):
    """
    Generate random tensor data using all types supported by both PyTorch and safetensors.

    get_safetensors_supported_types() returns only the types that:
    1. Are available in the current PyTorch version
    2. Can be saved by the current safetensors version
    """
    all_available = list(get_safetensors_supported_types().keys())

    if not all_available:
        raise RuntimeError(
            "No supported types detected! This should never happen. "
            "Check that PyTorch and safetensors are properly installed."
        )

    tensors = {}
    for _ in range(random.randint(min_tensors, max_tensors)):
        dtype = random.choice(all_available)
        shape = tuple(random.randint(1, 10) for _ in range(random.randint(1, 4)))

        # Sometimes a ZERO ELEMENT tensor (one dimension is 0). It is a real safetensors entry - the
        # reference implementation writes it with data_offsets [x, x] and yields it on read - and it is
        # where a zero sized range comes from in production. Every layer has to carry it: one response is
        # still owed for it, its shape must survive, and the distributed path must not drop it.
        if random.randint(1, 6) == 1:
            dims = list(shape)
            dims[random.randrange(len(dims))] = 0
            shape = tuple(dims)

        try:
            if dtype == torch.bool:
                t = torch.rand(shape) > 0.5
            elif dtype.is_complex:
                # Generate complex tensors with random real and imaginary parts
                real = torch.rand(shape, device="cpu") - 0.5
                imag = torch.rand(shape, device="cpu") - 0.5
                t = torch.complex(real, imag).to(dtype)
            elif dtype.is_floating_point:
                # For 8-bit float types (float8_e4m3fn, float8_e5m2, etc.), torch.rand().to(dtype)
                # is not supported. Instead, create random uint8 data and view as the target dtype.
                if dtype.itemsize == 1:  # 8-bit float types
                    # Create random uint8 data and view as the float8 dtype
                    t = torch.randint(0, 256, shape, dtype=torch.uint8, device="cpu").view(dtype)
                else:
                    # For standard float types, shift range to [-0.5, 0.5) to test the Sign Bit
                    t = (torch.rand(shape, device="cpu") - 0.5).to(dtype)
            else:
                t = torch.randint(0, 100, shape, dtype=dtype)
        except Exception:
            # If tensor generation fails for any reason (e.g., experimental dtype doesn't
            # support certain operations), fall back to zeros. This can happen if a dtype
            # passes _can_safetensors_save() but doesn't support rand/randint operations.
            t = torch.zeros(shape, dtype=dtype)

        tensors[random_name()] = t
    return tensors

def create_random_safetensors(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, "model.safetensors")
    tensors = generate_random_data(MIN_NUM_TENSORS, MAX_NUM_TENSORS)
    # Use official library to ensure compatibility
    save_file(tensors, path)
    return path

def create_sized_safetensors(dir_path, filename="big.safetensors", num_tensors=8, elements_per_tensor=1_000_000):
    """
    Create a safetensors file with a controllable, non-trivial payload (default ~32 MB:
    8 tensors x 1M float32). Large enough that streaming is not instantaneous, so tests
    can tear the streamer down while reads are still in flight.

    Every tensor gets a DISTINCT, position-sensitive value range (tensor i holds
    [i*N, (i+1)*N)) so an exact element-wise comparison actually validates the data part:
    it catches wrong offsets within a tensor and cross-tensor swaps, which identical
    payloads would hide. Values stay below 2**24 so they are exactly representable in
    float32 (8 * 1M = 8M < 16.7M), keeping the comparison exact.
    """
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, filename)
    tensors = {
        f"tensor_{i}": torch.arange(i * elements_per_tensor, (i + 1) * elements_per_tensor, dtype=torch.float32)
        for i in range(num_tensors)
    }
    # Use official library to ensure compatibility
    save_file(tensors, path)
    return path

def create_random_multi_safetensors(dir_path):
    """Generates multiple safetensors files to simulate a sharded model."""
    os.makedirs(dir_path, exist_ok=True)
    file_paths = []
    num_files = random.randint(MIN_NUM_FILES, MAX_NUM_FILES)

    for i in range(num_files):
        # Using the _SMALL constant to keep the total payload manageable
        file_path = os.path.join(dir_path, f"model-{i:05d}-of-{num_files:05d}.safetensors")
        data = generate_random_data(MIN_NUM_TENSORS, MAX_NUM_TENSORS_SMALL)
        # Use official library to ensure compatibility
        save_file(data, file_path)
        file_paths.append(file_path)

    return file_paths
