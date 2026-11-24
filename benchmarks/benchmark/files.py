# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model's files, and their state in this machine's page cache."""
import glob
import json
import os
import struct
import sys


def safetensors_files(model_dir):
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        sys.exit(f"no *.safetensors files in {model_dir}")
    return files


def is_remote(path):
    return path.startswith("s3://")


def model_files(a):
    """(what the loader reads, local files with the same names for headers and validation)."""
    if not is_remote(a.model_dir):
        files = safetensors_files(a.model_dir)
        return files, files
    if not a.ref_dir:
        sys.exit("--ref-dir is required when the model is s3://")
    ref = safetensors_files(a.ref_dir)
    return [f"{a.model_dir.rstrip('/')}/{os.path.basename(f)}" for f in ref], ref


def describe(files):
    """What a model's files hold, from their headers: how many files and tensors, the tensors' average and
    largest size in bytes, and each data type's share of the bytes."""
    sizes, by_dtype = [], {}
    for path in files:
        for meta in read_header(path).values():
            lo, hi = meta["data_offsets"]
            sizes.append(hi - lo)
            by_dtype[meta["dtype"]] = by_dtype.get(meta["dtype"], 0) + hi - lo
    total = sum(sizes) or 1
    shares = {k: round(v / total, 4) for k, v in sorted(by_dtype.items(), key=lambda kv: -kv[1])}
    return {"files": len(files), "tensors": len(sizes), "tensor_bytes_average": sum(sizes) // max(1, len(sizes)),
            "tensor_bytes_largest": max(sizes or [0]), "dtypes": shares}


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def resident(paths):
    """Bytes of the files that are in the page cache, from mincore."""
    import ctypes
    import mmap
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    total = 0
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = (ctypes.c_ubyte * pages)()
            libc.mincore(ctypes.c_void_p(addr), size, vec)
            total += sum(v & 1 for v in vec) * mmap.PAGESIZE
            libc.munmap(ctypes.c_void_p(addr), size)
        finally:
            os.close(fd)
    return total


def evict(paths):
    """Drop the files from the page cache. Returns the bytes still resident afterwards."""
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return resident(paths)


def prime(paths):
    """Read every file once so it is in the page cache. Returns the bytes resident afterwards."""
    for p in paths:
        with open(p, "rb", buffering=0) as f:
            while f.read(16 << 20):
                pass
    return resident(paths)
