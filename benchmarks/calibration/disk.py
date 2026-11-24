# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage: direct and cold buffered reads of temporary files, one thread per file."""
import os
import shutil

from . import GIB
from benchmark.files import evict
from .memory import Region, first_touch, run_threads

CHUNK = 16 << 20
BLOCK = 64 << 20


def make_files(directory, count, size):
    """Random data, so storage that compresses or dedups cannot flatter the result."""
    paths = []
    for i in range(count):
        p = os.path.join(directory, f"calib_{i}.bin")
        block = os.urandom(BLOCK)
        with open(p, "wb", buffering=0) as f:
            for _ in range(size // BLOCK):
                f.write(block)
            os.fsync(f.fileno())
        paths.append(p)
    return paths


def read_file(path, view, odirect):
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if odirect else 0))
    try:
        size, off = os.fstat(fd).st_size, 0
        while off < size:
            n = os.preadv(fd, [view[off:min(off + CHUNK, size)]], off)
            if n <= 0:
                break
            off += n
        return off
    finally:
        os.close(fd)


def read_files(paths, region, odirect):
    size = os.path.getsize(paths[0])

    def one(i):
        return read_file(paths[i], region.view[i * size:(i + 1) * size], odirect)

    return sum(run_threads(one, range(len(paths)), len(paths)))


def measure(meter, path, files, file_size, threads):
    """disk_direct_read_parallel   O_DIRECT, one thread per file, into memory already faulted in
    disk_direct_read_single     one thread, one file (the queue depth 1 floor)
    disk_direct_read_fresh_dest the same as parallel, into memory not yet touched (the kernel faults it
                                in during the read - what a loader that allocates fresh buffers sees)
    disk_buffered_read_cold     buffered reads after evicting the page cache"""
    total = file_size * files
    print(f"disk: {files} x {file_size / GIB:.2f} GiB on {path}")
    directory = os.path.join(path, f".streamer_calib_{os.getpid()}")
    os.makedirs(directory)
    try:
        paths = make_files(directory, files, file_size)
        warm = Region(total)
        first_touch(warm, threads)
        try:
            os.close(os.open(paths[0], os.O_RDONLY | os.O_DIRECT))
            direct = True
        except OSError as e:
            direct = False
            meter.notes.append(f"O_DIRECT is not supported on this filesystem ({e}); direct ceilings are null")
        if direct:
            meter.rate("disk_direct_read_parallel", lambda: read_files(paths, warm, True), total)
            meter.rate("disk_direct_read_single", lambda: read_file(paths[0], warm.view[:file_size], True), file_size)
            meter.rate("disk_direct_read_fresh_dest", lambda: read_files(paths, Region(total), True), total)

        def buffered():
            if evict(paths):
                meter.notes.append("page cache could not be fully evicted before a buffered read")
            return read_files(paths, warm, False)

        meter.rate("disk_buffered_read_cold", buffered, total)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
