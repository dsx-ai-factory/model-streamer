# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage: direct and cold buffered reads of temporary files with fio, one job per file."""
import json
import os
import shutil
import subprocess

from . import GIB
from benchmark.files import evict

BLOCK = 16 << 20  # what one read asks for, and what a file's size is a multiple of
NAME = "calib_$jobnum.bin"  # fio gives every job its own file


def fio(directory, jobs, file_size, *options):
    """Run fio with one job per file and return its report for the jobs together."""
    cmd = ["fio", "--name=calib", f"--directory={directory}", f"--filename_format={NAME}", f"--numjobs={jobs}",
           f"--size={file_size}", f"--bs={BLOCK}", "--ioengine=psync", "--group_reporting", "--output-format=json",
           *options]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"fio failed: {(p.stderr or p.stdout)[-400:]}")
    return json.loads(p.stdout[p.stdout.index("{"):])["jobs"][0]


def read_rate(report):
    """GiB/s of a read job, from the bytes and the time fio counted itself."""
    read = report["read"]
    if report.get("error") or not read["runtime"]:
        raise RuntimeError(f"fio read nothing it could time: error {report.get('error')}")
    return read["io_bytes"] / GIB / (read["runtime"] / 1000)


def measure(meter, path, files, file_size):
    """disk_direct_read_parallel   O_DIRECT, one job per file
    disk_direct_read_single     one job, one file (the queue depth 1 floor)
    disk_buffered_read_cold     buffered reads after evicting the page cache"""
    print(f"disk: {files} x {file_size / GIB:.2f} GiB on {path}, read with {version()}")
    directory = os.path.join(path, f".streamer_calib_{os.getpid()}")
    os.makedirs(directory)
    try:
        # random data, new for every block, so storage that compresses or dedups cannot flatter the result
        fio(directory, files, file_size, "--rw=write", "--direct=0", "--refill_buffers", "--end_fsync=1")
        paths = [os.path.join(directory, NAME.replace("$jobnum", str(i))) for i in range(files)]

        def reads(name, jobs, direct, before=None):
            values = []
            for _ in range(meter.repeats):
                if before:
                    before()
                values.append(read_rate(fio(directory, jobs, file_size, "--rw=read", f"--direct={direct}")))
            meter.record(name, values)

        try:
            os.close(os.open(paths[0], os.O_RDONLY | os.O_DIRECT))
            reads("disk_direct_read_parallel", files, 1)
            reads("disk_direct_read_single", 1, 1)
        except OSError as e:
            meter.notes.append(f"O_DIRECT is not supported on this filesystem ({e}); direct ceilings are null")

        def cold():
            if evict(paths):
                meter.notes.append("page cache could not be fully evicted before a buffered read")

        reads("disk_buffered_read_cold", files, 0, before=cold)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def version():
    return subprocess.run(["fio", "--version"], capture_output=True, text=True).stdout.strip() or "fio"
