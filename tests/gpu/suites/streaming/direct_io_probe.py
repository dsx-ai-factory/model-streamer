#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probes that a directory serves O_DIRECT reads.

    direct_io_probe.py DIRECTORY

Writes a 4 KiB file in DIRECTORY, reads it back with O_DIRECT and removes it. Fails (a traceback, exit status 1) when the
open or the read fails, which is what a filesystem that does not support direct I/O does.
"""
import mmap
import os
import sys


def main(directory):
    path = os.path.join(directory, ".direct-io-probe")
    with open(path, "wb") as probe:
        probe.write(bytes(4096))
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            if os.preadv(fd, [mmap.mmap(-1, 4096)], 0) != 4096:
                raise OSError("short O_DIRECT read")
        finally:
            os.close(fd)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: direct_io_probe.py DIRECTORY")
    main(sys.argv[1])
