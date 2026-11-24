# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host memory: first-touch of fresh anonymous memory (page faults + zeroing) and memory-to-memory copy."""
import ctypes
import mmap
from concurrent.futures import ThreadPoolExecutor

from . import GIB

MADV_POPULATE_WRITE = 23

libc = ctypes.CDLL(None, use_errno=True)
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


class Region:
    """Anonymous memory, page aligned, with huge pages requested the way numpy and torch do."""

    def __init__(self, size):
        self.size = size
        # PRIVATE: the default (MAP_SHARED) is shmem, which has no huge pages and faults slower
        self.map = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        if hasattr(mmap, "MADV_HUGEPAGE"):
            try:
                self.map.madvise(mmap.MADV_HUGEPAGE)
            except OSError:
                pass
        self._c = ctypes.c_char.from_buffer(self.map)
        self.addr = ctypes.addressof(self._c)
        self.view = memoryview(self.map)


def slabs(size, parts):
    """(offset, length) pieces of `size`, about `parts` of them, each a multiple of 2 MiB but the last."""
    step = max(2 << 20, -(-size // parts) // (2 << 20) * (2 << 20))
    return [(o, min(step, size - o)) for o in range(0, size, step)]


def run_threads(fn, items, threads):
    with ThreadPoolExecutor(max(1, min(threads, len(items)))) as ex:
        return list(ex.map(fn, items))


def first_touch(region, threads):
    """Fault in every page. MADV_POPULATE_WRITE where the kernel has it, else a memset."""
    def one(slab):
        off, n = slab
        if libc.madvise(region.addr + off, n, MADV_POPULATE_WRITE) != 0:
            ctypes.memset(region.addr + off, 0, n)
    run_threads(one, slabs(region.size, threads), threads)


def memcpy(dst, src, threads):
    def one(slab):
        off, n = slab
        ctypes.memmove(dst.addr + off, src.addr + off, n)
    run_threads(one, slabs(dst.size, threads), threads)


def meminfo_gib(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024 / GIB
    return None


def cgroup_left(limit, usage, stat, cache_keys):
    """Bytes the cgroup may still take: its limit less what it holds now, not counting the page cache,
    which the kernel gives back. None when there is no limit."""
    def text(path):
        with open(path) as f:
            return f.read()

    try:
        raw = text(limit).strip()
        if not raw.isdigit() or int(raw) >= 2**60:
            return None
        used = int(text(usage))
        for line in text(stat).splitlines():
            key, _, value = line.partition(" ")
            if key in cache_keys:
                used -= int(value)
        return max(0, int(raw) - max(0, used))
    except (OSError, ValueError):
        return None


def memory_budget_gib():
    """What this process may use: MemAvailable, capped by what is left of the cgroup limit when there is one."""
    budget = meminfo_gib("MemAvailable") or 8
    v2, v1 = "/sys/fs/cgroup/", "/sys/fs/cgroup/memory/"
    for left in (cgroup_left(v2 + "memory.max", v2 + "memory.current", v2 + "memory.stat", ("file",)),
                 cgroup_left(v1 + "memory.limit_in_bytes", v1 + "memory.usage_in_bytes", v1 + "memory.stat",
                             ("total_cache",))):
        if left is not None:
            budget = min(budget, left / GIB)
    return budget


def measure(meter, mem_gib, threads):
    """host_first_touch_1t / _nt and host_memcpy_1t / _nt."""
    print("host memory")
    nbytes = int(mem_gib * GIB) // (2 << 20) * (2 << 20)
    for label, n in (("1t", 1), ("nt", threads)):
        meter.rate(f"host_first_touch_{label}", lambda: first_touch(Region(nbytes), n), nbytes)
    src, dst = Region(nbytes), Region(nbytes)
    first_touch(src, threads)
    first_touch(dst, threads)
    for label, n in (("1t", 1), ("nt", threads)):
        meter.rate(f"host_memcpy_{label}", lambda: memcpy(dst, src, n), nbytes)
