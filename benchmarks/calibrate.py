#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure what this machine can physically do, and write it to a JSON machine profile.

Linux only. Standard library only; torch is used for the host-to-GPU numbers when it is importable.

    python3 calibrate.py --path /mnt/models --out machine_profile.json

--path is a directory on the storage under test. Temporary files are created there and removed.
Each ceiling is the median of --repeats runs, in GiB/s (2**30 bytes per second). The stages:

  disk_direct_read_parallel      O_DIRECT reads, one thread per file, into memory already faulted in
  disk_direct_read_fresh_dest    the same, into memory not yet touched (the kernel faults it in during
                                 the read - what a streamer that allocates fresh buffers sees)
  disk_direct_read_single        one thread, one file (queue depth 1 floor)
  disk_buffered_read_cold        buffered reads after evicting the page cache
  host_first_touch_1t / _nt      first-touch of fresh anonymous memory (page faults + zeroing)
  host_memcpy_1t / _nt           memory-to-memory copy
  h2d_pinned / h2d_pageable      host to GPU 0 copy, from pinned and from pageable memory
  s3_get_single                  one object, one part in flight (the per-connection floor)
  s3_get_parallel                every object of --s3-uri at once, the best of a sweep over parts in
                                 flight per object and part size (the settings are in params.s3_best)

The S3 stages run only with --s3-uri, and use s5cmd (pip install s5cmd): a native client, so the
ceiling is set by the network and the server, not by Python or by the streamer under test. Bytes go to
/dev/null. The endpoint comes from AWS_ENDPOINT_URL_S3, AWS_ENDPOINT_URL or S3_ENDPOINT_URL, the
credentials from the usual AWS variables. The first S3 read is reported apart (s3_first_run in
raw_gibps), because connection setup, server warm-up and network burst credits can make it differ.
"""
import argparse
import ctypes
import json
import mmap
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

SCHEMA = 1
GIB = 2**30
CHUNK = 16 << 20
BLOCK = 64 << 20
MADV_POPULATE_WRITE = 23

libc = ctypes.CDLL(None, use_errno=True)
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


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


def timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def meminfo_gib(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024 / GIB
    return None


def memory_budget_gib():
    """What this process may use: MemAvailable, capped by the cgroup limit when there is one."""
    budget = meminfo_gib("MemAvailable") or 8
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            if raw.isdigit() and int(raw) < 2**60:
                budget = min(budget, int(raw) / GIB)
        except OSError:
            pass
    return budget


def fs_type(path):
    best, kind = "", "unknown"
    real = os.path.realpath(path)
    with open("/proc/mounts") as f:
        for line in f:
            _, mount, fstype = line.split()[:3]
            if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
                best, kind = mount, fstype
    return {"mount": best, "type": kind}


def read_text(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def host_info(path):
    info = {
        "hostname": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
        "cpus": len(os.sched_getaffinity(0)), "mem_total_gib": round(meminfo_gib("MemTotal") or 0, 1),
        "memory_budget_gib": round(memory_budget_gib(), 1),
        "thp": read_text("/sys/kernel/mm/transparent_hugepage/enabled"),
        "numa_nodes": len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")])
        if os.path.isdir("/sys/devices/system/node") else None,
        "storage": {"path": os.path.realpath(path), **fs_type(path)},
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def evict(paths):
    """Drop each file from the page cache, then count what is still resident (bytes)."""
    resident = 0
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = (ctypes.c_ubyte * pages)()
            libc.mincore(ctypes.c_void_p(addr), size, vec)
            resident += sum(v & 1 for v in vec) * mmap.PAGESIZE
            libc.munmap(ctypes.c_void_p(addr), size)
        finally:
            os.close(fd)
    return resident


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


def h2d(pinned, gib=1, reps=8):
    import torch
    n = gib * GIB
    src = torch.empty(n, dtype=torch.uint8)
    src.fill_(1)
    if pinned:
        src = src.pin_memory()
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst.copy_(src, non_blocking=pinned)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dst.copy_(src, non_blocking=pinned)
    torch.cuda.synchronize()
    return gib * reps / (time.perf_counter() - t0)


def median_of(values):
    return round(statistics.median(values), 2) if values else None


def s3_endpoint():
    for k in ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def s5cmd(*args):
    endpoint = s3_endpoint()
    return ["s5cmd"] + (["--endpoint-url", endpoint] if endpoint else []) + list(args)


def s3_objects(uri):
    """(uri, size) of every object under the prefix; only the *.safetensors ones when there are any."""
    prefix = uri.rstrip("/")
    out = subprocess.run(s5cmd("ls", prefix + "/*"), capture_output=True, text=True, check=True).stdout
    objs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2].isdigit():
            objs.append((f"{prefix}/{parts[3]}", int(parts[2])))
    st = [o for o in objs if o[0].endswith(".safetensors")]
    return st or objs


def s3_read(objs, concurrency, part_mib):
    """Read every object at once, `concurrency` parts in flight each, into /dev/null."""
    procs = [subprocess.Popen(s5cmd("cat", "-c", str(concurrency), "-p", str(part_mib), u),
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for u, _ in objs]
    for p in procs:
        _, err = p.communicate()
        if p.returncode:
            raise RuntimeError(f"s5cmd failed: {err.decode()[-400:]}")
    return sum(n for _, n in objs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", help="directory on the storage under test (needed unless disk is skipped)")
    ap.add_argument("--out", default="machine_profile.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--size-gib", type=float, default=16, help="bytes read per disk measurement")
    ap.add_argument("--files", type=int, default=8, help="files read in parallel")
    ap.add_argument("--mem-gib", type=float, default=4, help="buffer size for the memory measurements")
    ap.add_argument("--threads", type=int, default=min(32, len(os.sched_getaffinity(0))))
    ap.add_argument("--skip", default="", help="comma list of: disk,host,gpu")
    ap.add_argument("--s3-uri", help="s3://bucket/prefix whose objects are read for the S3 ceilings")
    ap.add_argument("--s3-concurrency", default="4,8,16,32", help="parts in flight per object, swept")
    ap.add_argument("--s3-part-mib", default="8,32", help="part sizes in MiB, swept")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))

    if not sys.platform.startswith("linux"):
        sys.exit("calibrate.py needs Linux (O_DIRECT, /proc)")
    if "disk" not in skip:
        if not a.path:
            ap.error("--path is required unless --skip includes disk")
        os.makedirs(a.path, exist_ok=True)
    if a.s3_uri and not shutil.which("s5cmd"):
        sys.exit("--s3-uri needs s5cmd on PATH (pip install s5cmd)")

    budget = memory_budget_gib()
    notes, raw = [], {}
    size_gib = a.size_gib
    while size_gib * 2.5 > budget and size_gib > 1:
        size_gib /= 2
    if size_gib != a.size_gib and "disk" not in skip:
        notes.append(f"disk size reduced from {a.size_gib} to {size_gib} GiB to fit {budget:.0f} GiB of memory")
    mem_gib = min(a.mem_gib, budget / 3)
    file_size = max(1, int(size_gib * GIB / a.files) // BLOCK) * BLOCK
    total = file_size * a.files

    def measure(name, fn, nbytes):
        values = []
        for _ in range(a.repeats):
            _, dt = timed(fn)
            values.append(nbytes / GIB / dt)
        raw[name] = [round(v, 2) for v in values]
        print(f"  {name:32s} {median_of(values):8.1f} GiB/s   runs {raw[name]}", flush=True)

    print(f"host: {platform.node()}  cpus={len(os.sched_getaffinity(0))}  memory budget={budget:.0f} GiB")

    if "host" not in skip:
        print("host memory")
        nbytes = int(mem_gib * GIB) // (2 << 20) * (2 << 20)
        for label, threads in (("1t", 1), ("nt", a.threads)):
            measure(f"host_first_touch_{label}", lambda: first_touch(Region(nbytes), threads), nbytes)
        src, dst = Region(nbytes), Region(nbytes)
        first_touch(src, a.threads)
        first_touch(dst, a.threads)
        for label, threads in (("1t", 1), ("nt", a.threads)):
            measure(f"host_memcpy_{label}", lambda: memcpy(dst, src, threads), nbytes)
        del src, dst

    if "disk" not in skip:
        print(f"disk: {a.files} x {file_size / GIB:.2f} GiB on {a.path}")
        directory = os.path.join(a.path, f".streamer_calib_{os.getpid()}")
        os.makedirs(directory)
        try:
            paths = make_files(directory, a.files, file_size)
            try:
                os.close(os.open(paths[0], os.O_RDONLY | os.O_DIRECT))
                direct = True
            except OSError as e:
                direct = False
                notes.append(f"O_DIRECT is not supported on this filesystem ({e}); direct ceilings are null")
            if direct:
                warm = Region(total)
                first_touch(warm, a.threads)
                measure("disk_direct_read_parallel", lambda: read_files(paths, warm, True), total)
                measure("disk_direct_read_single",
                        lambda: read_file(paths[0], warm.view[:file_size], True), file_size)
                measure("disk_direct_read_fresh_dest", lambda: read_files(paths, Region(total), True), total)
            else:
                warm = Region(total)
                first_touch(warm, a.threads)

            def buffered():
                if evict(paths):
                    notes.append("page cache could not be fully evicted before a buffered read")
                return read_files(paths, warm, False)

            measure("disk_buffered_read_cold", buffered, total)
            del warm
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    if "gpu" not in skip:
        try:
            import torch
            if torch.cuda.is_available():
                print("host to GPU 0")
                for name, pinned in (("h2d_pinned", True), ("h2d_pageable", False)):
                    values = []
                    for _ in range(a.repeats):
                        values.append(h2d(pinned))
                    raw[name] = [round(v, 2) for v in values]
                    print(f"  {name:32s} {median_of(values):8.1f} GiB/s   runs {raw[name]}", flush=True)
            else:
                notes.append("no CUDA device: h2d ceilings are null")
        except ImportError:
            notes.append("torch not importable: h2d ceilings are null")

    s3_params, s3_best = None, None
    if a.s3_uri:
        objs = s3_objects(a.s3_uri)
        if not objs:
            sys.exit(f"no objects under {a.s3_uri}")
        total_s3 = sum(n for _, n in objs)
        endpoint = s3_endpoint()
        print(f"s3: {len(objs)} objects, {total_s3 / GIB:.2f} GiB under {a.s3_uri}"
              f" via {endpoint or 'the default AWS endpoint'}")
        _, dt = timed(lambda: s3_read(objs, 8, 8))
        raw["s3_first_run"] = [round(total_s3 / GIB / dt, 2)]
        print(f"  {'s3_first_run':32s} {raw['s3_first_run'][0]:8.2f} GiB/s   (8 x 8 MiB, not a ceiling)", flush=True)
        largest = [max(objs, key=lambda o: o[1])]
        measure("s3_get_single", lambda: s3_read(largest, 1, 8), largest[0][1])
        sweep = {}
        for c in (int(x) for x in a.s3_concurrency.split(",")):
            for p in (int(x) for x in a.s3_part_mib.split(",")):
                key = f"s3_get_c{c}_p{p}"
                measure(key, lambda: s3_read(objs, c, p), total_s3)
                sweep[key] = (c, p)
        best = max(sweep, key=lambda k: statistics.median(raw[k]))
        raw["s3_get_parallel"] = raw[best]
        s3_best = {"concurrency_per_object": sweep[best][0], "part_mib": sweep[best][1], "objects": len(objs)}
        s3_params = {"uri": a.s3_uri, "endpoint": endpoint, "gib": round(total_s3 / GIB, 3),
                     "tool": subprocess.run(["s5cmd", "version"], capture_output=True, text=True).stdout.strip()}
        print(f"  {'s3_get_parallel':32s} {median_of(raw[best]):8.2f} GiB/s   best: {best}")

    names = ["disk_direct_read_parallel", "disk_direct_read_fresh_dest", "disk_direct_read_single",
             "disk_buffered_read_cold", "host_first_touch_1t", "host_first_touch_nt",
             "host_memcpy_1t", "host_memcpy_nt", "h2d_pinned", "h2d_pageable",
             "s3_get_single", "s3_get_parallel"]
    host = host_info(a.path or ".")
    if "disk" in skip:
        host.pop("storage", None)
    if s3_params:
        host["s3"] = s3_params
    profile = {
        "schema": SCHEMA, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host,
        "params": {"repeats": a.repeats, "files": a.files, "file_gib": round(file_size / GIB, 3),
                   "mem_gib": round(mem_gib, 2), "threads": a.threads,
                   "data": "random 64 MiB block per file, repeated", "s3_best": s3_best},
        "ceilings_gibps": {n: median_of(raw.get(n, [])) for n in names},
        "raw_gibps": raw, "notes": notes,
    }
    with open(a.out, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"wrote {a.out}")
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()
