# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The measured process: one load, timed and validated, reported as a RESULT line on stdout."""
import json
import os
import resource
import sys
import time
import zlib

from . import GIB, TESTS, VALIDATE, VALIDATE_MAX_MIB, is_dist, loader_of
from .files import is_remote, model_files, read_header, resident


def thread_cpu():
    """{thread id: (user s, system s)} of this process, from /proc."""
    tick = os.sysconf("SC_CLK_TCK")
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            s = open(f"/proc/self/task/{tid}/stat").read()
        except OSError:
            continue
        r = s[s.rindex(")") + 2:].split()
        out[tid] = (int(r[11]) / tick, int(r[12]) / tick)
    return out


def expected_tensors(ref):
    """({name: (shape, nbytes)}, {name: file}) from the headers of the local files."""
    expected, where = {}, {}
    for f in ref:
        for name, meta in read_header(f).items():
            lo, hi = meta["data_offsets"]
            expected[name] = (tuple(meta["shape"]), hi - lo)
            where[name] = f
    return expected, where


def pick(names, expected, count, max_mib):
    """Spread `count` tensors of at most max_mib across `names`, deterministically."""
    small = [n for n in names if expected[n][1] <= max_mib << 20]
    step = max(1, len(small) // max(1, count))
    return set(small[::step][:count])


def check(seen, kept, expected, where):
    """Names, shapes and sizes of every tensor seen, and the bytes of the kept ones."""
    import torch
    from safetensors import safe_open
    errors = []
    if set(seen) != set(expected):
        errors.append(f"tensor names differ: {len(set(expected) ^ set(seen))} missing or extra")
    bad = [n for n, v in seen.items() if n in expected and v != expected[n]]
    if bad:
        errors.append(f"{bad[0]}: got {seen[bad[0]]}, header says {expected[bad[0]]} ({len(bad)} bad)")
    by_file = {}
    for n in kept:
        by_file.setdefault(where[n], []).append(n)
    for f, names in sorted(by_file.items()):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for n in sorted(names):
                got, ref = kept[n].cpu(), sf.get_tensor(n)
                # bytes, not values: torch.equal on values calls a tensor holding NaN different from itself
                if got.dtype != ref.dtype or got.shape != ref.shape or not torch.equal(
                        got.contiguous().reshape(-1).view(torch.uint8),
                        ref.contiguous().reshape(-1).view(torch.uint8)):
                    errors.append(f"{n}: data differs from safetensors")
    return errors


def versions():
    """Versions of the packages a load goes through. A package that is imported from somewhere other than
    where pip installed it, such as a build put first on PYTHONPATH, is not the version pip knows:
    it is reported as "local build"."""
    import importlib.util
    from importlib.metadata import PackageNotFoundError, distribution
    found = {}
    for dist, module in (("runai-model-streamer", "runai_model_streamer"), ("instanttensor", "instanttensor"),
                         ("safetensors", "safetensors"), ("torch", "torch")):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            continue
        imported = os.path.realpath(os.path.dirname(spec.origin))
        try:
            installed = distribution(dist)
        except PackageNotFoundError:
            found[dist] = "local build"
            continue
        same = imported == os.path.realpath(str(installed.locate_file(module)))
        found[dist] = installed.version if same else "local build"
    return found


def gpu_names():
    import torch
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def streamer_pkg():
    import runai_model_streamer
    return os.path.dirname(os.path.abspath(runai_model_streamer.__file__))


def load(loader, files, device, distributed=False):
    """Yield (name, tensor) for every tensor of the model, loaded by `loader` to `device`. With distributed,
    this is one rank of a torch.distributed job and its GPU is the current device."""
    import torch
    if loader == "streamer":
        from runai_model_streamer import SafetensorsStreamer
        with SafetensorsStreamer() as s:
            s.stream_files(files, device=device, is_distributed=distributed)
            yield from s.get_tensors()
    elif loader == "instanttensor":
        import torch.distributed as dist
        from instanttensor import safe_open
        group = dist.GroupMember.WORLD if distributed else None
        with safe_open(files, framework="pt", device=torch.cuda.current_device(), process_group=group) as f:
            yield from f.tensors()
    else:
        from safetensors import safe_open
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    if device == "cuda":
                        t = t.to("cuda", non_blocking=True)
                    else:
                        t.reshape(-1).view(torch.uint8)[::4096].sum()  # the file is mapped: read every page
                    yield name, t
    if device == "cuda":
        torch.cuda.synchronize()


def single(a):
    """A single-process test."""
    import torch
    files, ref = model_files(a)
    expected, where = expected_tensors(ref)
    keep = pick(sorted(expected), expected, VALIDATE, VALIDATE_MAX_MIB)
    name, device = a.child, TESTS[a.child]["device"]
    if device == "cuda":
        torch.zeros(1, device="cuda")

    kept, seen, nbytes = {}, {}, 0
    ru0, th0 = resource.getrusage(resource.RUSAGE_SELF), thread_cpu()
    t0 = time.perf_counter()
    for tensor_name, t in load(loader_of(name), files, device):
        n = t.numel() * t.element_size()
        nbytes += n
        seen[tensor_name] = (tuple(t.shape), n)
        if tensor_name in keep:
            kept[tensor_name] = t
    wall = time.perf_counter() - t0
    ru1, th1 = resource.getrusage(resource.RUSAGE_SELF), thread_cpu()

    sys_by_thread = max(th1[t][1] - th0.get(t, (0, 0))[1] for t in th1)
    result = {
        "load_s": round(wall, 3), "gib": round(nbytes / GIB, 2), "tensors": len(seen),
        "cpu_s": round(ru1.ru_utime + ru1.ru_stime - ru0.ru_utime - ru0.ru_stime, 2),
        "max_sys_thread_s": round(sys_by_thread, 2), "minor_faults": ru1.ru_minflt - ru0.ru_minflt,
        "peak_rss_gib": round(ru1.ru_maxrss / (1 << 20), 2),
        "page_cache_after_gib": None if is_remote(a.model_dir) else round(resident(files) / GIB, 2),
        "validated": len(kept), "pkg": streamer_pkg() if loader_of(name) == "streamer" else None,
        "versions": versions(), "gpus": gpu_names() if device == "cuda" else None,
        "errors": check(seen, kept, expected, where),
    }
    print("RESULT " + json.dumps(result), flush=True)
    sys.stdout.flush()
    os._exit(0)  # skip the loaders' teardown, which can hang where processes are not allowed to signal each other


def distributed(a):
    """One rank of a distributed test, started by torchrun.

    Every rank receives every tensor on its GPU, as a tensor-parallel engine's ranks do. Rank r keeps the
    tensors whose name hashes (crc32) to r mod world (a 1/world share, so device memory is under real
    pressure) and drops the rest. load_s is from a barrier before the load to a barrier after the last
    tensor, i.e. the slowest rank."""
    import torch
    import torch.distributed as dist

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()

    files, ref = model_files(a)
    expected, where = expected_tensors(ref)
    names = sorted(expected)
    keep = {n for n in names if zlib.crc32(n.encode()) % world == rank}
    chosen = pick([n for n in names if n in keep], expected, -(-VALIDATE // world), VALIDATE_MAX_MIB)
    loader = loader_of(a.child)

    kept, seen, nbytes = {}, {}, 0
    torch.zeros(1, device="cuda")
    dist.barrier()
    torch.cuda.synchronize()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    for name, t in load(loader, files, "cuda", distributed=True):
        n = t.numel() * t.element_size()
        nbytes += n
        seen[name] = (tuple(t.shape), n)
        if name in keep:
            kept[name] = t
    t_rank = time.perf_counter() - t0
    dist.barrier()
    wall = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)

    errors = check(seen, {n: kept[n] for n in chosen if n in kept}, expected, where)
    kept_gib = sum(t.numel() * t.element_size() for t in kept.values()) / GIB
    r = {"rank": rank, "rank_s": round(t_rank, 3), "load_s": round(wall, 3), "gib": round(nbytes / GIB, 2),
         "tensors": len(seen), "kept_gib": round(kept_gib, 2), "validated": len(chosen),
         "cpu_s": round(ru1.ru_utime + ru1.ru_stime - ru0.ru_utime - ru0.ru_stime, 2),
         "peak_rss_gib": round(ru1.ru_maxrss / (1 << 20), 2),
         "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
         "pkg": streamer_pkg() if loader == "streamer" else None, "versions": versions(), "gpus": gpu_names(),
         "errors": errors}
    out = [None] * world
    dist.all_gather_object(out, r)
    if rank == 0:
        print("RESULT " + json.dumps(out), flush=True)
    sys.stdout.flush()
    os._exit(0)  # skip NCCL teardown, which can hang where processes are not allowed to signal each other


def main(a):
    if is_dist(a.child):
        distributed(a)
    else:
        single(a)
