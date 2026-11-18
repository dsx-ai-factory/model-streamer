# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The measured process: one load, timed and validated, reported as a RESULT line on stdout."""
import json
import os
import resource
import sys
import time
import zlib

from . import DIST, GIB, TESTS, VALIDATE, VALIDATE_MAX_MIB, VLLM, is_dist
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
    """Installed versions of the packages a load goes through, as pip knows them."""
    from importlib.metadata import PackageNotFoundError, version
    found = {}
    for dist in ("runai-model-streamer", "torch", "safetensors", "vllm", "instanttensor"):
        try:
            found[dist] = version(dist)
        except PackageNotFoundError:
            pass
    return found


def gpu_names():
    import torch
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def streamer_pkg():
    import runai_model_streamer
    return os.path.dirname(os.path.abspath(runai_model_streamer.__file__))


class Recorder:
    """Counts bytes, keeps the tensors chosen for validation, and times its own copies so the caller can
    take them out of the load time."""

    def __init__(self, keep_names):
        self.keep_names = keep_names
        self.kept = {}
        self.nbytes = 0
        self.seen = {}
        self.overhead = 0.0

    def __call__(self, name, tensor, copy):
        n = tensor.numel() * tensor.element_size()
        self.nbytes += n
        self.seen[name] = (tuple(tensor.shape), n)
        if name in self.keep_names:
            t0 = time.perf_counter()
            self.kept[name] = tensor.clone() if copy else tensor
            self.overhead += time.perf_counter() - t0


def run_streamer(files, device, owned, rec):
    import torch
    from runai_model_streamer import SafetensorsStreamer
    with SafetensorsStreamer() as s:
        s.stream_files(files, device=device, owned=owned)
        for name, t in s.get_tensors():
            rec(name, t, copy=not owned)
    if device == "cuda":
        torch.cuda.synchronize()


def run_safeopen(files, device, rec):
    import torch
    from safetensors import safe_open
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                if device == "cuda":
                    rec(k, t.to("cuda", non_blocking=True), copy=False)
                else:
                    t.reshape(-1).view(torch.uint8)[::4096].sum()
                    rec(k, t, copy=True)
    if device == "cuda":
        torch.cuda.synchronize()


def single(a):
    """A single-process test."""
    import torch
    files, ref = model_files(a)
    expected, where = expected_tensors(ref)
    rec = Recorder(pick(sorted(expected), expected, VALIDATE, VALIDATE_MAX_MIB))
    name = a.child
    if TESTS[name]["device"] == "cuda":
        torch.zeros(1, device="cuda")

    ru0, th0 = resource.getrusage(resource.RUSAGE_SELF), thread_cpu()
    t0 = time.perf_counter()
    if name.startswith("streamer"):
        run_streamer(files, TESTS[name]["device"], name != "streamer_cpu_notowned", rec)
    else:
        run_safeopen(files, TESTS[name]["device"], rec)
    wall = time.perf_counter() - t0 - rec.overhead
    ru1, th1 = resource.getrusage(resource.RUSAGE_SELF), thread_cpu()

    sys_by_thread = max(th1[t][1] - th0.get(t, (0, 0))[1] for t in th1)
    result = {
        "load_s": round(wall, 3), "gib": round(rec.nbytes / GIB, 2), "tensors": len(rec.seen),
        "cpu_s": round(ru1.ru_utime + ru1.ru_stime - ru0.ru_utime - ru0.ru_stime, 2),
        "max_sys_thread_s": round(sys_by_thread, 2), "minor_faults": ru1.ru_minflt - ru0.ru_minflt,
        "peak_rss_gib": round(ru1.ru_maxrss / (1 << 20), 2),
        "page_cache_after_gib": None if is_remote(a.model_dir) else round(resident(files) / GIB, 2),
        "validated": len(rec.kept), "pkg": streamer_pkg() if name.startswith("streamer") else None,
        "versions": versions(), "gpus": gpu_names() if TESTS[name]["device"] == "cuda" else None,
        "errors": check(rec.seen, rec.kept, expected, where),
    }
    print("RESULT " + json.dumps(result), flush=True)
    os._exit(0)


def distributed(a):
    """One rank of the distributed test, started by torchrun.

    Every rank streams all files, as vLLM does, and receives every tensor. Rank r keeps the tensors whose
    name hashes (crc32) to r mod world on its GPU (a TP-like 1/world shard, so device memory is under real
    pressure) and drops the rest. load_s is from a barrier before stream_files to a barrier after the
    last tensor, i.e. the slowest rank."""
    import torch
    import torch.distributed as dist
    from runai_model_streamer import SafetensorsStreamer

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()

    files, ref = model_files(a)
    expected, where = expected_tensors(ref)
    names = sorted(expected)
    keep = {n for n in names if zlib.crc32(n.encode()) % world == rank}
    chosen = pick([n for n in names if n in keep], expected, -(-VALIDATE // world), VALIDATE_MAX_MIB)

    kept, seen, nbytes = {}, {}, 0
    torch.zeros(1, device="cuda")
    dist.barrier()
    torch.cuda.synchronize()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    with SafetensorsStreamer() as s:
        s.stream_files(files, device="cuda", is_distributed=True)
        for name, t in s.get_tensors():
            n = t.numel() * t.element_size()
            nbytes += n
            seen[name] = (tuple(t.shape), n)
            if name in keep:
                kept[name] = t
    torch.cuda.synchronize()
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
         "pkg": streamer_pkg(), "versions": versions(), "gpus": gpu_names(), "errors": errors}
    out = [None] * world
    dist.all_gather_object(out, r)
    if rank == 0:
        print("RESULT " + json.dumps(out), flush=True)
    sys.stdout.flush()
    os._exit(0)  # skip NCCL teardown, which can hang where processes are not allowed to signal each other


def vllm(a):
    """A vLLM engine start with the default loader, the streamer or InstantTensor. The load time is not taken
    here: vLLM logs it ("Model loading took ... seconds") from its engine process, and the runner reads
    it from the output. The tensors are not validated; a run is good when the engine came up.

    Eager mode and no tokenizer keep the start short and need only config.json next to the weights;
    neither changes how the weights are loaded. The engine runs in this process, so that leaving with
    os._exit frees the GPU without a child engine to shut down.

    A distributed test is one rank of torchrun: tensor parallelism over the ranks with vLLM's
    external launcher. For the streamer, vLLM's "distributed" loader option makes the GPU the
    destination; without it the streamer loads to host memory and vLLM copies from there."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import inspect
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    _, ref = model_files(a)
    expected, _ = expected_tensors(ref)
    nbytes = sum(n for _, n in expected.values())
    load_format = TESTS[a.child]["load_format"]
    streamer = load_format == "runai_streamer"

    options = dict(model=a.model_dir, load_format=load_format, enforce_eager=True, skip_tokenizer_init=True)
    if is_dist(a.child):
        options.update(tensor_parallel_size=int(os.environ["WORLD_SIZE"]),
                       distributed_executor_backend="external_launcher")
    if TESTS[a.child].get("extra"):
        options["model_loader_extra_config"] = dict(TESTS[a.child]["extra"])
    if "kernel_config" in inspect.signature(EngineArgs).parameters:
        options["kernel_config"] = {"enable_flashinfer_autotune": False}  # minutes of tuning after the load
    options.update(a.vllm_args)  # --vllm-args: what this model needs, e.g. trust_remote_code

    t0 = time.perf_counter()
    llm = LLM(**options)  # noqa: F841 - kept alive until the result is out
    startup = time.perf_counter() - t0
    if int(os.environ.get("RANK", "0")) == 0:
        result = {"startup_s": round(startup, 3), "gib": round(nbytes / GIB, 2), "tensors": len(expected),
                  "pkg": streamer_pkg() if streamer else None, "versions": versions(), "gpus": gpu_names(),
                  "errors": []}
        print("RESULT " + json.dumps(result), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip the engine's teardown, which can hang the same way as NCCL's


def main(a):
    if a.child in VLLM:
        vllm(a)
    elif a.child == DIST:
        distributed(a)
    else:
        single(a)
