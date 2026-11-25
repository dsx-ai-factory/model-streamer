# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The measured process: one load, timed and validated, reported as a RESULT line on stdout."""
import json
import os
import resource
import sys
import time
import zlib

from . import (GIB, TESTS, VALIDATE, VALIDATE_MAX_MIB, VLLM_OPTIONS, is_dist,
               is_vllm, loader_of)
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


# safetensors dtype names as torch names. A dtype that is not here is not compared.
DTYPES = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "U16": "uint16", "I32": "int32",
          "U32": "uint32", "I64": "int64", "U64": "uint64", "F16": "float16", "BF16": "bfloat16",
          "F32": "float32", "F64": "float64", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}


def expected_tensors(ref):
    """({name: (shape, nbytes, dtype)}, {name: file}) from the headers of the local files."""
    expected, where = {}, {}
    for f in ref:
        for name, meta in read_header(f).items():
            lo, hi = meta["data_offsets"]
            expected[name] = (tuple(meta["shape"]), hi - lo, DTYPES.get(meta["dtype"]))
            where[name] = f
    return expected, where


def pick(names, expected, count, max_mib):
    """Spread `count` tensors of at most max_mib across `names`, deterministically."""
    small = [n for n in names if expected[n][1] <= max_mib << 20]
    step = max(1, len(small) // max(1, count))
    return set(small[::step][:count])


def facts(t):
    """A loaded tensor as expected_tensors describes it."""
    return tuple(t.shape), t.numel() * t.element_size(), str(t.dtype).rpartition(".")[2]


def differs(got, want):
    return got[:2] != want[:2] or (want[2] is not None and got[2] != want[2])


def check(seen, kept, expected, where):
    """Names, shapes, sizes and dtypes of every tensor seen, and the bytes of the kept ones."""
    import torch
    from safetensors import safe_open
    errors = []
    if set(seen) != set(expected):
        errors.append(f"tensor names differ: {len(set(expected) ^ set(seen))} missing or extra")
    bad = [n for n, v in seen.items() if n in expected and differs(v, expected[n])]
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
                         ("safetensors", "safetensors"), ("torch", "torch"), ("vllm", "vllm")):
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
        # The GPU by its index, as vLLM's loader passes it, not left for the streamer to resolve.
        target = f"cuda:{torch.cuda.current_device()}" if device == "cuda" else device
        with SafetensorsStreamer() as s:
            s.stream_files(files, device=target, is_distributed=distributed)
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

    held, kept, seen, nbytes = [], {}, {}, 0
    ru0, th0 = resource.getrusage(resource.RUSAGE_SELF), thread_cpu()
    t0 = time.perf_counter()
    for tensor_name, t in load(loader_of(name), files, device):
        seen[tensor_name] = facts(t)
        nbytes += seen[tensor_name][1]
        held.append(t)  # the destination holds the whole model, as an engine's does
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
    tensor, i.e. the slowest rank.

    The job's NCCL group is created as vLLM creates its own: without a device_id, which would build the
    communicator at once, and with no collective before the load. The barriers therefore run on a gloo
    group of their own, and a loader that sends on the job's group sets up its communicator inside the
    measured time, as it does in an engine."""
    import torch
    import torch.distributed as dist

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    files, ref = model_files(a)
    expected, where = expected_tensors(ref)
    names = sorted(expected)
    keep = {n for n in names if zlib.crc32(n.encode()) % world == rank}
    chosen = pick([n for n in names if n in keep], expected, -(-VALIDATE // world), VALIDATE_MAX_MIB)
    loader = loader_of(a.child)

    kept, seen, nbytes = {}, {}, 0
    torch.zeros(1, device="cuda")
    lineup = dist.new_group(backend="gloo")
    dist.barrier(group=lineup)
    torch.cuda.synchronize()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    for name, t in load(loader, files, "cuda", distributed=True):
        seen[name] = facts(t)
        nbytes += seen[name][1]
        if name in keep:
            kept[name] = t
    torch.cuda.synchronize()  # the gloo barrier does not wait for the GPU
    t_rank = time.perf_counter() - t0
    dist.barrier(group=lineup)
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


def time_weight_loading():
    """Make every vLLM model loader time its load_weights step, to the point where the weights are on the
    GPU, and return the list the times are appended to. This is the loading itself, without building
    the model before it or the engine's warm-up after it."""
    import torch
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    took = []

    def timed(load_weights):
        def wrapper(self, model, model_config):
            t0 = time.perf_counter()
            try:
                return load_weights(self, model, model_config)
            finally:
                torch.cuda.synchronize()
                took.append(time.perf_counter() - t0)
        return wrapper

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    import vllm.model_executor.model_loader  # noqa: F401 - defines the loaders
    for cls in set(subclasses(BaseModelLoader)):
        if "load_weights" in vars(cls):
            cls.load_weights = timed(cls.load_weights)
    return took


def vllm(a):
    """A vLLM engine start with one of the loaders. The time reported is vLLM's weight loading step, taken
    around its loader's load_weights. The tensors are not validated; a run is good when the engine came
    up and its weights reached the GPU.

    Eager mode and no tokenizer keep the start short and need only config.json next to the weights;
    neither changes how the weights are loaded. The engine runs in this process, so that leaving with
    os._exit frees the GPU without a child engine to shut down. A distributed test is one rank of
    torchrun: tensor parallelism over the ranks with vLLM's external launcher."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import inspect
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    _, ref = model_files(a)
    expected, _ = expected_tensors(ref)
    nbytes = sum(v[1] for v in expected.values())

    options = dict(model=a.model_dir, enforce_eager=True, skip_tokenizer_init=True, **VLLM_OPTIONS[loader_of(a.child)])
    if is_dist(a.child):
        options.update(tensor_parallel_size=int(os.environ["WORLD_SIZE"]),
                       distributed_executor_backend="external_launcher")
    if "kernel_config" in inspect.signature(EngineArgs).parameters:
        options["kernel_config"] = {"enable_flashinfer_autotune": False}  # minutes of tuning after the load
    options.update(a.vllm_args)  # --vllm-args: what this model needs, e.g. trust_remote_code

    took = time_weight_loading()
    t0 = time.perf_counter()
    llm = LLM(**options)  # noqa: F841 - kept alive until the result is out
    startup = time.perf_counter() - t0
    print(f"WEIGHTS {sum(took):.3f}", flush=True)  # one line per rank
    if int(os.environ.get("RANK", "0")) == 0:
        result = {"startup_s": round(startup, 3), "gib": round(nbytes / GIB, 2), "tensors": len(expected),
                  "pkg": streamer_pkg() if loader_of(a.child) == "streamer" else None, "versions": versions(),
                  "gpus": gpu_names(), "errors": []}
        print("RESULT " + json.dumps(result), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip the engine's teardown, which can hang where processes are not allowed to signal each other


def main(a):
    if is_vllm(a.child):
        vllm(a)
    elif is_dist(a.child):
        distributed(a)
    else:
        single(a)
