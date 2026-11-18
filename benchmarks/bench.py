#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run load benchmarks and report each as a share of the machine's ceilings.

Linux only. Needs torch, safetensors and runai_model_streamer (torch with NCCL for --nproc).

    python3 bench.py /mnt/models/Qwen3-8B                       single process, the installed streamer
    python3 bench.py /mnt/models/Qwen3-8B --nproc 8             distributed, 8 ranks on this node

Every run is a fresh process (torchrun for the distributed test) on a cold page cache (evicted per file
and checked with mincore), and its tensors are checked against safetensors afterwards. A run that returns
wrong data is a failure. Runs are interleaved across the tests, so drift hits all alike.

  streamer_cpu            SafetensorsStreamer, device=cpu, owned=True (the default)
  streamer_cpu_notowned   the same with owned=False (slots reused; the caller must copy)
  streamer_cuda           SafetensorsStreamer, device=cuda
  safeopen_page_in        safe_open, one byte touched per page: the zero-copy page-cache view
  safeopen_to_cuda        safe_open + .to(cuda): the default loader's shape
  streamer_dist           SafetensorsStreamer, device=cuda, is_distributed=True on --nproc ranks

Without --tests: with --nproc only streamer_dist, otherwise every single-process test that applies here.

streamer_dist: every rank streams all files, as vLLM does, and receives every tensor. Rank r keeps the
tensors whose name hashes (crc32) to r mod world on its GPU (a TP-like 1/world shard, so device memory is
under real pressure) and drops the rest. load_s is from a barrier before stream_files to a barrier after
the last tensor, i.e. the slowest rank; GiB/s is the model size over that. cpu_s is summed over the ranks
and peak_rss_gib is the largest rank's.

The streamer measured is the runai_model_streamer this Python imports: the installed package, or one
earlier on PYTHONPATH. Its location is printed and recorded with the results. --env K=V (repeatable) is
set for every measured process, e.g. --env RUNAI_STREAMER_FS_STRATEGY=io_uring_buffered.

The MODEL may be s3://bucket/prefix; then --ref-dir must name a local copy of the same files, used for
the tensor names and the validation, only the streamer tests apply and nothing is evicted here (a
server-side cache is the caller's to drop). The S3 variables (endpoint, credentials, RUNAI_STREAMER_S3_*)
are taken from the environment.

--warm reads every local file once and checks it is resident instead of evicting it: a reload from the
page cache. The strategy the streamer actually used is read from its log and recorded per run.

The ceilings come from a machine profile written by calibrate.py. --profile FILE uses one made earlier.
Without it, machine_profile.json in the current directory is used when it is from this host and this
storage; otherwise calibrate.py is run first on the model's directory (temporary files are written there
and removed) and its profile is kept in machine_profile.json for the next run. --no-calibrate never
calibrates. A model on S3 or in a directory that cannot be written is not calibrated here: run
calibrate.py yourself and pass --profile. Without a profile the ceiling columns are empty.

A test is scored against disk_direct_read_parallel for host tests and streamer_dist (s3_get_parallel for
an S3 source, host_memcpy_nt with --warm) and the lower of that and h2d_pinned for single-process GPU
tests: "ceiling" in the results names it and gives the share reached. Its "nearest" is the stage in the
profile whose number is closest to what was achieved: a hint at what limited the run, not a proof.

The results file has the host, the streamer, the model and the settings once, then one entry per test:
the medians over its runs, the ceiling, and under "runs" what differed from run to run. A run carries
"errors" (wrong data) or "warnings" (e.g. a cache that was not cold) only when it has some. --log-dir
keeps each run's whole output, including the measured process's full record.
"""
import argparse
import glob
import json
import math
import os
import platform
import re
import resource
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
import zlib

GIB = 2**30
DIST = "streamer_dist"
DEFAULT_PROFILE = "machine_profile.json"
VALIDATE = 20  # tensors compared byte for byte per run (across the ranks, for the distributed test)
VALIDATE_MAX_MIB = 128  # the largest tensor picked for that
TIMEOUT_S = 3600  # allowed per run
TESTS = {
    "streamer_cpu": {"device": "cpu"},
    "streamer_cpu_notowned": {"device": "cpu"},
    "streamer_cuda": {"device": "cuda"},
    "safeopen_page_in": {"device": "cpu"},
    "safeopen_to_cuda": {"device": "cuda"},
    DIST: {"device": "cuda"},
}


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


def prime(paths):
    """Read every file once so it is in the page cache. Returns the bytes resident afterwards."""
    for p in paths:
        with open(p, "rb", buffering=0) as f:
            while f.read(16 << 20):
                pass
    return resident(paths)


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


def thread_cpu():
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


# ---------------------------------------------------------------------------------------------------
# The measured process
# ---------------------------------------------------------------------------------------------------

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


def child_single(a):
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
        "errors": check(rec.seen, rec.kept, expected, where),
    }
    print("RESULT " + json.dumps(result), flush=True)
    os._exit(0)


def child_dist(a):
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
         "pkg": streamer_pkg(), "errors": errors}
    out = [None] * world
    dist.all_gather_object(out, r)
    if rank == 0:
        print("RESULT " + json.dumps(out), flush=True)
    sys.stdout.flush()
    os._exit(0)  # skip NCCL teardown; processes in some pods cannot signal each other


# ---------------------------------------------------------------------------------------------------
# The machine profile
# ---------------------------------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def profile_mismatch(profile, a, remote):
    """Why the profile does not describe this host and this model's storage, or None when it does."""
    host = profile.get("host", {})
    if host.get("hostname") != platform.node():
        return f"is from host {host.get('hostname')}, this is {platform.node()}"
    if remote:
        if not profile.get("ceilings_gibps", {}).get("s3_get_parallel"):
            return "has no S3 ceilings"
        if (host.get("s3") or {}).get("uri", "").rstrip("/") != a.model_dir.rstrip("/"):
            return "was measured against another S3 prefix"
        return None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from calibrate import fs_type
    mount = fs_type(a.model_dir)["mount"]
    measured = (host.get("storage") or {}).get("mount")
    if measured != mount:
        return f"was measured on {measured or 'no storage'}, the model is on {mount}"
    return None


def get_profile(a, remote):
    """(profile, its path), or (None, None) when the run goes without ceilings."""
    if a.profile:
        profile = load_json(a.profile)
        why = profile_mismatch(profile, a, remote)
        if why:
            print(f"warning: {a.profile} {why}; its ceilings may not apply", file=sys.stderr)
        return profile, os.path.abspath(a.profile)
    if a.no_calibrate:
        return None, None
    if os.path.exists(DEFAULT_PROFILE):
        profile = load_json(DEFAULT_PROFILE)
        why = profile_mismatch(profile, a, remote)
        if not why:
            print(f"using {DEFAULT_PROFILE} (--profile FILE to use another)")
            return profile, os.path.abspath(DEFAULT_PROFILE)
        print(f"{DEFAULT_PROFILE} {why}")
    hint = "run calibrate.py yourself and pass --profile FILE. Continuing without ceilings."
    if remote:
        print(f"no machine profile for this S3 model, and S3 is not calibrated from here: {hint}", file=sys.stderr)
        return None, None
    if not os.access(a.model_dir, os.W_OK):
        print(f"no machine profile, and {a.model_dir} cannot be written to calibrate on it: {hint}",
              file=sys.stderr)
        return None, None
    print(f"calibrating this machine on {a.model_dir} first; the profile is kept in {DEFAULT_PROFILE} "
          "(--profile FILE reuses one, --no-calibrate skips this)", flush=True)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrate.py")
    if subprocess.run([sys.executable, script, "--path", a.model_dir, "--out", DEFAULT_PROFILE]).returncode:
        print("warning: calibration failed; continuing without ceilings", file=sys.stderr)
        return None, None
    return load_json(DEFAULT_PROFILE), os.path.abspath(DEFAULT_PROFILE)


def ceilings_for(test, profile, remote=False, warm=False):
    """(name of the ceiling this test is scored against, or None; the ceilings it may be nearest to)."""
    if not profile:
        return None, {}
    c = {k: v for k, v in profile["ceilings_gibps"].items() if v}
    storage = "s3_get_parallel" if remote else "host_memcpy_nt" if warm else "disk_direct_read_parallel"
    if test != DIST and TESTS[test]["device"] == "cuda":
        scored = [k for k in (storage, "h2d_pinned") if k in c]
        return (min(scored, key=c.get) if scored else None), c
    # the profile's h2d numbers are one stream to GPU 0, which says nothing about N ranks at once
    return (storage if storage in c else None), {k: v for k, v in c.items() if not k.startswith("h2d_")}


def nearest(achieved, ceilings):
    return min(ceilings, key=lambda k: abs(math.log(ceilings[k] / achieved))) if ceilings else None


# ---------------------------------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------------------------------

def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def launch(cmd, env, timeout):
    """(stdout, stderr, None or why it was killed). The whole process group goes on a timeout."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                         start_new_session=True)

    def kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            p.kill()

    try:
        out, err = p.communicate(timeout=timeout)
        return out, err, None
    except subprocess.TimeoutExpired:
        kill()
        out, err = p.communicate()
        return out, err, f"timed out after {timeout} s"
    except BaseException:
        kill()
        raise


def run_once(a, test, files, remote, base_env, tag):
    """One measured process: (the run's record, facts about the model and the streamer). A run that gave
    no result is ({"failed": why}, None). The record holds what differs from run to run; the child's
    full record is in the log under --log-dir."""
    warnings = []
    if not remote:
        left = prime(files) if a.warm else evict(files)
        if a.warm and left < sum(os.path.getsize(f) for f in files):
            warnings.append(f"only {left / GIB:.1f} GiB in the page cache after priming")
        if left and not a.warm:
            warnings.append(f"{left / GIB:.1f} GiB still in the page cache before the run")
    args = [os.path.abspath(__file__), a.model_dir, "--child", test]
    if a.ref_dir:
        args += ["--ref-dir", a.ref_dir]
    cmd = [sys.executable]
    if test == DIST:
        cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={a.nproc}"]
    out, err, killed = launch(cmd + args, base_env, TIMEOUT_S)
    if a.log_dir:
        with open(os.path.join(a.log_dir, f"{tag.replace(' ', '-')}.log"), "w") as f:
            f.write(out + "\n--- stderr ---\n" + err)
    line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
    if killed or not line:
        return {"failed": killed or (err or out)[-1500:]}, None

    raw = json.loads(line[7:])
    ranks = raw if test == DIST else [raw]
    load_s = max(x["load_s"] for x in ranks)  # the slowest rank
    r = {"load_s": load_s, "gibps": round(ranks[0]["gib"] / load_s, 2),
         "cpu_s": round(sum(x["cpu_s"] for x in ranks), 2),
         "peak_rss_gib": max(x["peak_rss_gib"] for x in ranks)}
    if test == DIST:
        r["rank_s"] = [x["rank_s"] for x in ranks]
        r["peak_gpu_gib"] = max(x["peak_gpu_gib"] for x in ranks)
        if "[Distributed]" not in err + out:
            warnings.append("the streamer logged no [Distributed] line")
    errors = [e for x in ranks for e in x["errors"]]
    if errors:
        r["errors"] = errors
    if warnings:
        r["warnings"] = warnings
    facts = {"gib": ranks[0]["gib"], "tensors": ranks[0]["tensors"], "pkg": ranks[0]["pkg"],
             "fs_strategy": sorted(set(re.findall(r"Filesystem strategy: (\S+)", out + err)))}
    return r, facts


def parent(a):
    remote = is_remote(a.model_dir)
    if remote and a.warm:
        sys.exit("--warm applies to local files only")
    files, _ = model_files(a)

    if a.tests:
        names = a.tests.split(",")
    elif a.nproc:
        names = [DIST]
    else:
        gpu = gpu_count() > 0
        names = [t for t in TESTS if t != DIST and (gpu or TESTS[t]["device"] == "cpu")]
        if remote:
            names = [n for n in names if n.startswith("streamer")]
    for n in names:
        if n not in TESTS:
            sys.exit(f"unknown test {n}; choose from {', '.join(TESTS)}")
        if remote and not n.startswith("streamer"):
            sys.exit(f"{n} reads local files; only the streamer tests apply to an s3:// model")
    if DIST in names and not a.nproc:
        sys.exit(f"{DIST} needs --nproc N")
    if a.nproc and DIST not in names:
        sys.exit(f"--nproc is for {DIST}, which --tests leaves out")

    env = {}
    for kv in a.env:
        k, sep, v = kv.partition("=")
        if not sep:
            sys.exit(f"--env {kv}: expected K=V")
        env[k] = v
    base_env = dict(os.environ, RUNAI_STREAMER_LOG_LEVEL="INFO", RUNAI_STREAMER_LOG_TO_STDERR="1", **env)
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)

    profile, profile_path = get_profile(a, remote)
    if remote and profile and not profile["ceilings_gibps"].get("s3_get_parallel"):
        print("warning: the profile has no s3_get_parallel; run calibrate.py with --s3-uri", file=sys.stderr)

    runs_of = {n: [] for n in names}
    strategy_of = {n: set() for n in names}
    model = {"path": a.model_dir if remote else os.path.abspath(a.model_dir)}
    if a.ref_dir:
        model["ref_dir"] = a.ref_dir
    pkgs = set()
    for rep in range(a.repeats):
        for name in names:
            tag = f"{name} run {rep}"
            r, facts = run_once(a, name, files, remote, base_env, tag)
            runs_of[name].append(r)
            if facts is None:
                print(f"  {tag}: FAILED\n{r['failed']}", file=sys.stderr, flush=True)
                continue
            model.setdefault("gib", facts["gib"])
            model.setdefault("tensors", facts["tensors"])
            strategy_of[name].update(facts["fs_strategy"])
            if facts["pkg"]:
                pkgs.add(facts["pkg"])
            print(f"  {tag}: {r['load_s']}s {r['gibps']} GiB/s"
                  + (f" [{','.join(facts['fs_strategy'])}]" if facts["fs_strategy"] else "")
                  + (f"  INVALID: {r['errors'][0]}" if "errors" in r else ""), flush=True)
            for w in r.get("warnings", []):
                print(f"  {tag}: warning: {w}", file=sys.stderr, flush=True)

    results = []
    for name, runs in runs_of.items():
        good = [r for r in runs if "failed" not in r]
        s = {"test": name, "ok": bool(good) and len(good) == len(runs) and all("errors" not in r for r in good)}
        if good:
            med = statistics.median(r["gibps"] for r in good)
            s.update(load_s=round(statistics.median(r["load_s"] for r in good), 2), gibps=round(med, 2),
                     gibps_min=min(r["gibps"] for r in good), gibps_max=max(r["gibps"] for r in good),
                     cpu_s_per_gib=round(statistics.median(r["cpu_s"] for r in good) / model["gib"], 2),
                     peak_rss_gib=max(r["peak_rss_gib"] for r in good))
            if strategy_of[name]:
                s["fs_strategy"] = sorted(strategy_of[name])
            scored, candidates = ceilings_for(name, profile, remote, a.warm)
            if scored:
                s["ceiling"] = {"name": scored, "gibps": candidates[scored],
                                "pct": round(100 * med / candidates[scored], 1),
                                "nearest": nearest(med, candidates)}
        s["runs"] = runs
        results.append(s)

    print(f"\nstreamer: {', '.join(sorted(pkgs)) or 'not loaded'}")
    print(f"{'test':24s} {'GiB/s':>7s} {'range':>13s} {'load s':>7s} {'% ceil':>7s} {'cpu s/GiB':>10s} "
          f"{'rss GiB':>8s}  valid  nearest ceiling")
    for s in results:
        if "gibps" not in s:
            print(f"{s['test']:24s} FAILED")
            continue
        rng = f"{s['gibps_min']:.1f}-{s['gibps_max']:.1f}"
        c = s.get("ceiling")
        print(f"{s['test']:24s} {s['gibps']:7.1f} {rng:>13s} {s['load_s']:7.2f} "
              f"{(str(round(c['pct'])) + '%') if c else 'n/a':>7s} "
              f"{s['cpu_s_per_gib']:10.2f} {s['peak_rss_gib']:8.1f}  {'yes' if s['ok'] else 'NO ':5s}  "
              f"{c['nearest'] if c else 'n/a'}")
    settings = {"repeats": a.repeats, "warm": a.warm}
    if a.nproc:
        settings["nproc"] = a.nproc
    if env:
        settings["env"] = env
    out = {"schema": 2, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
           "streamer": sorted(pkgs), "profile": profile_path, "model": model, "settings": settings,
           "results": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")
    if not all(s["ok"] for s in results):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", metavar="MODEL",
                    help="directory with *.safetensors files, or s3://bucket/prefix")
    ap.add_argument("--model-dir", help=argparse.SUPPRESS)  # the older spelling of MODEL
    ap.add_argument("--nproc", type=int, help=f"ranks for {DIST}; without --tests, run only that test")
    ap.add_argument("--tests", default="", help=f"comma list of: {', '.join(TESTS)}")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="set for every measured process (repeatable)")
    ap.add_argument("--warm", action="store_true", help="load from the page cache instead of evicting first")
    ap.add_argument("--ref-dir", help="local copy of an s3:// model, for tensor names and validation")
    ap.add_argument("--profile", help=f"machine profile from calibrate.py (default: {DEFAULT_PROFILE} here, "
                                      "made by calibrating first when it is missing)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="never calibrate; without a profile the ceiling columns are empty")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--log-dir", help="keep each run's stdout/stderr here")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.model and a.model_dir:
        ap.error("give MODEL once")
    a.model_dir = a.model or a.model_dir
    if not a.model_dir:
        ap.error("MODEL is required")
    if a.nproc is not None and a.nproc < 1:
        ap.error("--nproc must be at least 1")
    if not sys.platform.startswith("linux"):
        sys.exit("bench.py needs Linux")
    if a.child == DIST:
        child_dist(a)
    elif a.child:
        child_single(a)
    else:
        parent(a)


if __name__ == "__main__":
    main()
