# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark itself: which tests to run, and one fresh measured process per run."""
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from . import ENTRY, GIB, GPU_IDLE_GIB, SCRATCH, SKIPPED, TESTS, TIMEOUT_S, group_of, is_dist, loader_of
from .files import describe, evict, is_remote, model_files, prime
from .profile import get_profile
from .report import report


def gpu_memory_gib():
    """The memory of each GPU on this machine, in GiB; empty without a GPU or without torch."""
    try:
        import torch
        return [torch.cuda.get_device_properties(i).total_memory / GIB for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def gpu_count():
    return len(gpu_memory_gib())


def gpu_used_gib():
    """The memory already in use on each GPU the tests can see, in GiB, as nvidia-smi reports it, or None
    where that cannot be told. Asked of nvidia-smi and not of torch, so that this process does not
    itself hold memory on the GPUs it is about to measure."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
        used = [float(x) / 1024 for x in out.stdout.split()]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:  # nvidia-smi lists every GPU, the tests see only these
            used = [used[int(i)] for i in visible.split(",") if i.strip()]
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None
    return used if used and len(used) == gpu_count() else None


def gpus_of(group, nproc=None):
    """The GPUs a destination loads to: the first one, or one per rank."""
    return list(range(nproc or 0)) if group == "dist" else [0] if group == "cuda" else []


def gpus_in_use(group, nproc=None):
    """{GPU: GiB in use} for the GPUs a destination loads to that something else is already using.
    A GPU nobody uses still shows a few MiB, so a little is not counted."""
    used = gpu_used_gib() or []
    return {i: round(used[i], 1) for i in gpus_of(group, nproc) if i < len(used) and used[i] > GPU_IDLE_GIB}


def misfit(group, size_gib, nproc=None):
    """Why a model of this size cannot be loaded to a destination on this machine, or None when it can.
    One GPU has to hold the whole model; with N ranks each GPU keeps a 1/N share of it. What counts is
    the memory that is free now, and some room is left for the loader's own buffers."""
    gpus = gpu_memory_gib()
    used = gpu_used_gib() or [0.0] * len(gpus)
    free = [total - taken for total, taken in zip(gpus, used)]
    taken = lambda i: f", of which {used[i]:.0f} GiB is in use by something else" if used[i] > GPU_IDLE_GIB else ""
    if group == "cuda" and gpus and size_gib > 0.95 * free[0]:
        return f"the model is {size_gib:.0f} GiB and one GPU has {gpus[0]:.0f} GiB{taken(0)}"
    if group == "dist" and gpus and nproc and size_gib / nproc > 0.9 * min(free[:nproc]):
        tight = min(range(nproc), key=lambda i: free[i])
        need = next((n for n in range(nproc + 1, len(gpus) + 1) if size_gib / n <= 0.9 * min(free[:n])), None)
        return (f"each of {nproc} ranks keeps {size_gib / nproc:.0f} GiB of the model and a GPU has "
                f"{gpus[tight]:.0f} GiB{taken(tight)}" + (f"; it needs at least {need} ranks" if need else ""))
    return None


def unavailable(a, test, remote=False):
    """Why a test cannot run here, or None. Its loader has to be installed, and a loader that reads only
    local files needs s5cmd to download an s3:// model first."""
    if loader_of(test) == "instanttensor" and importlib.util.find_spec("instanttensor") is None:
        return f"instanttensor is not installed in this Python ({sys.executable}): pip install instanttensor"
    if remote and loader_of(test) != "streamer" and not shutil.which("s5cmd"):
        return "it reads local files, and downloading the s3:// model needs s5cmd on PATH (pip install s5cmd)"
    return None


def download(uri, scratch, profile):
    """Download an s3:// model's *.safetensors files into the emptied scratch directory with s5cmd:
    (seconds, None), or (None, why it failed). The files in flight, the parts in flight per file and
    the part size are the profile's best S3 setting when it has one, so that the download asks of the
    storage what the calibration found it serves fastest."""
    from .calibration.s3 import s5cmd
    best = ((profile or {}).get("params") or {}).get("s3_best") or {}
    clear(scratch)
    cmd = s5cmd("--numworkers", str(best.get("objects", 16)), "cp", "-c", str(best.get("concurrency_per_object", 8)),
                "-p", str(best.get("part_mib", 8)), f"{uri.rstrip('/')}/*.safetensors", scratch.rstrip("/") + "/")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        return None, f"s5cmd could not download {uri}: {p.stderr[-800:]}"
    return time.perf_counter() - t0, None


def clear(scratch):
    """Empty the scratch directory of model files, and make it when it is missing."""
    os.makedirs(scratch, exist_ok=True)
    for name in os.listdir(scratch):
        if name.endswith(".safetensors"):
            os.remove(os.path.join(scratch, name))


def describe_machine(a, remote):
    """The machine the benchmark ran on, for the results: kernel, CPUs, memory and the model's storage."""
    from .calibration.host import cpu_model, fs_type, memory_dimms, read_text
    from .calibration.memory import meminfo_gib
    info = {"kernel": platform.release(), "python": platform.python_version(),
            "cpus": len(os.sched_getaffinity(0)), "mem_total_gib": round(meminfo_gib("MemTotal") or 0, 1),
            "thp": read_text("/sys/kernel/mm/transparent_hugepage/enabled")}
    model, sockets = cpu_model()
    if model:
        info.update(cpu_model=model, cpu_sockets=sockets)
    if memory_dimms():
        info["memory_dimms"] = memory_dimms()
    if not remote:
        info["storage"] = fs_type(a.model_dir)
    return info


def nothing_fits(why):
    """Leave with the status that says no test ran because the model does not fit, not because one failed."""
    print(f"nothing to run: {why}", file=sys.stderr)
    sys.exit(SKIPPED)


def select_tests(a, remote, size_gib=0.0):
    """The tests to run: --tests, else the distributed ones with --nproc, else the single-process ones
    that apply here (to host memory and, with a GPU, to one GPU; --device keeps one of the two). A
    test that cannot run here is left out with a note unless --tests asked for it."""
    if a.tests:
        names = a.tests.split(",")
    else:
        gpu = gpu_count() > 0
        names = [t for t in TESTS if is_dist(t) == bool(a.nproc) and (gpu or TESTS[t]["device"] == "cpu")]
        if a.device and not a.nproc:
            names = [n for n in names if TESTS[n]["device"] == a.device]
        if a.loaders:
            names = [n for n in names if loader_of(n) in a.loaders]
        for n in list(names):
            why = unavailable(a, n, remote) or (None if is_dist(n) else misfit(group_of(n), size_gib))
            if why:
                print(f"skipping {n}: {why}", file=sys.stderr)
                names.remove(n)
        if a.nproc and misfit("dist", size_gib, a.nproc):
            nothing_fits(misfit("dist", size_gib, a.nproc))
        if not names and a.device and misfit(a.device, size_gib):
            nothing_fits(misfit(a.device, size_gib))
    for n in names:
        if n not in TESTS:
            sys.exit(f"unknown test {n}; choose from {', '.join(TESTS)}")
        why = a.tests and (unavailable(a, n, remote) or misfit(group_of(n), size_gib, a.nproc))
        if why:
            sys.exit(f"{n} cannot run: {why}")
    if not names:
        sys.exit("no test applies here" + (f" with --device {a.device}: this machine has no GPU" if a.device else ""))
    dist = [n for n in names if is_dist(n)]
    if dist and not a.nproc:
        sys.exit(f"{dist[0]} needs --nproc N")
    if a.nproc and not dist:
        sys.exit("--nproc is for the distributed tests, which --tests leaves out")
    if a.nproc and a.nproc > gpu_count():  # each rank loads onto its own GPU; NCCL does not share one
        sys.exit(f"--nproc {a.nproc} needs {a.nproc} GPUs, one per rank; this machine has {gpu_count()}")
    return names


def launch(cmd, env, timeout):
    """(stdout, stderr, None or why it was killed). The whole process group goes on a timeout.

    The output goes to files, not pipes: a process the measured one leaves behind would hold a pipe open
    and stall the benchmark, and where signals are restricted it cannot be killed either."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, start_new_session=True)

        def kill():
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                p.kill()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass  # it could not be signalled; go on without it

        def text():
            out.seek(0)
            err.seek(0)
            return out.read(), err.read()

        try:
            p.wait(timeout=timeout)
            return (*text(), None)
        except subprocess.TimeoutExpired:
            kill()
            return (*text(), f"timed out after {timeout} s")
        except BaseException:
            kill()
            raise


def run_once(a, test, files, remote, env, tag, profile=None):
    """One measured process: (the run's record, facts about the model and the streamer). A run that gave
    no result is ({"failed": why}, None). The record holds what differs from run to run; the measured
    process's full record is in the log under --log-dir.

    An s3:// model is read from S3 by the streamer. The other loaders read local files only, so for them
    the run is the download to --scratch-dir followed by the load from there, as it is downloaded: its
    load time is the two together, and download_s says how much of it the download was."""
    warnings = []
    model_dir, download_s = a.model_dir, None
    if remote and loader_of(test) != "streamer":
        download_s, why = download(a.model_dir, a.scratch_dir, profile)
        if why:
            return {"failed": why}, None
        model_dir = a.scratch_dir
    if not remote:
        left = prime(files) if a.warm else evict(files)
        if a.warm and left < sum(os.path.getsize(f) for f in files):
            warnings.append(f"only {left / GIB:.1f} GiB in the page cache after priming")
        if left and not a.warm:
            warnings.append(f"{left / GIB:.1f} GiB still in the page cache before the run")
    args = [ENTRY, model_dir, "--child", test]
    if a.ref_dir and model_dir == a.model_dir:
        args += ["--ref-dir", a.ref_dir]
    cmd = [sys.executable]
    if is_dist(test):
        cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={a.nproc}"]
    out, err, killed = launch(cmd + args, env, TIMEOUT_S)
    if download_s is not None:
        clear(a.scratch_dir)
    if a.log_dir:
        with open(os.path.join(a.log_dir, f"{tag.replace(' ', '-')}.log"), "w") as f:
            f.write(out + "\n--- stderr ---\n" + err)
    line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
    if killed or not line:
        return {"failed": killed or (err or out)[-1500:]}, None

    raw = json.loads(line[7:])
    ranks = raw if is_dist(test) else [raw]
    load_s = max(x["load_s"] for x in ranks)  # the slowest rank
    if download_s is not None:
        load_s = round(download_s + load_s, 3)
    r = {"load_s": load_s, "gibps": round(ranks[0]["gib"] / load_s, 2),
         "cpu_s": round(sum(x["cpu_s"] for x in ranks), 2), "peak_rss_gib": max(x["peak_rss_gib"] for x in ranks)}
    if download_s is not None:
        r["download_s"] = round(download_s, 3)
    if is_dist(test):
        r["rank_s"] = [x["rank_s"] for x in ranks]
        r["peak_gpu_gib"] = max(x["peak_gpu_gib"] for x in ranks)
        if loader_of(test) == "streamer" and "[Distributed]" not in err + out:
            warnings.append("the streamer logged no [Distributed] line")
    errors = [e for x in ranks for e in x["errors"]]
    if errors:
        r["errors"] = errors
    if warnings:
        r["warnings"] = warnings
    facts = {"gib": ranks[0]["gib"], "tensors": ranks[0]["tensors"], "pkg": ranks[0]["pkg"],
             "versions": ranks[0].get("versions", {}), "gpus": ranks[0].get("gpus"),
             "fs_strategy": sorted(set(re.findall(r"Filesystem strategy: (\S+)", out + err)))}
    return r, facts


def prepare_s3(a):
    """What an s3:// model needs on this machine: a scratch directory for the loaders that read only
    local files, and a local copy of the files for the tensor names and for validation. Without
    --ref-dir that copy is downloaded once, into the scratch directory, and removed after the run.
    Returns the directory to remove, or None."""
    a.scratch_dir = a.scratch_dir or SCRATCH
    if a.ref_dir:
        return None
    if not shutil.which("s5cmd"):
        sys.exit("an s3:// model needs s5cmd on PATH to fetch a reference copy (pip install s5cmd), "
                 "or a local copy of the same files given with --ref-dir")
    from .calibration.s3 import s3_objects
    size = sum(n for _, n in s3_objects(a.model_dir))
    os.makedirs(a.scratch_dir, exist_ok=True)
    free = shutil.disk_usage(a.scratch_dir).free
    if free < 2.1 * size:  # the reference copy, and the copy each download-then-load run makes
        sys.exit(f"{a.scratch_dir} has {free / GIB:.0f} GiB free and this {size / GIB:.0f} GiB model needs twice its "
                 "size there: give --scratch-dir on a larger disk, or --ref-dir with a local copy")
    ref = os.path.join(a.scratch_dir, "ref")
    print(f"downloading a reference copy of {a.model_dir} to {ref} (--ref-dir DIR uses a local one)", flush=True)
    _, why = download(a.model_dir, ref, None)
    if why:
        sys.exit(why)
    a.ref_dir = ref
    return ref


def main(a):
    remote = is_remote(a.model_dir)
    if remote and a.warm:
        sys.exit("--warm applies to local files only")
    fetched = prepare_s3(a) if remote else None
    try:
        run(a, remote)
    finally:
        if fetched:
            clear(fetched)


def run(a, remote):
    files, ref = model_files(a)
    names = select_tests(a, remote, sum(os.path.getsize(f) for f in ref) / GIB)

    extra = {}
    for kv in a.env:
        k, sep, v = kv.partition("=")
        if not sep:
            sys.exit(f"--env {kv}: expected K=V")
        extra[k] = v
    env = dict(os.environ, RUNAI_STREAMER_LOG_LEVEL="INFO", RUNAI_STREAMER_LOG_TO_STDERR="1", **extra)
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)

    profile, profile_path = get_profile(a, remote)
    if remote and profile and not profile["ceilings_gibps"].get("s3_get_parallel"):
        print("warning: the profile has no s3_get_parallel; run calibrate.py with --s3-uri", file=sys.stderr)

    runs_of = {n: [] for n in names}
    strategy_of = {n: set() for n in names}
    environment = describe_machine(a, remote)
    busy = {}
    for name in names:
        busy.update(gpus_in_use(group_of(name), a.nproc))
    if busy:  # the numbers are then not this machine's best, and the results have to say so
        environment["gpus_in_use"] = {str(i): gib for i, gib in sorted(busy.items())}
        print("warning: something else is using " + ", ".join(f"GPU {i} ({gib} GiB)" for i, gib in sorted(busy.items()))
              + "; the GPU tests share it, and their times may be slower than on an idle GPU", file=sys.stderr)
    versions = {}
    model = {"path": a.model_dir if remote else os.path.abspath(a.model_dir), **describe(ref)}
    pkgs = set()
    for rep in range(a.repeats):  # interleaved across the tests, so drift hits all alike
        for name in names:
            tag = f"{name} run {rep}"
            r, facts = run_once(a, name, files, remote, env, tag, profile)
            runs_of[name].append(r)
            if facts is None:
                print(f"  {tag}: FAILED\n{r['failed']}", file=sys.stderr, flush=True)
                continue
            model.setdefault("gib", facts["gib"])
            model.setdefault("tensors", facts["tensors"])
            strategy_of[name].update(facts["fs_strategy"])
            if facts["pkg"]:
                pkgs.add(facts["pkg"])
            versions.update(facts["versions"])
            if facts["gpus"]:
                environment["gpus"] = facts["gpus"]
            print(f"  {tag}: {r['load_s']}s {r['gibps']} GiB/s"
                  + (f" (download {r['download_s']}s)" if "download_s" in r else "")
                  + (f" [{','.join(facts['fs_strategy'])}]" if facts["fs_strategy"] else "")
                  + (f"  INVALID: {r['errors'][0]}" if "errors" in r else ""), flush=True)
            for w in r.get("warnings", []):
                print(f"  {tag}: warning: {w}", file=sys.stderr, flush=True)

    if not report(a, runs_of, strategy_of, model, sorted(pkgs), versions, environment, extra, profile,
                  profile_path, remote):
        sys.exit(1)
