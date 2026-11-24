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

from . import ENTRY, GIB, TESTS, TIMEOUT_S, is_dist, loader_of
from .files import evict, is_remote, model_files, prime
from .profile import get_profile
from .report import report


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def unavailable(a, test, remote=False):
    """Why a test cannot run here, or None. Its loader has to be installed, and a loader that reads only
    local files needs somewhere to download an s3:// model to, and s5cmd to do it."""
    if loader_of(test) == "instanttensor" and importlib.util.find_spec("instanttensor") is None:
        return f"instanttensor is not installed in this Python ({sys.executable}): pip install instanttensor"
    if remote and loader_of(test) != "streamer":
        if not a.scratch_dir:
            return "it reads local files; give --scratch-dir DIR to download the s3:// model there first"
        if not shutil.which("s5cmd"):
            return "downloading the s3:// model needs s5cmd on PATH (pip install s5cmd)"
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
    from .calibration.host import fs_type, read_text
    from .calibration.memory import meminfo_gib
    info = {"kernel": platform.release(), "python": platform.python_version(),
            "cpus": len(os.sched_getaffinity(0)), "mem_total_gib": round(meminfo_gib("MemTotal") or 0, 1),
            "thp": read_text("/sys/kernel/mm/transparent_hugepage/enabled")}
    if not remote:
        info["storage"] = fs_type(a.model_dir)
    return info


def select_tests(a, remote):
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
            if unavailable(a, n, remote):
                print(f"skipping {n}: {unavailable(a, n, remote)}", file=sys.stderr)
                names.remove(n)
    for n in names:
        if n not in TESTS:
            sys.exit(f"unknown test {n}; choose from {', '.join(TESTS)}")
        if a.tests and unavailable(a, n, remote):
            sys.exit(f"{n} cannot run: {unavailable(a, n, remote)}")
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


def main(a):
    remote = is_remote(a.model_dir)
    if remote and a.warm:
        sys.exit("--warm applies to local files only")
    files, _ = model_files(a)
    names = select_tests(a, remote)

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
    versions = {}
    model = {"path": a.model_dir if remote else os.path.abspath(a.model_dir)}
    if a.ref_dir:
        model["ref_dir"] = a.ref_dir
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
