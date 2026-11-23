#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run bench.py on several models and make one report: load time, throughput and speedup per loader.

    python3 suite.py --gpu /models/Qwen3-32B --multi-gpu /models/DeepSeek-R1 --nproc 8 --cpu /models/Kimi-K3

Each option is one case, and any of them can be left out:

    --gpu DIR          the model loaded to one GPU
    --multi-gpu DIR    the model loaded to the GPUs of --nproc ranks
    --cpu DIR          the model loaded to host memory

Every case is one bench.py run, so the loaders, the cold page cache, the validation and the machine profile
are bench.py's. The results of each case and the combined report.md go to --out-dir.
"""
import argparse
import os
import subprocess
import sys

from streamer_bench import markdown

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", metavar="DIR", help="model to load to one GPU")
    ap.add_argument("--multi-gpu", metavar="DIR", help="model to load to the GPUs of --nproc ranks")
    ap.add_argument("--nproc", type=int, help="ranks for --multi-gpu, one GPU each")
    ap.add_argument("--cpu", metavar="DIR", help="model to load to host memory")
    ap.add_argument("--out-dir", default="bench_suite", help="where the results and the report go")
    ap.add_argument("--repeats", type=int, default=3, help="runs per loader")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="set for every measured process (repeatable)")
    ap.add_argument("--profile", help="machine profile from calibrate.py, as for bench.py")
    ap.add_argument("--no-calibrate", action="store_true", help="never calibrate")
    a = ap.parse_args()
    if a.multi_gpu and not a.nproc:
        ap.error("--multi-gpu needs --nproc N")
    cases = [(name, model, how) for name, model, how in (
        ("gpu", a.gpu, ["--device", "cuda"]),
        ("multi_gpu", a.multi_gpu, ["--nproc", str(a.nproc)]),
        ("cpu", a.cpu, ["--device", "cpu"])) if model]
    if not cases:
        ap.error("give at least one of --gpu, --multi-gpu and --cpu")

    os.makedirs(a.out_dir, exist_ok=True)
    common = ["--repeats", str(a.repeats)] + [x for kv in a.env for x in ("--env", kv)]
    if a.profile:
        common += ["--profile", a.profile]
    if a.no_calibrate:
        common.append("--no-calibrate")
    written, failed = [], []
    for name, model, how in cases:
        out = os.path.join(a.out_dir, f"{name}.json")
        print(f"\n=== {name}: {model}", flush=True)
        if os.path.exists(out):
            os.remove(out)  # a case that gives no results must not report an older run's
        code = subprocess.run([sys.executable, BENCH, model, *how, *common, "--out", out,
                               "--log-dir", os.path.join(a.out_dir, f"logs_{name}")]).returncode
        if os.path.exists(out):
            written.append(out)
        if code:
            failed.append(name)

    if written:
        report = os.path.join(a.out_dir, "report.md")
        print("\n" + markdown.combine(written, report) + f"\n\nwrote {report}")
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
