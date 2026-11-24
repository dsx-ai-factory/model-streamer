#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run bench.py on several models and make one report: load time, throughput and speedup per loader.

    python3 suite.py --gpu /models/Qwen3-32B --multi-gpu /models/DeepSeek-R1 --cpu /models/Kimi-K3

Each option is one case. Any of them can be left out, and each can be given more than once:

    --gpu MODEL          the model loaded to one GPU
    --multi-gpu MODEL    the model loaded to the GPUs of --nproc ranks (default: every GPU)
    --cpu MODEL          the model loaded to host memory

A model is a directory or an s3://bucket/prefix. Every case is one bench.py run, so the loaders, the cold
page cache, the validation and the machine profile are bench.py's. The results of each case and the
combined report.md go to --out-dir.
"""
import argparse
import os
import subprocess
import sys

from streamer_bench import SKIPPED, markdown, runner

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", metavar="MODEL", action="append", default=[], help="model to load to one GPU")
    ap.add_argument("--multi-gpu", metavar="MODEL", action="append", default=[],
                    help="model to load to the GPUs of --nproc ranks")
    ap.add_argument("--nproc", type=int, help="ranks for the multi-GPU case, one GPU each (default: every GPU)")
    ap.add_argument("--cpu", metavar="MODEL", action="append", default=[], help="model to load to host memory")
    ap.add_argument("--out-dir", default="bench_suite", help="where the results and the report go")
    ap.add_argument("--repeats", type=int, default=3, help="runs per loader")
    ap.add_argument("--loaders", help="only these loaders, a comma list as for bench.py (default: all)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="set for every measured process (repeatable)")
    ap.add_argument("--profile", help="machine profile from calibrate.py, as for bench.py")
    ap.add_argument("--ref-root", metavar="DIR", help="for s3:// models: DIR/<model name> is a local copy to use "
                                                      "for tensor names and validation (default: downloaded)")
    ap.add_argument("--scratch-dir", metavar="DIR", help="for s3:// models: where they are downloaded to")
    ap.add_argument("--no-calibrate", action="store_true", help="never calibrate")
    a = ap.parse_args()

    nproc = a.nproc or runner.gpu_count()
    cases = ([("gpu", m, ["--device", "cuda"]) for m in a.gpu]
             + [("multi_gpu", m, ["--nproc", str(nproc)]) for m in a.multi_gpu]
             + [("cpu", m, ["--device", "cpu"]) for m in a.cpu])
    if not cases:
        ap.error("give at least one of --gpu, --multi-gpu and --cpu")
    if a.multi_gpu and nproc < 1:
        ap.error("--multi-gpu needs a GPU: this machine has none")

    os.makedirs(a.out_dir, exist_ok=True)
    common = ["--repeats", str(a.repeats)] + [x for kv in a.env for x in ("--env", kv)]
    for flag, value in (("--profile", a.profile), ("--scratch-dir", a.scratch_dir), ("--loaders", a.loaders)):
        if value:
            common += [flag, value]
    if a.no_calibrate:
        common.append("--no-calibrate")
    written, failed = [], []
    for case, model, how in cases:
        name = f"{os.path.basename(model.rstrip('/'))}_{case}"
        out = os.path.join(a.out_dir, f"{name}.json")
        print(f"\n=== {name}: {model}", flush=True)
        if os.path.exists(out):
            os.remove(out)  # a case that gives no results must not report an older run's
        ref = []
        if model.startswith("s3://") and a.ref_root:
            ref = ["--ref-dir", os.path.join(a.ref_root, os.path.basename(model.rstrip("/")))]
        code = subprocess.run([sys.executable, BENCH, model, *how, *common, *ref, "--out", out,
                               "--log-dir", os.path.join(a.out_dir, f"logs_{name}")]).returncode
        if os.path.exists(out):
            written.append(out)
        if code:
            failed.append(name + (" (the model does not fit there)" if code == SKIPPED else ""))

    if written:
        report = os.path.join(a.out_dir, "report.md")
        print("\n" + markdown.combine(written, report) + f"\n\nwrote {report}")
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
