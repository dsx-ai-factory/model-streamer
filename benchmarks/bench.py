#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure how fast a model loads here, as a share of what this machine can do.

Linux only. Needs torch, safetensors and runai_model_streamer (torch with NCCL for --nproc).

    python3 bench.py /mnt/models/Qwen3-8B              every single-process test that applies here
    python3 bench.py /mnt/models/Qwen3-8B --nproc 8    the distributed load on 8 ranks of this node

Every run is a fresh process on a cold page cache, and its tensors are checked against safetensors.
The tests, the machine profile and the results file are described in README.md next to this script.
"""
import argparse
import importlib.util
import sys

from streamer_bench import DEFAULT_PROFILE, DIST, TESTS


def check_installed():
    """Say what is missing before anything runs, instead of failing inside the first measured process."""
    needed = {"torch": "torch", "safetensors": "safetensors", "runai_model_streamer": "runai-model-streamer"}
    missing = [pip for module, pip in needed.items() if importlib.util.find_spec(module) is None]
    if missing:
        sys.exit(f"bench.py needs {', '.join(missing)} in this Python ({sys.executable}):\n"
                 f"    pip install {' '.join(missing)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", metavar="MODEL",
                    help="directory with *.safetensors files, or s3://bucket/prefix")
    ap.add_argument("--model-dir", help=argparse.SUPPRESS)  # the older spelling of MODEL
    ap.add_argument("--nproc", type=int, help=f"ranks for {DIST}; without --tests, run only that test")
    ap.add_argument("--vllm", action="store_true",
                    help="also load the model in vLLM, with its default loader and with the streamer")
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
    ap.add_argument("--child", help=argparse.SUPPRESS)  # set for a measured process
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
    if a.child:
        from streamer_bench import worker
        worker.main(a)
    else:
        check_installed()
        from streamer_bench import runner
        runner.main(a)


if __name__ == "__main__":
    main()
