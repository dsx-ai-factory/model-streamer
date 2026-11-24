#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure how fast a model loads here, as a share of what this machine can do.

Linux only. Needs torch, safetensors and runai_model_streamer (torch with NCCL for --nproc).

    python3 bench.py /mnt/models/Qwen3-8B                  to host memory and to one GPU, with each loader
    python3 bench.py /mnt/models/Qwen3-8B --device cuda    to one GPU only
    python3 bench.py /mnt/models/Qwen3-8B --nproc 8        to the GPUs of 8 ranks on this node

The loaders are safetensors, InstantTensor and the streamer. Every run is a fresh process on a cold page
cache, and its tensors are checked against safetensors. suite.py runs several models and makes one report.
The tests, the machine profile and the results file are described in README.md next to this script.
"""
import argparse
import importlib.util
import os
import sys

from streamer_bench import DEFAULT_PROFILE, TESTS


def check_installed():
    """Say what is missing before anything runs, instead of failing inside the first measured process."""
    needed = {"torch": "torch", "safetensors": "safetensors", "runai_model_streamer": "runai-model-streamer"}
    missing = [pip for module, pip in needed.items() if importlib.util.find_spec(module) is None]
    if missing:
        sys.exit(f"bench.py needs {', '.join(missing)} in this Python ({sys.executable}):\n"
                 f"    pip install -r {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", metavar="MODEL",
                    help="directory with *.safetensors files, or s3://bucket/prefix")
    ap.add_argument("--model-dir", help=argparse.SUPPRESS)  # the older spelling of MODEL
    ap.add_argument("--nproc", type=int, help="run the distributed tests on this many ranks, one GPU each")
    ap.add_argument("--device", choices=("cpu", "cuda"),
                    help="single-process tests to this destination only (default: both)")
    ap.add_argument("--tests", default="", help=f"comma list of: {', '.join(TESTS)}")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="set for every measured process (repeatable)")
    ap.add_argument("--warm", action="store_true", help="load from the page cache instead of evicting first")
    ap.add_argument("--ref-dir", help="local copy of an s3:// model, for tensor names and validation")
    ap.add_argument("--scratch-dir", help="where the loaders that read only local files download an s3:// model "
                                          "to; their load time then includes the download")
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
    if a.nproc and a.device:
        ap.error("--device is for the single-process tests; --nproc loads to the GPUs")
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
