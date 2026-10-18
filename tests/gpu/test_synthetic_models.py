# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the synthetic models (see gen_models.py) to real GPUs with the model streamer and compare every
tensor with what the generator recorded.

Launched by torchrun with two ranks, one GPU each, once per filesystem strategy:

    RUNAI_STREAMER_FS_STRATEGY=sync_buffered \\
        torchrun --standalone --nproc_per_node=2 tests/gpu/test_synthetic_models.py

Each row of ROWS is one load: a model, distributed or not, a partition policy, `owned`, a destination and a
memory limit. After a load, every tensor is checked for name, dtype, shape, device and the CRC32 of its
bytes. A tensor is compared by its bytes, never by value, so NaN payloads and negative zero compare right.

    GPU_FIXTURES   directory written by gen_models.py (default .gpu-fixtures)
    GPU_RESULTS    where to write timing and results (default test-results/gpu)
    GPU_ROWS       comma-separated row ids to run (default all)
"""

import gc
import json
import os
import sys
import time
import zlib
from collections import namedtuple
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_timer import PhaseTimer, dump_jsonl, report, stamp  # noqa: E402

from runai_model_streamer import SafetensorsStreamer  # noqa: E402
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype  # noqa: E402

MIB = 1 << 20
WORLD = 2
D2H_CHUNK = 256 * MIB
DIST_TIMEOUT_SECONDS = 300
MAX_ERRORS_PER_ROW = 20

# dist: None = not distributed. device: "cuda" streams straight to the rank's GPU, "cpu_to" streams to host
# memory and copies each tensor to the GPU. limit: "tight" forces the host ring to be reused many times,
# "default" leaves RUNAI_STREAMER_MEMORY_LIMIT unset.
Row = namedtuple("Row", "id model dist policy owned device limit strategies")

# Model C holds the 2 GiB tensor, so it runs under two strategies only: its load dominates the time.
HEAVY_STRATEGIES = ("sync_buffered", "io_uring_direct,sync_buffered")

ROWS = [
    Row("A1", "A", False, None, True, "cuda", "tight", None),
    Row("A2", "A", False, None, False, "cuda", "tight", None),
    Row("A3", "A", True, "spans", True, "cuda", "tight", None),
    Row("A4", "A", True, "chunks", True, "cuda", "tight", None),
    Row("A5", "A", False, None, True, "cpu_to", "tight", None),
    Row("A6", "A", False, None, True, "cuda", "default", None),
    Row("B1", "B", False, None, True, "cuda", "tight", None),
    Row("B2", "B", True, "spans", True, "cuda", "tight", None),
    Row("C1", "C", False, None, True, "cuda", "tight", HEAVY_STRATEGIES),
    Row("C2", "C", True, "spans", True, "cuda", "tight", HEAVY_STRATEGIES),
    Row("D1", "D", False, None, True, "cuda", "tight", None),
    Row("D2", "D", True, "files", True, "cuda", "tight", None),
    Row("D3", "D", True, "spans", True, "cuda", "tight", None),
]


class Fixtures:
    def __init__(self, directory):
        self.directory = directory
        self.manifests = {}
        for model in "ABCD":
            with open(os.path.join(directory, model + ".manifest.json")) as handle:
                self.manifests[model] = json.load(handle)

    def paths(self, model):
        return [os.path.join(self.directory, model, f["path"]) for f in self.manifests[model]["files"]]

    def expected(self, model):
        """name -> manifest entry, across all files of the model."""
        return {t["name"]: t for f in self.manifests[model]["files"] for t in f["tensors"]}


def memory_limit(manifest, distributed):
    """About a sixteenth of the model, so the host ring is reused many times - but never below the largest
    tensor times the ranks sharing the limit, because a tensor must fit in one buffer and a distributed
    rank only gets its share (times 1.5 so there is room for a second buffer)."""
    ranks = WORLD if distributed else 1
    floor = int(1.5 * manifest["largest_tensor_bytes"]) * ranks
    return max(manifest["total_data_bytes"] // 16, floor, MIB)


@contextmanager
def environment(settings):
    """Set environment variables for a block; a value of None removes the variable."""
    saved = {key: os.environ.get(key) for key in settings}
    for key, value in settings.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def tensor_crc32(tensor):
    """CRC32 of the tensor's bytes, copied to the host in slices so a 2 GiB tensor needs no 2 GiB buffer."""
    flat = tensor.reshape(-1).view(torch.uint8)
    crc = 0
    for start in range(0, flat.numel(), D2H_CHUNK):
        crc = zlib.crc32(flat[start:start + D2H_CHUNK].cpu().numpy(), crc)
    return crc


class RowRun:
    """One load of one model, and its verdict."""

    def __init__(self, row, fixtures, device, timer):
        self.row = row
        self.expected = fixtures.expected(row.model)
        self.paths = fixtures.paths(row.model)
        self.manifest = fixtures.manifests[row.model]
        self.device = device
        self.timer = timer
        self.errors = []
        self.seen = set()

    def error(self, message):
        if len(self.errors) < MAX_ERRORS_PER_ROW:
            self.errors.append(message)

    def check(self, name, tensor):
        """Name, dtype, shape, device and bytes of one tensor."""
        entry = self.expected.get(name)
        if entry is None:
            self.error("unexpected tensor %r" % name)
            return
        if name in self.seen:
            self.error("tensor %r yielded twice" % name)
        self.seen.add(name)
        want_dtype = safetensors_to_torch_dtype[entry["dtype"]]
        if tensor.dtype != want_dtype:
            self.error("%s: dtype %s, expected %s" % (name, tensor.dtype, want_dtype))
            return
        if list(tensor.shape) != entry["shape"]:
            self.error("%s: shape %s, expected %s" % (name, list(tensor.shape), entry["shape"]))
            return
        if tensor.device != self.device:
            self.error("%s: on %s, expected %s" % (name, tensor.device, self.device))
            return
        crc = tensor_crc32(tensor)
        if crc != entry["crc32"]:
            self.error("%s (%s %s, %d bytes): crc32 %08x, expected %08x" % (
                name, entry["dtype"], entry["shape"], entry["end"] - entry["start"], crc, entry["crc32"]))

    def run(self):
        row = self.row
        settings = {
            "RUNAI_STREAMER_PARTITION_POLICY": row.policy or "spans",
            "RUNAI_STREAMER_DIST": "auto",
            "RUNAI_STREAMER_MEMORY_LIMIT": (
                None if row.limit == "default" else str(memory_limit(self.manifest, bool(row.dist)))),
        }
        destination = "cpu" if row.device == "cpu_to" else str(self.device)
        held = {}
        load_seconds = 0.0
        tag = row.id
        with environment(settings):
            streamer = SafetensorsStreamer()
            with self.timer.phase("%s.setup" % tag):
                streamer.__enter__()
                streamer.stream_files(self.paths, device=destination, is_distributed=bool(row.dist),
                                      owned=row.owned)
            try:
                if self.timer.announce:
                    stamp("START %s.load" % tag)
                if streamer.file_streamer.is_distributed != bool(row.dist):
                    self.error("the streamer silently ran %s, the row asked for %s" % (
                        "distributed" if streamer.file_streamer.is_distributed else "not distributed",
                        "distributed" if row.dist else "not distributed"))
                iterator = iter(streamer.get_tensors())
                while True:
                    start = time.perf_counter()
                    try:
                        name, tensor = next(iterator)
                    except StopIteration:
                        load_seconds += time.perf_counter() - start
                        break
                    if row.device == "cpu_to":
                        tensor = tensor.to(self.device)
                    load_seconds += time.perf_counter() - start
                    if row.owned:
                        # No clone: an owned tensor must stay right while the stream goes on.
                        if name in held:
                            self.error("tensor %r yielded twice" % name)
                        held[name] = tensor
                    else:
                        # A borrowed tensor is only good until the iteration advances.
                        with self.timer.phase("%s.verify" % tag, nbytes=tensor.numel() * tensor.element_size(),
                                              announce=False):
                            self.check(name, tensor)
                start = time.perf_counter()
                torch.cuda.synchronize(self.device)
                load_seconds += time.perf_counter() - start
                if self.timer.announce:
                    stamp("END   %s.load  %.1fs" % (tag, load_seconds))
            finally:
                with self.timer.phase("%s.teardown" % tag):
                    streamer.__exit__(None, None, None)
        self.timer.records.append({"phase": "%s.load" % tag, "seconds": load_seconds,
                                   "bytes": self.manifest["total_data_bytes"]})
        if self.timer.announce and held:
            stamp("START %s.verify (%d tensors)" % (tag, len(held)))
        verify_start = time.perf_counter()
        for name, tensor in held.items():
            with self.timer.phase("%s.verify" % tag, nbytes=tensor.numel() * tensor.element_size(),
                                  announce=False):
                self.check(name, tensor)
        if self.timer.announce and held:
            stamp("END   %s.verify  %.1fs" % (tag, time.perf_counter() - verify_start))
        missing = set(self.expected) - self.seen
        if missing:
            self.error("%d tensor(s) never arrived, for example %r" % (len(missing), sorted(missing)[0]))
        held.clear()


def release_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def selected_rows(strategy):
    wanted = os.environ.get("GPU_ROWS")
    rows = [r for r in ROWS if r.strategies is None or strategy in r.strategies]
    if wanted:
        ids = set(wanted.split(","))
        rows = [r for r in rows if r.id in ids]
    return rows


def main():
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD:
        raise RuntimeError("Launch with torchrun --nproc_per_node=%d" % WORLD)
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        raise RuntimeError("%d CUDA GPUs are required; GPU tests must not skip" % WORLD)
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL is required")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    strategy = os.environ.get("RUNAI_STREAMER_FS_STRATEGY", "")
    results_dir = os.environ.get("GPU_RESULTS", "test-results/gpu")
    os.makedirs(results_dir, exist_ok=True)
    os.environ.setdefault("RUNAI_STREAMER_DIST_TIMEOUT", str(DIST_TIMEOUT_SECONDS))

    timer = PhaseTimer(announce=(rank == 0))
    with timer.phase("init.nccl"):
        dist.init_process_group("nccl", timeout=timedelta(seconds=DIST_TIMEOUT_SECONDS))
    with timer.phase("init.manifests"):
        fixtures = Fixtures(os.environ.get("GPU_FIXTURES", ".gpu-fixtures"))

    rows = selected_rows(strategy)
    outcomes = []
    failed = False
    try:
        for row in rows:
            dist.barrier()
            run = RowRun(row, fixtures, device, timer)
            if rank == 0:
                stamp("ROW %s  model %s  %s  owned=%s  device=%s  limit=%s  strategy=%s" % (
                    row.id, row.model, ("distributed " + row.policy) if row.dist else "not distributed",
                    row.owned, row.device, row.limit, strategy))
            started = time.perf_counter()
            try:
                run.run()
            except Exception as error:  # a streamer exception is a failure of the row, not of the harness
                run.error("%s: %s" % (type(error).__name__, error))
            elapsed = time.perf_counter() - started
            # Every rank must agree before the next row, or one rank enters a collective alone and hangs.
            flag = torch.tensor([1 if run.errors else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            ok = flag.item() == 0
            outcomes.append({"row": row.id, "strategy": strategy, "ok": ok, "seconds": elapsed,
                             "errors": run.errors})
            print("[rank %d] %s %-3s %-12s %s" % (rank, "PASS" if ok else "FAIL", row.id, strategy,
                                                  "%.1fs" % elapsed), flush=True)
            for message in run.errors:
                print("[rank %d]   %s" % (rank, message), flush=True)
            release_gpu_memory()
            if not ok:
                failed = True
                break  # the collectives of a failed row may be out of step; do not run on
    finally:
        dump_jsonl(timer.records, os.path.join(results_dir, "timing-%s-rank%d.jsonl" % (
            strategy.replace(",", "+") or "default", rank)))
        with open(os.path.join(results_dir, "rows-%s-rank%d.json" % (
                strategy.replace(",", "+") or "default", rank)), "w") as handle:
            json.dump(outcomes, handle, indent=1)
        if rank == 0:
            print(report(timer.records, top=15), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
    if not outcomes:
        print("no rows selected", file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
