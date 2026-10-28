# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the synthetic models (see gen_models.py), and the small real model (see real_model.py), to real GPUs with the
model streamer and compare every tensor with what the manifest recorded.

Launched by torchrun with two ranks, one GPU each, once per filesystem strategy:

    RUNAI_STREAMER_FS_STRATEGY=sync_buffered \\
        torchrun --standalone --nproc_per_node=2 tests/gpu/test_synthetic_models.py

Each row (see matrix.py) is one load: a model, distributed or not, a partition policy, `owned`, a destination
and a memory limit. After a load, every tensor is checked for name, dtype, shape, device and the CRC32 of its
bytes. A tensor is compared by its bytes, never by value, so NaN payloads and negative zero compare right.

    GPU_MATRIX     core (default), full or smoke: which rows to run, see matrix.py
    GPU_FIXTURES   directory written by gen_models.py and real_model.py (default .gpu-fixtures)
    GPU_RESULTS    where to write timing and results (default test-results/gpu)
    GPU_ROWS       comma-separated row id prefixes: run only the rows whose id starts with one of them. A leg with no
                   such row has nothing to run and passes, saying so
    GPU_DRILL      comma-separated KIND:ROW_ID, to prove that the checks fire: the run injects a fault into that row
                   and passes only if the fault is reported (see drills.py). Off by default.
"""

import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drills import LEAK_DRILL_BYTES, drill_caught, parse_drills  # noqa: E402
from gen_models import packed_shape  # noqa: E402
from matrix import WORLD, filter_rows, memory_limit, row_prefixes, rows_for  # noqa: E402
from phase_timer import PhaseTimer, dump_jsonl, report, stamp  # noqa: E402
from tensor_bytes import as_bytes, tensor_crc32  # noqa: E402

from runai_model_streamer import SafetensorsStreamer  # noqa: E402
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype  # noqa: E402

MIB = 1 << 20
DIST_TIMEOUT_SECONDS = 300
MAX_ERRORS_PER_ROW = 40
# The CPU-only group that the ranks agree on waits longer than the streamer's own distributed timeout, so that a rank
# that is only slow is never mistaken for a failed one. (When a rank does fail inside a distributed row, the other one
# is aborted by the NCCL watchdog after the streamer's timeout, and the whole run ends: see the README.)
CONTROL_MARGIN_SECONDS = 120
# Bytes the allocator may hold above the start of the run after a row: rounding and small bookkeeping, not a tensor.
GPU_LEAK_SLACK = 8 * MIB

class Fixtures:
    """The manifests of the models that the selected rows load."""

    def __init__(self, directory, models):
        self.directory = directory
        self.manifests = {}
        for model in models:
            with open(os.path.join(directory, model + ".manifest.json")) as handle:
                self.manifests[model] = json.load(handle)

    def paths(self, model):
        return [os.path.join(self.directory, model, f["path"]) for f in self.manifests[model]["files"]]

    def expected(self, model):
        """name -> manifest entry, across all files of the model."""
        return {t["name"]: t for f in self.manifests[model]["files"] for t in f["tensors"]}


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


class RowRun:
    """One load of one model, and its verdict."""

    def __init__(self, row, fixtures, device, timer, drill=None):
        self.row = row
        self.drill = drill
        self.leaked = None
        self.corrupted_tensor = None  # the tensor that the corrupt drill flipped a bit of
        self.expected = fixtures.expected(row.model)
        self.paths = fixtures.paths(row.model)
        self.manifest = fixtures.manifests[row.model]
        self.device = device
        self.timer = timer
        self.errors = []
        self.error_count = 0
        self.seen = set()

    def error(self, message):
        self.error_count += 1
        if len(self.errors) < MAX_ERRORS_PER_ROW:
            self.errors.append(message)

    def errors_with_total(self):
        """The recorded errors, and a last line saying how many more there were."""
        more = self.error_count - len(self.errors)
        return self.errors + (["... and %d more error(s)" % more] if more > 0 else [])

    def check(self, name, tensor):
        """Name, dtype, shape, device and bytes of one tensor."""
        entry = self.expected.get(name)
        if entry is None:
            self.error("unexpected tensor %r" % name)
            return
        if name in self.seen:
            self.error("tensor %r yielded twice" % name)
        self.seen.add(name)
        # The expected dtype comes from the streamer's own table, so this checks that the dtype survives the copy to
        # the device, not that the table is right: that is checked against safetensors in the CPU tests (issue #267).
        want_dtype = safetensors_to_torch_dtype[entry["dtype"]]
        if tensor.dtype != want_dtype:
            self.error("%s: dtype %s, expected %s" % (name, tensor.dtype, want_dtype))
            return
        # The manifest has the shape of the header, in values; the torch shape of a packed dtype (F4) has the last
        # dimension halved.
        want_shape = packed_shape(entry["dtype"], entry["shape"])
        if list(tensor.shape) != want_shape:
            self.error("%s: shape %s, expected %s" % (name, list(tensor.shape), want_shape))
            return
        if tensor.device != self.device:
            self.error("%s: on %s, expected %s" % (name, tensor.device, self.device))
            return
        if self.drill == "corrupt" and self.corrupted_tensor is None and entry["end"] > entry["start"]:
            self.corrupted_tensor = name
            as_bytes(tensor)[0] ^= 1
        crc = tensor_crc32(tensor)
        if crc != entry["crc32"]:
            self.error("%s (%s %s, %d bytes): crc32 %08x, expected %08x" % (
                name, entry["dtype"], entry["shape"], entry["end"] - entry["start"], crc, entry["crc32"]))

    def run(self):
        row = self.row
        limit = memory_limit(self.manifest, bool(row.dist), row.limit)
        settings = {
            "RUNAI_STREAMER_PARTITION_POLICY": row.policy or "spans",
            "RUNAI_STREAMER_DIST": "auto",
            "RUNAI_STREAMER_MEMORY_LIMIT": None if limit is None else str(limit),
        }
        destination = "cpu" if row.device == "cpu_to" else str(self.device)
        held = {}
        load_seconds = 0.0
        verify_seconds = 0.0
        tag = row.id
        with environment(settings):
            # A real `with`, so the streamer's __exit__ gets the exception of a failed row: the distributed streamer
            # skips its closing barrier then, which the other rank would not reach.
            with SafetensorsStreamer() as streamer:
                with self.timer.phase("%s.setup" % tag):
                    streamer.stream_files(self.paths, device=destination, is_distributed=bool(row.dist),
                                          owned=row.owned)
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
                        check_start = time.perf_counter()
                        self.check(name, tensor)
                        verify_seconds += time.perf_counter() - check_start
                start = time.perf_counter()
                torch.cuda.synchronize(self.device)
                load_seconds += time.perf_counter() - start
                if self.timer.announce:
                    stamp("END   %s.load  %.1fs" % (tag, load_seconds))
        self.timer.records.append({"phase": "%s.load" % tag, "seconds": load_seconds,
                                   "bytes": self.manifest["total_data_bytes"]})
        if self.timer.announce and held:
            stamp("START %s.verify (%d tensors)" % (tag, len(held)))
        verify_start = time.perf_counter()
        for name, tensor in held.items():
            self.check(name, tensor)
        verify_seconds += time.perf_counter() - verify_start
        if self.timer.announce and held:
            stamp("END   %s.verify  %.1fs" % (tag, time.perf_counter() - verify_start))
        # One record per row: a record per tensor was millions of records, and the file rewritten after every row
        # took about 8 minutes of a run.
        self.timer.records.append({"phase": "%s.verify" % tag, "seconds": verify_seconds,
                                   "bytes": self.manifest["total_data_bytes"]})
        missing = set(self.expected) - self.seen
        if missing:
            self.error("%d tensor(s) never arrived, for example %r" % (len(missing), sorted(missing)[0]))
        held.clear()


def release_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def selected_rows():
    return filter_rows(rows_for(os.environ.get("GPU_MATRIX", "core")), os.environ.get("GPU_ROWS"))


def write_results(results_dir, strategy, rank, timer, outcomes, expected_rows):
    """The rows of this run so far. A row is listed with "ok": null from the moment it starts, so that if the process
    is killed in the middle of it (the NCCL watchdog does that when a rank fails inside a distributed row), the file
    names that row. expected_rows lets a reader see how many rows never started."""
    name = strategy.replace(",", "+") or "default"
    dump_jsonl(timer.records, os.path.join(results_dir, "timing-%s-rank%d.jsonl" % (name, rank)))
    with open(os.path.join(results_dir, "rows-%s-rank%d.json" % (name, rank)), "w") as handle:
        json.dump({"strategy": strategy, "expected_rows": expected_rows, "rows": outcomes}, handle, indent=1)


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
    streamer_timeout = int(os.environ.setdefault("RUNAI_STREAMER_DIST_TIMEOUT", str(DIST_TIMEOUT_SECONDS)))

    rows = selected_rows()
    timer = PhaseTimer(announce=(rank == 0))
    with timer.phase("init.nccl"):
        dist.init_process_group("nccl", timeout=timedelta(seconds=DIST_TIMEOUT_SECONDS))
    with timer.phase("init.manifests"):
        fixtures = Fixtures(os.environ.get("GPU_FIXTURES", ".gpu-fixtures"), sorted({r.model for r in rows}))
    # The ranks agree on the outcome of a row over a CPU-only group of their own, so the test adds no NCCL traffic
    # and no GPU work next to the streamer's, and can still report after a distributed row went wrong.
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=streamer_timeout + CONTROL_MARGIN_SECONDS))
    gpu_baseline = torch.cuda.memory_allocated(device)

    drills = parse_drills(os.environ.get("GPU_DRILL", ""))
    unknown = sorted(set(drills) - {r.id for r in rows})
    if unknown:
        raise RuntimeError("GPU_DRILL names rows that are not selected: %s" % unknown)
    if rank == 0:
        stamp("matrix %s: %d rows, strategy %s" % (os.environ.get("GPU_MATRIX", "core"), len(rows), strategy))
        if drills:
            stamp("DRILL: faults are injected on purpose into %s; the run passes only if all of them are reported" % drills)
    outcomes = []
    failed = False
    try:
        for row in rows:
            dist.barrier(group=control)
            drill = drills.get(row.id)
            run = RowRun(row, fixtures, device, timer, drill)
            if rank == 0:
                stamp("ROW %s  model %s  %s  owned=%s  device=%s  limit=%s  strategy=%s" % (
                    row.id, row.model, ("distributed " + row.policy) if row.dist else "not distributed",
                    row.owned, row.device, row.limit, strategy))
            outcome = {"row": row.id, "strategy": strategy, "ok": None}
            outcomes.append(outcome)
            write_results(results_dir, strategy, rank, timer, outcomes, len(rows))
            started = time.perf_counter()
            raised = False
            try:
                run.run()
            except Exception as error:  # a streamer exception is a failure of the row, not of the harness
                raised = True
                run.error("%s: %s" % (type(error).__name__, error))
            elapsed = time.perf_counter() - started
            if drill == "leak":
                run.leaked = torch.empty(LEAK_DRILL_BYTES, dtype=torch.uint8, device=device)
            release_gpu_memory()
            # Everything the row allocated on the GPU is gone by now: tensors held, copies, the streamer's. What is
            # still allocated is a leak of the streamer or of the test.
            gpu_allocated = torch.cuda.memory_allocated(device)
            if not raised and gpu_allocated > gpu_baseline + GPU_LEAK_SLACK:
                run.error("GPU memory leak: %.1f MiB still allocated after the row, %.1f MiB before the run" % (
                    gpu_allocated / MIB, gpu_baseline / MIB))
            # Every rank must agree before the next row, or one rank enters a collective alone and hangs.
            # The second element is 1 when a rank raised: its collectives may be out of step.
            flag = torch.tensor([1 if run.errors else 0, 1 if raised else 0])
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=control)
            ok = flag[0].item() == 0
            any_raised = flag[1].item() == 1
            # A drilled row must fail, and with the report that its fault is meant to cause.
            caught = None
            if drill:
                local = drill_caught(drill, run.errors, run.corrupted_tensor, gpu_allocated - gpu_baseline)
                seen = torch.tensor([1 if local else 0])
                dist.all_reduce(seen, op=dist.ReduceOp.MAX, group=control)  # reported by any rank is reported
                caught = seen.item() == 1
            outcome.update({"ok": ok, "seconds": elapsed, "drill": drill, "drill_caught": caught,
                            "gpu_allocated_bytes": gpu_allocated, "errors": run.errors_with_total()})
            print("[rank %d] %s %-26s %-12s %s" % (rank, "PASS" if ok else "FAIL", row.id, strategy,
                                                  "%.1fs" % elapsed), flush=True)
            for message in run.errors_with_total():
                print("[rank %d]   %s" % (rank, message), flush=True)
            if drill:
                print("[rank %d] DRILL %s on %s: %s" % (rank, drill, row.id,
                                                         "REPORTED, as it must be" if caught else "NOT REPORTED"),
                      flush=True)
            # After every row, not only at the end: a hang that the NCCL watchdog turns into a killed process
            # would otherwise take the results of the rows that already ran with it.
            write_results(results_dir, strategy, rank, timer, outcomes, len(rows))
            if drill and not caught:
                failed = True  # the fault went unnoticed: a check does not do its job
            if not ok:
                failed = failed or not drill
                # A row that only found wrong tensors ended in step on every rank, so the next row can run and
                # one run reports every failing row. So does a row that raised without distributed streaming,
                # which has no collectives. After an exception in a distributed row they may be out of step.
                if any_raised and row.dist:
                    print("[rank %d] a rank raised in row %s: not running the rows after it" % (rank, row.id),
                          flush=True)
                    break
    finally:
        write_results(results_dir, strategy, rank, timer, outcomes, len(rows))
        if rank == 0:
            print(report(timer.records, top=15), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
    if not outcomes:
        if row_prefixes(os.environ.get("GPU_ROWS")):
            # A filter was asked for, and this matrix has no such row (the oldest-torch leg runs only the core rows):
            # there is nothing to run here, which is not a failure.
            print("no row of the %s matrix starts with %s: nothing to run" % (
                os.environ.get("GPU_MATRIX", "core"), list(row_prefixes(os.environ["GPU_ROWS"]))), flush=True)
            return 0
        print("no rows selected", file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
