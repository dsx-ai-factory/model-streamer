# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the synthetic models (see gen_models.py) to real GPUs with the model streamer and compare every
tensor with what the generator recorded.

Launched by torchrun with two ranks, one GPU each, once per filesystem strategy:

    RUNAI_STREAMER_FS_STRATEGY=sync_buffered \\
        torchrun --standalone --nproc_per_node=2 tests/gpu/test_synthetic_models.py

Each row (see matrix.py) is one load: a model, distributed or not, a partition policy, `owned`, a destination
and a memory limit. After a load, every tensor is checked for name, dtype, shape, device and the CRC32 of its
bytes. A tensor is compared by its bytes, never by value, so NaN payloads and negative zero compare right.

    GPU_MATRIX     core (default) or full: which rows to run, see matrix.py
    GPU_FIXTURES   directory written by gen_models.py (default .gpu-fixtures)
    GPU_RESULTS    where to write timing and results (default test-results/gpu)
    GPU_ROWS       comma-separated row id prefixes: run only the rows whose id starts with one of them
    GPU_DRILL      comma-separated KIND:ROW_ID, to prove that the checks fire: the run injects a fault into that row
                   and passes only if the fault is reported (see DRILLS below). Off by default.
"""

import gc
import json
import os
import sys
import time
import zlib
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_models import MODEL_NAMES  # noqa: E402
from matrix import WORLD, memory_limit, rows_for  # noqa: E402
from phase_timer import PhaseTimer, dump_jsonl, report, stamp  # noqa: E402

from runai_model_streamer import SafetensorsStreamer  # noqa: E402
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype  # noqa: E402

MIB = 1 << 20
D2H_CHUNK = 256 * MIB
DIST_TIMEOUT_SECONDS = 300
MAX_ERRORS_PER_ROW = 40
# The CPU-only group that the ranks agree on must wait longer than the streamer's own distributed timeout: after a rank
# raised, the other one stays inside the streamer until that timeout, and only then reaches the agreement. A drill that
# makes a rank raise would otherwise end in a timeout of the agreement (the drill run of 2026-10-19 did) and not in
# a report. A drilled run also shortens the streamer's timeout, so the faulted row does not take minutes.
CONTROL_MARGIN_SECONDS = 120
DRILL_DIST_TIMEOUT_SECONDS = 30
# Bytes the allocator may hold above the start of the run after a row: rounding and small bookkeeping, not a tensor.
GPU_LEAK_SLACK = 8 * MIB

# Fault injection, to show that the checks catch what they are for. GPU_DRILL=corrupt:ROW,leak:ROW,raise:ROW makes the
# test break that row on purpose; the run then passes only if the break is reported, and fails if it goes unnoticed.
#   corrupt  flips one bit of the first tensor of the row after it arrived: the CRC32 check must report it
#   leak     keeps a 32 MiB GPU tensor alive after the row: the GPU memory check must report it
#   raise    makes rank 1 raise in the middle of the row: the row must fail on every rank, and the run must go on or
#            stop in an orderly way (a hang is cut by the timeout, and shows as a failed run)
# The value is the text that must appear in the row's errors; None means that any error of the row will do.
DRILL_MARKERS = {"corrupt": "crc32", "leak": "GPU memory leak", "raise": None}
LEAK_DRILL_BYTES = 32 * MIB


def parse_drills(text):
    drills = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        kind, _, row_id = item.partition(":")
        if kind not in DRILL_MARKERS or not row_id:
            raise RuntimeError("GPU_DRILL: %r is not KIND:ROW_ID with KIND one of %s" % (item, sorted(DRILL_MARKERS)))
        drills[row_id] = kind
    return drills


class Fixtures:
    def __init__(self, directory):
        self.directory = directory
        self.manifests = {}
        for model in MODEL_NAMES:
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


def tensor_crc32(tensor):
    """CRC32 of the tensor's bytes, copied to the host in slices so a 2 GiB tensor needs no 2 GiB buffer."""
    flat = tensor.reshape(-1).view(torch.uint8)
    crc = 0
    for start in range(0, flat.numel(), D2H_CHUNK):
        crc = zlib.crc32(flat[start:start + D2H_CHUNK].cpu().numpy(), crc)
    return crc


class RowRun:
    """One load of one model, and its verdict."""

    def __init__(self, row, fixtures, device, timer, drill=None):
        self.row = row
        self.drill = drill
        self.leaked = None
        self.drilled = False
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
        if list(tensor.shape) != entry["shape"]:
            self.error("%s: shape %s, expected %s" % (name, list(tensor.shape), entry["shape"]))
            return
        if tensor.device != self.device:
            self.error("%s: on %s, expected %s" % (name, tensor.device, self.device))
            return
        if self.drill == "corrupt" and not self.drilled and entry["end"] > entry["start"]:
            self.drilled = True
            tensor.reshape(-1).view(torch.uint8)[0] ^= 1
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
                    if self.drill == "raise" and dist.get_rank() == 1:
                        raise RuntimeError("drill: rank 1 raises in the middle of the row")
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
    rows = rows_for(os.environ.get("GPU_MATRIX", "core"))
    wanted = os.environ.get("GPU_ROWS")
    if wanted:
        prefixes = tuple(wanted.split(","))
        rows = [r for r in rows if r.id.startswith(prefixes)]
    return rows


def write_results(results_dir, strategy, rank, timer, outcomes):
    name = strategy.replace(",", "+") or "default"
    dump_jsonl(timer.records, os.path.join(results_dir, "timing-%s-rank%d.jsonl" % (name, rank)))
    with open(os.path.join(results_dir, "rows-%s-rank%d.json" % (name, rank)), "w") as handle:
        json.dump(outcomes, handle, indent=1)


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
    streamer_timeout = int(os.environ.setdefault(
        "RUNAI_STREAMER_DIST_TIMEOUT",
        str(DRILL_DIST_TIMEOUT_SECONDS if os.environ.get("GPU_DRILL") else DIST_TIMEOUT_SECONDS)))

    timer = PhaseTimer(announce=(rank == 0))
    with timer.phase("init.nccl"):
        dist.init_process_group("nccl", timeout=timedelta(seconds=DIST_TIMEOUT_SECONDS))
    with timer.phase("init.manifests"):
        fixtures = Fixtures(os.environ.get("GPU_FIXTURES", ".gpu-fixtures"))
    # The ranks agree on the outcome of a row over a CPU-only group of their own, so the test adds no NCCL traffic
    # and no GPU work next to the streamer's, and can still report after a distributed row went wrong.
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=streamer_timeout + CONTROL_MARGIN_SECONDS))
    gpu_baseline = torch.cuda.memory_allocated(device)

    rows = selected_rows()
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
                marker = DRILL_MARKERS[drill]
                local = (not ok) if marker is None else any(marker in message for message in run.errors)
                seen = torch.tensor([1 if local else 0])
                dist.all_reduce(seen, op=dist.ReduceOp.MAX, group=control)  # reported by any rank is reported
                caught = seen.item() == 1
            outcomes.append({"row": row.id, "strategy": strategy, "ok": ok, "seconds": elapsed,
                             "drill": drill, "drill_caught": caught,
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
            write_results(results_dir, strategy, rank, timer, outcomes)
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
        write_results(results_dir, strategy, rank, timer, outcomes)
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
