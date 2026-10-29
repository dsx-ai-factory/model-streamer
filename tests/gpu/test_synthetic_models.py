# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the synthetic models (see gen_models.py), and the small real model (see real_model.py), to real GPUs with the
model streamer and compare every tensor with what the manifest recorded.

A unittest module, launched by torchrun with two ranks, one GPU each, once per filesystem strategy:

    RUNAI_STREAMER_FS_STRATEGY=sync_buffered \\
        torchrun --standalone --nproc_per_node=2 tests/gpu/test_synthetic_models.py Rows

Two test cases, chosen on the command line:

    Rows         every row of the selected matrix (see matrix.py), one subtest each, so that a failing row does not stop
                 the others. A row is one load: a model, distributed or not, a partition policy, `owned`, a destination
                 and a memory limit. After a load, every tensor is checked for name, dtype, shape, device and the CRC32
                 of its bytes. A tensor is compared by its bytes, never by value, so NaN payloads and negative zero
                 compare right.
    SelfChecks   proves that the checks fire: a bit flipped in a tensor must be reported by the CRC32 check, and a GPU
                 tensor that is kept alive by the leak check.

The exit status is unittest's: a failing row, or a failing self-check, fails the run.

    GPU_MATRIX     core (default), full or smoke: which rows Rows runs, see matrix.py
    GPU_FIXTURES   directory written by gen_models.py and real_model.py (default .gpu-fixtures)
    GPU_RESULTS    where to write timing and results (default test-results/gpu)
    GPU_ROWS       comma-separated row id prefixes: run only the rows whose id starts with one of them. A filter that
                   matches no row fails the run (a typo), unless GPU_ALLOW_EMPTY_ROWS=true: a leg that has only some of
                   the rows (the oldest-torch leg has the core rows) then has nothing to run and passes, saying so
"""

import gc
import json
import os
import sys
import time
import unittest
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drills import LEAK_DRILL_BYTES, is_crc_error_of, is_leak_error  # noqa: E402
from gen_models import packed_shape  # noqa: E402
from matrix import WORLD, filter_rows, memory_limit, no_rows_outcome, rows_for, unmatched_prefixes  # noqa: E402
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

# The rows of the self-checks: the richest model, loaded to the GPU by each rank on its own, with the tensors held
# (owned) and not (borrowed).
SELF_CHECK_CORRUPT_ROW = "mixed.solo.owned.cuda.tight"
SELF_CHECK_LEAK_ROW = "mixed.solo.borrowed.cuda.tight"


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
    """One load of one model, and its errors."""

    def __init__(self, row, fixtures, device, timer, fault=None):
        """fault: a fault to inject on purpose, for the self-checks: "corrupt" flips a bit of the first tensor after it
        arrived, "leak" keeps a 32 MiB GPU tensor alive after the row."""
        self.row = row
        self.fault = fault
        self.leaked = None
        self.corrupted_tensor = None  # the tensor that the corrupt fault flipped a bit of
        self.gpu_excess = 0  # bytes that the GPU holds after the row, above what it held before the run
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

    def check(self, name, tensor, host_copy=False):
        """Name, dtype, shape, device and bytes of one tensor. host_copy: the tensor as the streamer yielded it, on the
        host, checked again after the stream (an owned tensor of a row that copies to the device)."""
        entry = self.expected.get(name)
        if entry is None:
            self.error("unexpected tensor %r" % name)
            return
        device = torch.device("cpu") if host_copy else self.device
        where = " (the tensor on the host, as yielded, after the stream)" if host_copy else ""
        if not host_copy:
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
        if tensor.device != device:
            self.error("%s: on %s, expected %s" % (name, tensor.device, device))
            return
        if not host_copy and self.fault == "corrupt" and self.corrupted_tensor is None and entry["end"] > entry["start"]:
            self.corrupted_tensor = name
            as_bytes(tensor)[0] ^= 1
        crc = tensor_crc32(tensor)
        if crc != entry["crc32"]:
            self.error("%s (%s %s, %d bytes): crc32 %08x, expected %08x%s" % (
                name, entry["dtype"], entry["shape"], entry["end"] - entry["start"], crc, entry["crc32"], where))

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
                    yielded = tensor
                    if row.device == "cpu_to":
                        tensor = tensor.to(self.device)
                    load_seconds += time.perf_counter() - start
                    if row.owned:
                        # No clone: an owned tensor must stay right while the stream goes on. On a row that copies to
                        # the device the tensor to keep is the one the streamer yielded, on the host: the copy is
                        # checked too, but it cannot show that the streamer reused the buffer of the original.
                        if name in held:
                            self.error("tensor %r yielded twice" % name)
                        held[name] = (tensor, yielded if row.device == "cpu_to" else None)
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
        for name, (tensor, on_host) in held.items():
            self.check(name, tensor)
            if on_host is not None:
                self.check(name, on_host, host_copy=True)
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


class Harness:
    """What the test cases of a run share: the two groups of ranks, the fixtures, the timer and the results of the rows."""

    def __init__(self):
        if int(os.environ.get("WORLD_SIZE", "0")) != WORLD:
            raise RuntimeError("Launch with torchrun --nproc_per_node=%d" % WORLD)
        if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
            raise RuntimeError("%d CUDA GPUs are required; GPU tests must not skip" % WORLD)
        if not dist.is_nccl_available():
            raise RuntimeError("NCCL is required")
        self.rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        self.strategy = os.environ.get("RUNAI_STREAMER_FS_STRATEGY", "")
        self.results_dir = os.environ.get("GPU_RESULTS", "test-results/gpu")
        os.makedirs(self.results_dir, exist_ok=True)
        streamer_timeout = int(os.environ.setdefault("RUNAI_STREAMER_DIST_TIMEOUT", str(DIST_TIMEOUT_SECONDS)))
        self.timer = PhaseTimer(announce=(self.rank == 0))
        with self.timer.phase("init.nccl"):
            dist.init_process_group("nccl", timeout=timedelta(seconds=DIST_TIMEOUT_SECONDS))
        # The ranks agree on the outcome of a row over a CPU-only group of their own, so the test adds no NCCL traffic
        # and no GPU work next to the streamer's, and can still report after a distributed row went wrong.
        self.control = dist.new_group(backend="gloo",
                                      timeout=timedelta(seconds=streamer_timeout + CONTROL_MARGIN_SECONDS))
        self.gpu_baseline = torch.cuda.memory_allocated(self.device)
        self.fixtures = None
        self.outcomes = []
        self.expected_rows = 0

    def load_fixtures(self, models):
        with self.timer.phase("init.manifests"):
            self.fixtures = Fixtures(os.environ.get("GPU_FIXTURES", ".gpu-fixtures"), models)

    def write(self):
        write_results(self.results_dir, self.strategy, self.rank, self.timer, self.outcomes, self.expected_rows)

    def row_run(self, row, fault=None):
        """Load one row on both ranks, check it, and agree on the outcome. Returns (run, any_raised): run.errors are
        the errors of this rank, and any_raised is True when a rank raised, which may leave the collectives out of step."""
        dist.barrier(group=self.control)
        run = RowRun(row, self.fixtures, self.device, self.timer, fault)
        if self.rank == 0:
            stamp("ROW %s  model %s  %s  owned=%s  device=%s  limit=%s  strategy=%s" % (
                row.id, row.model, ("distributed " + row.policy) if row.dist else "not distributed",
                row.owned, row.device, row.limit, self.strategy))
        outcome = {"row": row.id, "strategy": self.strategy, "ok": None}
        self.outcomes.append(outcome)
        self.write()
        started = time.perf_counter()
        raised = False
        try:
            run.run()
        except Exception as error:  # a streamer exception is a failure of the row, not of the harness
            raised = True
            run.error("%s: %s" % (type(error).__name__, error))
        elapsed = time.perf_counter() - started
        if run.fault == "leak":
            run.leaked = torch.empty(LEAK_DRILL_BYTES, dtype=torch.uint8, device=self.device)
        release_gpu_memory()
        # Everything the row allocated on the GPU is gone by now: tensors held, copies, the streamer's. What is still
        # allocated is a leak of the streamer or of the test.
        gpu_allocated = torch.cuda.memory_allocated(self.device)
        run.gpu_excess = gpu_allocated - self.gpu_baseline
        if not raised and gpu_allocated > self.gpu_baseline + GPU_LEAK_SLACK:
            run.error("GPU memory leak: %.1f MiB still allocated after the row, %.1f MiB before the run" % (
                gpu_allocated / MIB, self.gpu_baseline / MIB))
        # Every rank must agree before the next row, or one rank enters a collective alone and hangs. The second
        # element is 1 when a rank raised: its collectives may be out of step.
        flag = torch.tensor([1 if run.errors else 0, 1 if raised else 0])
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.control)
        ok = flag[0].item() == 0
        any_raised = flag[1].item() == 1
        outcome.update({"ok": ok, "seconds": elapsed, "fault": run.fault, "gpu_allocated_bytes": gpu_allocated,
                        "errors": run.errors_with_total()})
        print("[rank %d] %s %-26s %-12s %s" % (self.rank, "PASS" if ok else "FAIL", row.id, self.strategy,
                                              "%.1fs" % elapsed), flush=True)
        for message in run.errors_with_total():
            print("[rank %d]   %s" % (self.rank, message), flush=True)
        # After every row, not only at the end: a hang that the NCCL watchdog turns into a killed process would
        # otherwise take the results of the rows that already ran with it.
        self.write()
        return run, any_raised

    def finish(self):
        self.write()
        if self.rank == 0:
            print(report(self.timer.records, top=15), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()


HARNESS = None


def setUpModule():
    global HARNESS
    HARNESS = Harness()


def tearDownModule():
    if HARNESS is not None:
        HARNESS.finish()


class Rows(unittest.TestCase):
    """Every row of the selected matrix. A subtest each: a failing row is recorded with its id and errors, and the next
    row runs, so one run reports every failing row; the exit status is unittest's."""

    def test_every_row(self):
        rows = selected_rows()
        HARNESS.expected_rows = len(rows)
        # A leg that has just some of the rows (GPU_ALLOW_EMPTY_ROWS) is given prefixes that it does not all have.
        unmatched = unmatched_prefixes(rows_for(os.environ.get("GPU_MATRIX", "core")), os.environ.get("GPU_ROWS"))
        if unmatched and os.environ.get("GPU_ALLOW_EMPTY_ROWS") != "true":
            self.fail("no row of the %s matrix starts with %s: a typo in the filter?" % (
                os.environ.get("GPU_MATRIX", "core"), unmatched))
        if not rows:
            status, message = no_rows_outcome(os.environ.get("GPU_ROWS"),
                                              os.environ.get("GPU_ALLOW_EMPTY_ROWS") == "true",
                                              os.environ.get("GPU_MATRIX", "core"))
            print(message, flush=True)
            if status:
                self.fail(message)
            return
        if HARNESS.rank == 0:
            stamp("matrix %s: %d rows, strategy %s" % (os.environ.get("GPU_MATRIX", "core"), len(rows),
                                                      HARNESS.strategy))
        HARNESS.load_fixtures(sorted({r.model for r in rows}))
        for row in rows:
            run, any_raised = HARNESS.row_run(row)
            with self.subTest(row=row.id):
                if run.errors:
                    self.fail("\n".join(run.errors_with_total()))
            if any_raised and row.dist:
                # A row that only found wrong tensors ended in step on every rank, so the next row can run. After an
                # exception in a distributed row the collectives may be out of step, and the next row could hang.
                print("[rank %d] a rank raised in row %s: not running the rows after it" % (HARNESS.rank, row.id),
                      flush=True)
                self.fail("a rank raised in row %s: the rows after it were not run" % row.id)


class SelfChecks(unittest.TestCase):
    """The checks of Rows must fire: a fault is injected on purpose and has to be reported, as itself and with no other
    error. A failed assertion here fails the run like any other test: nothing is declared as an expected failure."""

    @classmethod
    def setUpClass(cls):
        HARNESS.expected_rows = 2
        HARNESS.load_fixtures(["mixed"])
        cls.rows = {row.id: row for row in rows_for("full")}

    def test_a_corrupted_tensor_is_reported_by_the_crc32_check(self):
        run, _ = HARNESS.row_run(self.rows[SELF_CHECK_CORRUPT_ROW], fault="corrupt")
        self.assertIsNotNone(run.corrupted_tensor, "no tensor was corrupted: the fault was not injected")
        other = [e for e in run.errors_with_total() if not is_crc_error_of(e, run.corrupted_tensor)]
        self.assertEqual(other, [], "the row has errors that are not the report of the corrupted tensor")
        self.assertTrue(any(is_crc_error_of(e, run.corrupted_tensor) for e in run.errors),
                        "the CRC32 check did not report %s" % run.corrupted_tensor)

    def test_a_leaked_gpu_tensor_is_reported_by_the_memory_check(self):
        run, _ = HARNESS.row_run(self.rows[SELF_CHECK_LEAK_ROW], fault="leak")
        self.assertGreaterEqual(run.gpu_excess, LEAK_DRILL_BYTES, "the leaked tensor is not held: the fault was not injected")
        other = [e for e in run.errors_with_total() if not is_leak_error(e)]
        self.assertEqual(other, [], "the row has errors that are not the report of the leak")
        self.assertTrue(any(is_leak_error(e) for e in run.errors), "the GPU memory check did not report the leak")


if __name__ == "__main__":
    unittest.main()
