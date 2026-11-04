# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the synthetic models (see gen_models.py), and the small real model (see real_model.py), to real GPUs with the
model streamer and compare every tensor with what the manifest recorded.

A unittest module, launched by torchrun with two ranks, one GPU each, once per filesystem strategy (suite.sh does it):

    RUNAI_STREAMER_FS_STRATEGY=sync_buffered \\
        torchrun --standalone --nproc_per_node=2 tests/gpu/suites/streaming/test_synthetic_models.py Rows

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
import resource
import sys
import time
import unittest
from contextlib import contextmanager

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "harness"))  # gpu_harness: what every suite shares
from drills import LEAK_DRILL_BYTES, is_crc_error_of, is_leak_error  # noqa: E402
from gen_models import packed_shape  # noqa: E402
from gpu_harness.phase_timer import stamp  # noqa: E402
from gpu_harness.session import Session  # noqa: E402
from gpu_harness.tensors import as_bytes, tensor_crc32  # noqa: E402
from matrix import (filter_rows, gpu_world, memory_limit, no_rows_outcome, rows_for, skipped_rows,  # noqa: E402
                    unmatched_prefixes)

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
# The soft limit of open files that the test process asks for. The streamer keeps a file open until its submission ends
# (https://github.com/dsx-ai-factory/model-streamer/issues/261), so the many_tensors model (one file, about 100,000
# tensors) in a distributed row needs far more than the container's 1024. The limit is raised here, in the process of
# the test, and nowhere else. The hard limit caps it.
OPEN_FILES_WANTED = 262144


def raise_open_files_limit(wanted=OPEN_FILES_WANTED):
    """Raise the soft limit of open files of this process to wanted, or to the hard limit if that is lower. Returns
    (soft limit before, soft limit now)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        return soft, target
    return soft, soft

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
        # From the manifest alone, so both ranks get the same limit, or the same refusal, before any collective: a model
        # that the limit cannot be set for is a failure of the row, not of a rank in the middle of a distributed load.
        self.limit = None
        self.limit_error = None
        try:
            self.limit = memory_limit(self.manifest, bool(row.dist), row.limit)
        except ValueError as error:
            self.limit_error = error

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
        limit = self.limit
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
    mode = os.environ.get("GPU_MATRIX", "core")
    return filter_rows(rows_for(mode, gpu_world()), os.environ.get("GPU_ROWS"))


def skipped_selected_rows():
    """The rows that the filter selects and that a run of this many ranks leaves out (the distributed ones, on one GPU)."""
    return filter_rows(skipped_rows(os.environ.get("GPU_MATRIX", "core"), gpu_world()), os.environ.get("GPU_ROWS"))


class Harness(Session):
    """The session of the ranks (gpu_harness), and what the rows of this suite add to it: the fixtures, and the load of
    a row on every rank, checked, with the ranks agreeing on its outcome."""

    def __init__(self):
        before, now = raise_open_files_limit()
        stamp("open files limit: %s -> %s (hard %s)" % (before, now, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        self.strategy = os.environ.get("RUNAI_STREAMER_FS_STRATEGY", "")
        # The ranks agree over a CPU-only group that waits longer than the streamer's own distributed timeout, so that a
        # rank that is only slow is never mistaken for a failed one.
        streamer_timeout = int(os.environ.setdefault("RUNAI_STREAMER_DIST_TIMEOUT", str(DIST_TIMEOUT_SECONDS)))
        super().__init__("streaming", run=self.strategy, nccl_timeout=DIST_TIMEOUT_SECONDS,
                         control_timeout=streamer_timeout + CONTROL_MARGIN_SECONDS)
        self.fixtures = None

    def load_fixtures(self, models):
        with self.timer.phase("init.manifests"):
            self.fixtures = Fixtures(os.environ.get("GPU_FIXTURES", ".gpu-fixtures"), models)

    def row_run(self, row, fault=None):
        """Load one row on both ranks, check it, and agree on the outcome. Returns (run, any_raised): run.errors are
        the errors of this rank, and any_raised is True when a rank raised, which may leave the collectives out of step."""
        self.barrier()
        run = RowRun(row, self.fixtures, self.device, self.timer, fault)
        if self.rank == 0:
            stamp("ROW %s  model %s  %s  owned=%s  device=%s  limit=%s  strategy=%s" % (
                row.id, row.model, ("distributed " + row.policy) if row.dist else "not distributed",
                row.owned, row.device, row.limit, self.strategy))
        outcome = self.begin(row.id, strategy=self.strategy)
        started = time.perf_counter()
        raised = False
        try:
            if run.limit_error is not None:
                run.error("the memory limit of the row cannot be set: %s" % run.limit_error)
            else:
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
        ok, any_raised = self.agree(bool(run.errors), raised)
        # The results are written after every row (end), not only at the end: a hang that the NCCL watchdog turns into a
        # killed process would otherwise take the results of the rows that already ran with it.
        self.end(outcome, ok, seconds=elapsed, fault=run.fault, gpu_allocated_bytes=gpu_allocated,
                 errors=run.errors_with_total())
        print("[rank %d] %s %-26s %-12s %s" % (self.rank, "PASS" if ok else "FAIL", row.id, self.strategy,
                                              "%.1fs" % elapsed), flush=True)
        for message in run.errors_with_total():
            print("[rank %d]   %s" % (self.rank, message), flush=True)
        return run, any_raised


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
        HARNESS.expected_cases = len(rows)
        skipped = len(skipped_selected_rows())
        if skipped:
            note = "ONE GPU: %d distributed rows were not run (they need two GPUs)" % skipped
            HARNESS.skip(skipped, note)
            if HARNESS.rank == 0:
                stamp(note)
        # A leg that has just some of the rows (GPU_ALLOW_EMPTY_ROWS) is given prefixes that it does not all have.
        unmatched = unmatched_prefixes(rows_for(os.environ.get("GPU_MATRIX", "core")), os.environ.get("GPU_ROWS"))
        if unmatched and os.environ.get("GPU_ALLOW_EMPTY_ROWS") != "true":
            self.fail("no row of the %s matrix starts with %s: a typo in the filter?" % (
                os.environ.get("GPU_MATRIX", "core"), unmatched))
        if not rows and skipped:
            self.fail("every row selected is a distributed row, which needs two GPUs (%d rows): nothing to run on one" %
                      skipped)
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
        HARNESS.expected_cases = 2
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
