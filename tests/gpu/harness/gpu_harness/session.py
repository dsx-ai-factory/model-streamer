# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ranks of a test run launched by torchrun, one GPU each, and what they write.

A suite whose tests need several ranks (a distributed load, a collective) builds a Session in setUpModule and uses it in
its tests:

    class MyTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            SESSION.expected_cases = 3
        def test_something(self):
            outcome = SESSION.begin("something")            # written as started: a kill in the middle names the case
            errors, raised = run_the_case()                  # on every rank
            ok, any_raised = SESSION.agree(bool(errors), raised)   # every rank gets the same answer
            SESSION.end(outcome, ok, errors=errors)

What the session does for a suite: checks that the run has the ranks and the GPUs that it was asked for (no skipping: a
GPU test that cannot run is a failure), starts the NCCL group of the ranks and a CPU-only (gloo) group that the ranks
agree on the outcome of a case over, so that the agreement adds no NCCL traffic and works after a collective went wrong,
times the phases, and writes the results files (results.py) after every case, so that a killed run still says where it
was. Needs torch and CUDA: a test of one process, with no rank to agree with, does not need a Session (see run_unittest).
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from .phase_timer import PhaseTimer, dump_jsonl, report
from .results import timing_filename, write_results
from .world import gpu_world

NCCL_TIMEOUT_SECONDS = 300
# The gloo group that the ranks agree over waits longer than the NCCL group, so that a rank that is only slow is never
# taken for one that failed.
CONTROL_MARGIN_SECONDS = 120


class Session:
    def __init__(self, suite, run="default", world=None, results_dir=None, nccl_timeout=NCCL_TIMEOUT_SECONDS,
                 control_timeout=None):
        """suite: the name of the suite (the results say it). run: the name of this run of it (the results file). world:
        the ranks, GPU_WORLD by default. control_timeout: how long the ranks wait for each other when they agree."""
        self.suite = suite
        self.run = run
        self.world = gpu_world() if world is None else world
        if int(os.environ.get("WORLD_SIZE", "0")) != self.world:
            raise RuntimeError("Launch with torchrun --nproc_per_node=%d" % self.world)
        if not torch.cuda.is_available() or torch.cuda.device_count() < self.world:
            raise RuntimeError("%d CUDA GPUs are required; GPU tests must not skip" % self.world)
        if not dist.is_nccl_available():
            raise RuntimeError("NCCL is required")
        self.rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        self.results_dir = results_dir or os.environ.get("GPU_RESULTS", "test-results/gpu")
        os.makedirs(self.results_dir, exist_ok=True)
        self.timer = PhaseTimer(announce=(self.rank == 0))
        with self.timer.phase("init.nccl"):
            dist.init_process_group("nccl", timeout=timedelta(seconds=nccl_timeout))
        wait = control_timeout if control_timeout is not None else nccl_timeout + CONTROL_MARGIN_SECONDS
        self.control = dist.new_group(backend="gloo", timeout=timedelta(seconds=wait))
        self.gpu_baseline = torch.cuda.memory_allocated(self.device)  # what the GPU holds before any case
        self.cases = []
        self.expected_cases = 0
        self.skipped_cases = 0
        self.skipped_note = ""

    def skip(self, count, note):
        """Cases that this run leaves out on purpose (count of them), and the line that says why: the summary prints it,
        so a run that left cases out is not taken for one that ran them all."""
        self.skipped_cases = count
        self.skipped_note = note

    def barrier(self):
        dist.barrier(group=self.control)

    def begin(self, case, **fields):
        """Start a case: it is in the results, with "ok": null, before anything runs."""
        outcome = {"case": case, "ok": None}
        outcome.update(fields)
        self.cases.append(outcome)
        self.write()
        return outcome

    def agree(self, failed, raised=False):
        """The ranks agree on the outcome of a case, before any goes on to the next, or one rank enters a collective alone
        and hangs. failed: this rank found an error. raised: this rank raised an exception, after which its collectives
        may be out of step. Returns (ok, any_raised) of all the ranks."""
        flag = torch.tensor([1 if failed else 0, 1 if raised else 0])
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.control)
        return flag[0].item() == 0, flag[1].item() == 1

    def end(self, outcome, ok, **fields):
        """Finish a case. After every case, not only at the end: a hang that the NCCL watchdog turns into a killed
        process would otherwise take the results of the cases that already ran with it."""
        outcome["ok"] = ok
        outcome.update(fields)
        self.write()

    def write(self):
        dump_jsonl(self.timer.records, os.path.join(self.results_dir, timing_filename(self.run, self.rank)))
        write_results(self.results_dir, self.suite, self.run, self.rank, self.cases, self.expected_cases, self.world,
                      self.skipped_cases, self.skipped_note)

    def finish(self):
        self.write()
        if self.rank == 0:
            print(report(self.timer.records, top=15), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
