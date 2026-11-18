# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runner and the report with the measured processes stubbed out: no torch, GPU or Linux needed.

    python3 -m unittest discover -s benchmarks/tests
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamer_bench import DIST, profile, runner  # noqa: E402

PKG = "/site/runai_model_streamer"
PROFILE = {"host": {"hostname": "h"},
           "ceilings_gibps": {"disk_direct_read_parallel": 10.0, "h2d_pinned": 6.0, "host_memcpy_nt": 30.0}}


def child_output(cmd, load_s=5.0, errors=()):
    """What a measured process prints for the test named in cmd."""
    test = cmd[cmd.index("--child") + 1]
    one = {"load_s": load_s, "gib": 40.0, "tensors": 7, "cpu_s": 8.0, "peak_rss_gib": 4.0,
           "pkg": PKG if test.startswith("streamer") else None, "errors": list(errors)}
    if test == DIST:
        one = [dict(one, rank=i, rank_s=load_s - 1, peak_gpu_gib=6.0) for i in range(2)]
    log = "Filesystem strategy: io_uring\n[RunAI Streamer][Distributed] x"
    if test.startswith("vllm_"):  # vLLM reports its own load time, in its log
        took = 8.0 if test == "vllm_streamer" else 40.0
        one = {"startup_s": took + 20, "gib": 40.0, "tensors": 7, "vllm": "0.33.0", "errors": [],
               "pkg": PKG if test == "vllm_streamer" else None}
        log = f"INFO Model loading took 15.1 GiB memory and {took:.6f} seconds"
    return "RESULT " + json.dumps(one), log, None


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        open(os.path.join(self.dir.name, "m.safetensors"), "w").close()
        self.out = os.path.join(self.dir.name, "results.json")
        for name in ("evict", "prime"):  # the page cache calls are Linux only
            p = mock.patch.object(runner, name, return_value=0)
            p.start()
            self.addCleanup(p.stop)

    def run_bench(self, launch=child_output, prof=PROFILE, **over):
        a = argparse.Namespace(model_dir=self.dir.name, ref_dir=None, nproc=None, tests="", vllm=False, env=[],
                               warm=False,
                               profile=None, no_calibrate=True, repeats=2, out=self.out, log_dir=None)
        vars(a).update(over)
        self.calls = []

        def fake(cmd, env, timeout):
            self.calls.append((cmd, env))
            return launch(cmd)

        code = 0
        with mock.patch.object(runner, "launch", fake), mock.patch.object(runner, "check_vllm"), \
                mock.patch.object(runner, "get_profile", return_value=(prof, "p.json" if prof else None)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                runner.main(a)
            except SystemExit as e:
                code = e.code
        if not os.path.exists(self.out):
            return code, None
        with open(self.out) as f:
            return code, json.load(f)

    def test_results_layout_and_ceilings(self):
        code, out = self.run_bench(tests="streamer_cpu,streamer_cuda,safeopen_page_in")
        self.assertEqual(code, 0)
        self.assertEqual(out["streamer"], [PKG])
        self.assertEqual(out["model"]["gib"], 40.0)
        cpu, cuda, base = out["results"]
        self.assertEqual((cpu["gibps"], cpu["load_s"], cpu["cpu_s_per_gib"]), (8.0, 5.0, 0.2))
        self.assertEqual(cpu["ceiling"], {"name": "disk_direct_read_parallel", "gibps": 10.0, "pct": 80.0,
                                          "nearest": "disk_direct_read_parallel"})
        self.assertEqual(cuda["ceiling"]["name"], "h2d_pinned")  # the lower of storage and host-to-GPU
        self.assertEqual(cpu["runs"][0], {"load_s": 5.0, "gibps": 8.0, "cpu_s": 8.0, "peak_rss_gib": 4.0})
        self.assertEqual(base["fs_strategy"], ["io_uring"])
        with open(os.path.splitext(self.out)[0] + ".html") as f:
            page = f.read()
        self.assertIn("streamer_cuda", page)
        self.assertIn("80% of disk_direct_read_parallel", page)
        # interleaved: every test once, then every test again
        order = [c[c.index("--child") + 1] for c, _ in self.calls]
        self.assertEqual(order, ["streamer_cpu", "streamer_cuda", "safeopen_page_in"] * 2)

    def test_no_profile_has_no_ceiling(self):
        code, out = self.run_bench(prof=None, tests="streamer_cpu")
        self.assertEqual(code, 0)
        self.assertNotIn("ceiling", out["results"][0])
        self.assertIsNone(out["profile"])

    def test_distributed_runs_under_torchrun(self):
        code, out = self.run_bench(nproc=2, env=["X=1"])
        self.assertEqual(code, 0)
        cmd, env = self.calls[0]
        self.assertIn("torch.distributed.run", cmd)
        self.assertIn("--nproc_per_node=2", cmd)
        self.assertEqual(env["X"], "1")
        (r,) = out["results"]
        self.assertEqual(r["test"], DIST)
        self.assertEqual(r["runs"][0]["rank_s"], [4.0, 4.0])
        self.assertEqual(r["runs"][0]["cpu_s"], 16.0)  # summed over the ranks
        self.assertEqual(r["ceiling"]["name"], "disk_direct_read_parallel")
        self.assertEqual(out["settings"], {"repeats": 2, "warm": False, "nproc": 2, "env": {"X": "1"}})

    def test_vllm_load_time_comes_from_its_log(self):
        code, out = self.run_bench(tests="streamer_cuda", vllm=True, repeats=1)
        self.assertEqual(code, 0)
        _, default, streamer = out["results"]
        self.assertEqual((default["load_s"], default["gibps"], default["startup_s"]), (40.0, 1.0, 60.0))
        self.assertEqual(streamer["runs"], [{"load_s": 8.0, "gibps": 5.0, "startup_s": 28.0}])
        self.assertNotIn("cpu_s_per_gib", streamer)
        self.assertEqual(streamer["ceiling"]["name"], "h2d_pinned")
        self.assertEqual((out["vllm_speedup"], out["vllm"]), (5.0, ["0.33.0"]))
        with open(os.path.splitext(self.out)[0] + ".html") as f:
            self.assertIn("5.0x as fast", f.read())

    def test_vllm_without_its_log_line_fails(self):
        code, out = self.run_bench(launch=lambda cmd: ('RESULT {"startup_s": 1, "gib": 1, "tensors": 1, '
                                                       '"pkg": null, "vllm": "x", "errors": []}', "", None),
                                   tests="vllm_default", repeats=1)
        self.assertEqual(code, 1)
        self.assertIn("Model loading took", out["results"][0]["runs"][0]["failed"])

    def test_vllm_is_opt_in(self):
        self.run_bench()
        self.assertFalse(any("vllm" in " ".join(c) for c, _ in self.calls))

    def test_wrong_data_fails_the_test(self):
        code, out = self.run_bench(launch=lambda cmd: child_output(cmd, errors=["w: data differs from safetensors"]),
                                   tests="streamer_cpu")
        self.assertEqual(code, 1)
        self.assertFalse(out["results"][0]["ok"])
        self.assertEqual(out["results"][0]["runs"][0]["errors"], ["w: data differs from safetensors"])

    def test_a_run_without_a_result_fails_the_test(self):
        code, out = self.run_bench(launch=lambda cmd: ("", "boom", "timed out after 3600 s"), tests="streamer_cpu")
        self.assertEqual(code, 1)
        self.assertEqual(out["results"][0], {"test": "streamer_cpu", "ok": False,
                                             "runs": [{"failed": "timed out after 3600 s"}] * 2})
        with open(os.path.splitext(self.out)[0] + ".html") as f:
            self.assertIn("failed", f.read())

    def test_cache_left_before_a_cold_run_is_a_warning(self):
        with mock.patch.object(runner, "evict", return_value=2 << 30):
            code, out = self.run_bench(tests="streamer_cpu", repeats=1)
        self.assertEqual(code, 0)
        self.assertEqual(out["results"][0]["runs"][0]["warnings"], ["2.0 GiB still in the page cache before the run"])

    def test_test_selection_errors(self):
        self.assertIn("needs --nproc", self.run_bench(tests=DIST)[0])
        self.assertIn("--nproc is for", self.run_bench(nproc=2, tests="streamer_cuda")[0])
        self.assertIn("unknown test", self.run_bench(tests="nope")[0])

    def test_warm_is_scored_against_memcpy(self):
        self.assertEqual(profile.ceilings_for("streamer_cpu", PROFILE, warm=True)[0], "host_memcpy_nt")
        self.assertEqual(profile.ceilings_for("streamer_cpu", PROFILE, remote=True)[0], None)  # no S3 ceiling


if __name__ == "__main__":
    unittest.main()
