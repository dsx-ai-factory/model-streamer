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
MACHINE = {"kernel": "6.8", "python": "3.12", "cpus": 32, "mem_total_gib": 64.0, "thp": "always",
           "storage": {"mount": "/mnt", "type": "ext4"}}
PROFILE = {"schema": 1, "created": "2026-11-18T00:00:00+00:00", "raw_gibps": {"h2d_pinned": [6.0]},
           "notes": ["a note"], "params": {"repeats": 3, "files": 8, "file_gib": 2.0, "mem_gib": 4, "threads": 32},
           "host": {"hostname": "h", "storage": {"mount": "/mnt", "type": "ext4"}},
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
        one = {"startup_s": took + 20, "gib": 40.0, "tensors": 7, "errors": [], "gpus": ["H200"],
               "versions": {"vllm": "0.33.0", "runai-model-streamer": "0.17.1"},
               "pkg": PKG if test == "vllm_streamer" else None}
        log = f"INFO Model loading took 40.2 GiB memory and {took:.6f} seconds"
        if "_dist_" in test:  # one line per rank
            took = 9.0 if "streamer" in test else 30.0
            log = "\n".join(f"(rank {i}) Model loading took 20.1 GiB memory and {took + i:.6f} seconds"
                            for i in range(2))
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

    def run_bench(self, launch=child_output, prof=PROFILE, no_vllm=None, gpus=0, **over):
        a = argparse.Namespace(model_dir=self.dir.name, ref_dir=None, nproc=None, tests="", env=[], warm=False,
                               vllm_args={},
                               profile=None, no_calibrate=True, repeats=2, out=self.out, log_dir=None)
        vars(a).update(over)
        self.calls = []

        def fake(cmd, env, timeout):
            self.calls.append((cmd, env))
            return launch(cmd)

        code = 0
        with mock.patch.object(runner, "launch", fake), \
                mock.patch.object(runner, "vllm_unavailable", return_value=no_vllm), \
                mock.patch.object(runner, "gpu_count", return_value=gpus), \
                mock.patch.object(runner, "describe_machine", return_value=dict(MACHINE)), \
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
        code, out = self.run_bench(tests="streamer_cpu,streamer_cuda,safeopen_to_cuda")
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
        self.assertIn("80% of ceiling", page)
        self.assertIn("delivers the tensors on one GPU", page)  # the test's setup, shown on its name
        # interleaved: every test once, then every test again
        order = [c[c.index("--child") + 1] for c, _ in self.calls]
        self.assertEqual(order, ["streamer_cpu", "streamer_cuda", "safeopen_to_cuda"] * 2)

    def test_no_profile_has_no_ceiling(self):
        code, out = self.run_bench(prof=None, tests="streamer_cpu")
        self.assertEqual(code, 0)
        self.assertNotIn("ceiling", out["results"][0])
        self.assertIsNone(out["calibration"])
        with open(os.path.splitext(self.out)[0] + ".html") as f:
            self.assertIn("No machine profile was used", f.read())

    def test_distributed_runs_under_torchrun(self):
        code, out = self.run_bench(gpus=2, nproc=2, env=["X=1"], tests=DIST)
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
        code, out = self.run_bench(tests="streamer_cuda,vllm_default,vllm_streamer", repeats=1)
        self.assertEqual(code, 0)
        _, default, streamer = out["results"]
        self.assertEqual((default["load_s"], default["gibps"], default["startup_s"]), (40.0, 1.0, 60.0))
        self.assertEqual(streamer["runs"], [{"load_s": 8.0, "gibps": 5.0, "startup_s": 28.0, "gpu_gib": 40.2}])
        self.assertNotIn("cpu_s_per_gib", streamer)
        self.assertEqual(streamer["ceiling"]["name"], "h2d_pinned")
        self.assertEqual(out["vllm_speedup"], 5.0)
        self.assertEqual(out["versions"], {"vllm": "0.33.0", "runai-model-streamer": "0.17.1"})
        self.assertEqual(out["environment"]["gpus"], ["H200"])
        with open(os.path.splitext(self.out)[0] + ".html") as f:
            page = f.read()
        for text in ("5.0x as fast", "vLLM 0.33.0", "0.17.1", "H200", "ext4 on /mnt"):
            self.assertIn(text, page)

    def test_calibration_is_kept_with_the_results(self):
        code, out = self.run_bench(tests="streamer_cpu", repeats=1)
        cal = out["calibration"]
        self.assertEqual((cal["path"], cal["created"], cal["notes"]), ("p.json", PROFILE["created"], ["a note"]))
        self.assertEqual(cal["ceilings_gibps"]["h2d_pinned"], 6.0)
        self.assertNotIn("raw_gibps", cal)
        with open(os.path.splitext(self.out)[0] + ".html") as f:
            page = f.read()
        for text in ("Calibration", "disk_direct_read_parallel", "scored against", "a note", "8 files of 2.0 GiB"):
            self.assertIn(text, page)

    def test_vllm_weights_missing_from_the_gpu_is_a_warning(self):
        log = "Model loading took 3.0 GiB memory and 8.000000 seconds"
        result = 'RESULT {"startup_s": 9, "gib": 40.0, "tensors": 7, "pkg": null, "versions": {}, "errors": []}'
        _, out = self.run_bench(launch=lambda cmd: (result, log, None), tests="vllm_default", repeats=1)
        self.assertEqual(out["results"][0]["runs"][0]["warnings"],
                         ["vLLM reports 3.0 GiB on the GPU for 40.0 GiB of weights"])

    def test_vllm_args_reach_only_the_vllm_tests(self):
        _, out = self.run_bench(tests="streamer_cuda,vllm_default", repeats=1, vllm_args={"trust_remote_code": True})
        by = {c[c.index("--child") + 1]: c for c, _ in self.calls}
        self.assertNotIn("--vllm-args", by["streamer_cuda"])
        self.assertEqual(json.loads(by["vllm_default"][by["vllm_default"].index("--vllm-args") + 1]),
                         {"trust_remote_code": True})
        self.assertEqual(out["settings"]["vllm_args"], {"trust_remote_code": True})

    def test_vllm_without_its_log_line_fails(self):
        code, out = self.run_bench(launch=lambda cmd: ('RESULT {"startup_s": 1, "gib": 1, "tensors": 1, '
                                                       '"pkg": null, "vllm": "x", "errors": []}', "", None),
                                   tests="vllm_default", repeats=1)
        self.assertEqual(code, 1)
        self.assertIn("Model loading took", out["results"][0]["runs"][0]["failed"])

    def ran(self):
        return [c[c.index("--child") + 1] for c, _ in self.calls]

    def test_vllm_runs_by_default_on_a_gpu_machine(self):
        self.run_bench(gpus=1, repeats=1)
        self.assertEqual(self.ran(), ["streamer_cpu", "streamer_cuda", "safeopen_to_cuda", "vllm_default", "vllm_streamer",
                                      "vllm_streamer_gpu", "vllm_instanttensor"])
        self.assertFalse(any("torch.distributed.run" in c for c, _ in self.calls))

    def test_one_vllm_loader_missing_leaves_out_only_that_test(self):
        with mock.patch.object(runner.importlib.util, "find_spec",
                               lambda name: None if name == "instanttensor" else object()):
            open(os.path.join(self.dir.name, "config.json"), "w").close()
            a = argparse.Namespace(model_dir=self.dir.name)
            self.assertIsNone(runner.vllm_unavailable(a, "vllm_streamer"))
            self.assertIn("instanttensor is not installed", runner.vllm_unavailable(a, "vllm_instanttensor"))
            os.remove(os.path.join(self.dir.name, "config.json"))
            self.assertIn("config.json", runner.vllm_unavailable(a, "vllm_default"))

    def test_vllm_is_left_out_when_it_cannot_run(self):
        code, _ = self.run_bench(gpus=1, repeats=1, no_vllm="vllm is not installed")
        self.assertEqual(code, 0)
        self.assertFalse([t for t in self.ran() if t.startswith("vllm")])
        self.assertFalse([t for t in self.run_bench(gpus=0, repeats=1) and self.ran() if t.startswith("vllm")])
        # asked for by name, it is an error instead
        self.assertIn("cannot run", self.run_bench(tests="vllm_default", no_vllm="vllm is not installed")[0])

    def test_more_ranks_than_gpus_is_refused(self):
        self.assertIn("needs 8 GPUs, one per rank; this machine has 1", self.run_bench(gpus=1, nproc=8)[0])

    def test_nproc_alone_runs_only_the_distributed_test(self):
        self.run_bench(gpus=8, nproc=2, repeats=1, tests=DIST)
        self.assertEqual(self.ran(), [DIST])

    def test_nproc_runs_the_distributed_tests(self):
        code, out = self.run_bench(gpus=8, nproc=2, repeats=1)
        self.assertEqual(code, 0)
        self.assertEqual(self.ran(), [DIST, "vllm_dist_default", "vllm_dist_streamer", "vllm_dist_instanttensor"])
        self.assertTrue(all("torch.distributed.run" in c for c, _ in self.calls))
        streamer = out["results"][2]
        # the slowest rank's time, and the GPU memory of all ranks
        self.assertEqual(streamer["runs"], [{"load_s": 10.0, "gibps": 4.0, "startup_s": 60.0, "gpu_gib": 40.2}])
        self.assertEqual(streamer["ceiling"]["name"], "disk_direct_read_parallel")
        self.assertEqual(out["vllm_dist_speedup"], 3.1)
        with open(os.path.splitext(self.out)[0] + ".html") as f:
            self.assertIn("vLLM with tensor parallelism (2 ranks)", f.read())

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
        self.assertIn("--nproc is for", self.run_bench(gpus=2, nproc=2, tests="streamer_cuda")[0])
        self.assertIn("unknown test", self.run_bench(tests="nope")[0])

    def test_warm_is_scored_against_memcpy(self):
        self.assertEqual(profile.ceilings_for("streamer_cpu", PROFILE, warm=True)[0], "host_memcpy_nt")
        self.assertEqual(profile.ceilings_for("streamer_cpu", PROFILE, remote=True)[0], None)  # no S3 ceiling


if __name__ == "__main__":
    unittest.main()
