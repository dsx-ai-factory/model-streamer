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
from streamer_bench import markdown, profile, runner  # noqa: E402

DIST = "streamer_dist"

PKG = "/site/runai_model_streamer"
MACHINE = {"kernel": "6.8", "python": "3.12", "cpus": 32, "mem_total_gib": 64.0, "thp": "always",
           "storage": {"mount": "/mnt", "type": "ext4"}}
PROFILE = {"schema": 1, "created": "2026-11-18T00:00:00+00:00", "raw_gibps": {"h2d_pinned": [6.0]},
           "notes": ["a note"], "params": {"repeats": 3, "files": 8, "file_gib": 2.0, "mem_gib": 4, "threads": 32},
           "host": {"hostname": "h", "storage": {"mount": "/mnt", "type": "ext4"}},
           "ceilings_gibps": {"disk_direct_read_parallel": 10.0, "h2d_pinned": 6.0, "host_memcpy_nt": 30.0}}


def child_output(cmd, load_s=None, errors=()):
    """What a measured process prints for the test named in cmd: safetensors takes 20 s, the others 5 s."""
    test = cmd[cmd.index("--child") + 1]
    if load_s is None:
        load_s = 20.0 if test.startswith("safetensors") else 5.0
    one = {"load_s": load_s, "gib": 40.0, "tensors": 7, "cpu_s": 8.0, "peak_rss_gib": 4.0, "gpus": ["H200"],
           "versions": {"runai-model-streamer": "0.17.1", "instanttensor": "0.2.3"},
           "pkg": PKG if test.startswith("streamer") else None, "errors": list(errors)}
    if test.endswith("_dist"):
        one = [dict(one, rank=i, rank_s=load_s - 1, peak_gpu_gib=6.0) for i in range(2)]
    return "RESULT " + json.dumps(one), "Filesystem strategy: io_uring\n[RunAI Streamer][Distributed] x", None


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

    def run_bench(self, launch=child_output, prof=PROFILE, missing=None, gpus=0, **over):
        a = argparse.Namespace(model_dir=self.dir.name, ref_dir=None, nproc=None, device=None, tests="", env=[],
                               warm=False, scratch_dir=None,
                               profile=None, no_calibrate=True, repeats=2, out=self.out, log_dir=None)
        vars(a).update(over)
        self.calls = []

        def fake(cmd, env, timeout):
            self.calls.append((cmd, env))
            return launch(cmd)

        code = 0
        with mock.patch.object(runner, "launch", fake), \
                mock.patch.object(runner, "unavailable",
                                  lambda a, test, remote=False: missing if "instanttensor" in test else None), \
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
        code, out = self.run_bench(tests="streamer_cpu,streamer_cuda,safetensors_cuda")
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
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        self.assertIn("`streamer_cuda`: The streamer reads every file and delivers the tensors on one GPU", page)
        # interleaved: every test once, then every test again
        order = [c[c.index("--child") + 1] for c, _ in self.calls]
        self.assertEqual(order, ["streamer_cpu", "streamer_cuda", "safetensors_cuda"] * 2)

    def test_no_profile_has_no_ceiling(self):
        code, out = self.run_bench(prof=None, tests="streamer_cpu")
        self.assertEqual(code, 0)
        self.assertNotIn("ceiling", out["results"][0])
        self.assertIsNone(out["calibration"])
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            self.assertNotIn("parallel direct read", f.read())

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

    def test_calibration_is_kept_with_the_results(self):
        code, out = self.run_bench(tests="streamer_cpu", repeats=1)
        cal = out["calibration"]
        self.assertEqual((cal["path"], cal["created"], cal["notes"]), ("p.json", PROFILE["created"], ["a note"]))
        self.assertEqual(cal["ceilings_gibps"]["h2d_pinned"], 6.0)
        self.assertNotIn("raw_gibps", cal)
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        self.assertIn("- Storage: ext4 on /mnt, 10 GiB/s parallel direct read", page)

    def ran(self):
        return [c[c.index("--child") + 1] for c, _ in self.calls]

    def test_every_loader_runs_to_each_destination_by_default(self):
        code, out = self.run_bench(gpus=1, repeats=1)
        self.assertEqual(code, 0)
        self.assertEqual(self.ran(), ["safetensors_cpu", "streamer_cpu", "safetensors_cuda", "instanttensor_cuda",
                                      "streamer_cuda"])
        self.assertFalse(any("torch.distributed.run" in c for c, _ in self.calls))
        self.run_bench(gpus=0, repeats=1)  # without a GPU, to host memory only
        self.assertEqual(self.ran(), ["safetensors_cpu", "streamer_cpu"])
        self.run_bench(gpus=1, repeats=1, device="cuda")
        self.assertEqual(self.ran(), ["safetensors_cuda", "instanttensor_cuda", "streamer_cuda"])
        self.assertIn("no test applies", self.run_bench(gpus=0, device="cuda")[0])

    def test_speedup_is_against_safetensors_to_the_same_destination(self):
        code, out = self.run_bench(gpus=1, repeats=1, device="cuda")
        by = {s["test"]: s for s in out["results"]}
        self.assertEqual([by[t]["speedup"] for t in ("safetensors_cuda", "instanttensor_cuda", "streamer_cuda")],
                         [1.0, 4.0, 4.0])
        self.assertEqual((by["streamer_cuda"]["loader"], by["streamer_cuda"]["group"]), ("streamer", "cuda"))
        self.assertEqual(out["versions"], {"runai-model-streamer": "0.17.1", "instanttensor": "0.2.3"})
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        for text in ("| Qwen | 1 × H200 | Safetensors | 20.00 | 2.0 | 1.00x |",
                     "| | | InstantTensor | 5.00 | 8.0 | **4.00x** |",
                     "| | | Model Streamer | 5.00 | 8.0 | **4.00x** |", "- instanttensor: 0.2.3",
                     f"- runai_model_streamer: 0.17.1 (imported from {PKG})", "- GPUs: 1 × H200",
                     "- Runs: median of 1 per loader, cold page cache"):
            self.assertIn(text.replace("Qwen", os.path.basename(self.dir.name)), page)
        # without the safetensors test there is nothing to compare with
        _, out = self.run_bench(tests="streamer_cuda", repeats=1)
        self.assertNotIn("speedup", out["results"][0])

    def test_a_missing_loader_leaves_out_only_its_tests(self):
        code, _ = self.run_bench(gpus=1, repeats=1, device="cuda", missing="instanttensor is not installed")
        self.assertEqual(code, 0)
        self.assertEqual(self.ran(), ["safetensors_cuda", "streamer_cuda"])
        # asked for by name, it is an error instead
        self.assertIn("cannot run",
                      self.run_bench(tests="instanttensor_cuda", missing="instanttensor is not installed")[0])

    def test_an_s3_model_is_downloaded_for_the_loaders_that_read_local_files(self):
        scratch = os.path.join(self.dir.name, "scratch")
        with mock.patch.object(runner, "download", return_value=(15.0, None)) as download, \
                mock.patch.object(runner, "clear") as clear:
            code, out = self.run_bench(gpus=1, repeats=1, device="cuda", model_dir="s3://b/Qwen",
                                       ref_dir=self.dir.name, scratch_dir=scratch)
        self.assertEqual(code, 0)
        self.assertEqual(download.call_count, 2)  # not for the streamer
        self.assertEqual(clear.call_count, 2)
        by = {c[c.index("--child") + 1]: c for c, _ in self.calls}
        self.assertEqual(by["safetensors_cuda"][2], scratch)  # loads what was downloaded
        self.assertEqual(by["streamer_cuda"][2], "s3://b/Qwen")
        self.assertIn("--ref-dir", by["streamer_cuda"])
        self.assertNotIn("--ref-dir", by["safetensors_cuda"])
        st, it, streamer = out["results"]
        self.assertEqual((st["load_s"], st["download_s"], st["runs"][0]["download_s"]), (35.0, 15.0, 15.0))
        self.assertEqual((it["load_s"], streamer["load_s"]), (20.0, 5.0))
        self.assertEqual((it["speedup"], streamer["speedup"]), (1.75, 7.0))
        self.assertNotIn("download_s", streamer)
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        self.assertIn("| Qwen on S3 | 1 × H200 | download + Safetensors | 35.00 (15.00 download + 20.00 load) |", page)
        self.assertIn("| | | Model Streamer | 5.00 | 8.0 | **7.00x** |", page)

    def test_without_a_scratch_dir_only_the_streamer_reads_s3(self):
        a = argparse.Namespace(scratch_dir=None)
        self.assertIn("--scratch-dir", runner.unavailable(a, "safetensors_cuda", remote=True))
        self.assertIsNone(runner.unavailable(a, "streamer_cuda", remote=True))
        self.assertIsNone(runner.unavailable(a, "safetensors_cuda"))

    def test_nproc_runs_the_distributed_tests(self):
        code, out = self.run_bench(gpus=8, nproc=2, repeats=1)
        self.assertEqual(code, 0)
        self.assertEqual(self.ran(), ["safetensors_dist", "instanttensor_dist", DIST])
        self.assertTrue(all("torch.distributed.run" in c for c, _ in self.calls))
        self.assertEqual([s["group"] for s in out["results"]], ["dist"] * 3)
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            self.assertIn("| 2 × H200 | Safetensors |", f.read())

    def test_several_results_make_one_page_and_one_table(self):
        self.run_bench(gpus=1, repeats=1, device="cuda")
        first = os.path.join(self.dir.name, "first.json")
        os.rename(self.out, first)
        self.run_bench(gpus=0, repeats=1)
        page = os.path.join(self.dir.name, "report.md")
        table = markdown.combine([first, self.out], page)
        name = os.path.basename(self.dir.name)
        self.assertIn(f"| {name} | 1 × H200 | Safetensors | 20.00 | 2.0 | 1.00x |", table)
        self.assertIn("| | | Model Streamer | 5.00 | 8.0 | **4.00x** |", table)
        self.assertIn(f"| {name} | host memory | Safetensors | 20.00 | 2.0 | 1.00x |", table)
        with open(page) as f:
            self.assertIn(table, f.read())

    def test_more_ranks_than_gpus_is_refused(self):
        self.assertIn("needs 8 GPUs, one per rank; this machine has 1", self.run_bench(gpus=1, nproc=8)[0])

    def test_nproc_alone_runs_only_the_distributed_test(self):
        self.run_bench(gpus=8, nproc=2, repeats=1, tests=DIST)
        self.assertEqual(self.ran(), [DIST])

    def test_wrong_data_fails_the_test(self):
        code, out = self.run_bench(launch=lambda cmd: child_output(cmd, errors=["w: data differs from safetensors"]),
                                   tests="streamer_cpu")
        self.assertEqual(code, 1)
        self.assertFalse(out["results"][0]["ok"])
        self.assertEqual(out["results"][0]["runs"][0]["errors"], ["w: data differs from safetensors"])

    def test_a_run_without_a_result_fails_the_test(self):
        code, out = self.run_bench(launch=lambda cmd: ("", "boom", "timed out after 3600 s"), tests="streamer_cpu")
        self.assertEqual(code, 1)
        self.assertEqual(out["results"][0], {"test": "streamer_cpu", "loader": "streamer", "group": "cpu", "ok": False,
                                             "runs": [{"failed": "timed out after 3600 s"}] * 2})
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            self.assertIn("| Model Streamer | failed | | |", f.read())

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
