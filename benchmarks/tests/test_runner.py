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
from benchmark import LOADERS, markdown, profile, runner  # noqa: E402

DIST = "streamer_dist"

PKG = "/site/runai_model_streamer"
MACHINE = {"kernel": "6.8", "python": "3.12", "cpus": 32, "mem_total_gib": 64.0, "thp": "always",
           "cpu_model": "Xeon 8570", "cpu_sockets": 2, "memory_dimms": {"type": "DDR5", "count": 4, "gib": 16.0},
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
    log = "Filesystem strategy: io_uring\n[RunAI Streamer][Distributed] x"
    if test.startswith("vllm"):  # a vLLM run: each rank prints its weight loading time, vLLM logs its own line
        ranks = 2 if "_dist_" in test else 1
        weights = 8.0 if "streamer" in test else 10.0 if "instanttensor" in test else 40.0
        one = {"startup_s": weights + 20, "gib": 40.0, "tensors": 7, "errors": [], "gpus": ["H200"],
               "versions": {"vllm": "0.34.0", "runai-model-streamer": "0.17.1"},
               "pkg": PKG if "streamer" in test else None}
        out = "\n".join([f"WEIGHTS {weights + i:.3f}" for i in range(ranks)] + ["RESULT " + json.dumps(one)])
        cache = 99.4 if "instanttensor" in test else 100.0  # its communicator stays on the GPU
        return out, "\n".join(f"Model loading took {40.2 / ranks} GiB memory and {weights + 2 + i:.6f} seconds\n"
                              f"Extended KV cache to {700 + i} blocks ({cache + i} GiB)."
                              for i in range(ranks)), None
    if test.endswith("_dist"):
        one = [dict(one, rank=i, rank_s=load_s - 1, peak_gpu_gib=6.0) for i in range(2)]
    return "RESULT " + json.dumps(one), log, None


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        header = json.dumps({"a": {"dtype": "BF16", "shape": [6], "data_offsets": [0, 12]},
                             "b": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]}}).encode()
        with open(os.path.join(self.dir.name, "m.safetensors"), "wb") as f:
            f.write(len(header).to_bytes(8, "little") + header + bytes(16))
        self.out = os.path.join(self.dir.name, "results.json")
        # the page cache call is Linux only, and the host's memory is not the test's business
        for name, value in (("evict", 0), ("host_memory_gib", None)):
            p = mock.patch.object(runner, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def run_bench(self, launch=child_output, prof=PROFILE, missing=None, gpus=0, vllm=False, **over):
        # every loader and the vLLM tests, as --loaders all --vllm: most tests here are about all of them
        a = argparse.Namespace(model_dir=self.dir.name, ref_dir=None, nproc=None, device=None, loaders=list(LOADERS),
                               in_vllm=True, tests="",
                               env=[], scratch_dir=None, vllm_args={},
                               profile=None, no_calibrate=True, repeats=2, out=self.out, log_dir=None)
        vars(a).update(over)
        self.calls = []

        def fake(cmd, env, timeout):
            self.calls.append((cmd, env))
            return launch(cmd)

        code = 0
        with mock.patch.object(runner, "launch", fake), \
                mock.patch.object(runner, "unavailable",
                                  lambda a, test, remote=False: "vllm is not installed"
                                  if test.startswith("vllm") and not vllm
                                  else missing if "instanttensor" in test else None), \
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
        self.assertEqual((out["model"]["files"], out["model"]["tensor_bytes_largest"], out["model"]["dtypes"]),
                         (1, 12, {"BF16": 0.75, "F32": 0.25}))
        cpu, cuda, base = out["results"]
        self.assertEqual((cpu["gibps"], cpu["load_s"], cpu["cpu_s_per_gib"]), (8.0, 5.0, 0.2))
        self.assertEqual(cpu["ceiling"], {"name": "disk_direct_read_parallel", "gibps": 10.0, "pct": 80.0,
                                          "nearest": "disk_direct_read_parallel"})
        self.assertEqual(cuda["ceiling"]["name"], "h2d_pinned")  # the lower of storage and host-to-GPU
        self.assertEqual(cpu["runs"][0], {"load_s": 5.0, "gibps": 8.0, "cpu_s": 8.0, "peak_rss_gib": 4.0})
        self.assertEqual(base["fs_strategy"], ["io_uring"])
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
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
        self.assertEqual(out["settings"], {"repeats": 2, "nproc": 2, "env": {"X": "1"}})

    def test_calibration_is_kept_with_the_results(self):
        code, out = self.run_bench(tests="streamer_cpu", repeats=1)
        cal = out["calibration"]
        self.assertEqual((cal["path"], cal["created"], cal["notes"]), ("p.json", PROFILE["created"], ["a note"]))
        self.assertEqual(cal["ceilings_gibps"]["h2d_pinned"], 6.0)
        self.assertNotIn("raw_gibps", cal)
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        self.assertIn("- Storage: ext4, 10 GiB/s parallel direct read", page)
        self.assertIn("- CPUs: 2 × Xeon 8570, 32 CPUs", page)
        self.assertIn(f"| {os.path.basename(self.dir.name)} | 40.0 GiB | 1 | 2 | 8 B | 12 B | BF16 75%, F32 25% |",
                      page)
        self.assertIn("- Memory: 64.0 GiB DDR5 (4 × 16 GiB)", page)
        self.assertNotIn("/mnt", page)
        self.assertIn("## Test on local disk (10 GiB/s parallel direct read)", page)

    def ran(self):
        return [c[c.index("--child") + 1] for c, _ in self.calls]

    def test_every_loader_runs_to_each_destination_by_default(self):
        code, out = self.run_bench(gpus=1, repeats=1)
        self.assertEqual(code, 0)
        self.assertEqual(self.ran(), ["safetensors_cpu", "streamer_cpu", "safetensors_cuda",
                                      "instanttensor_cuda", "streamer_cuda"])
        self.assertFalse(any("torch.distributed.run" in c for c, _ in self.calls))
        self.run_bench(gpus=0, repeats=1)  # without a GPU, to host memory only
        self.assertEqual(self.ran(), ["safetensors_cpu", "streamer_cpu"])
        self.run_bench(gpus=1, repeats=1, device="cuda")
        self.assertEqual(self.ran(), ["safetensors_cuda", "instanttensor_cuda", "streamer_cuda"])
        self.assertIn("no test applies", self.run_bench(gpus=0, device="cuda")[0])
        self.run_bench(gpus=1, repeats=1, device="cuda", loaders=["safetensors", "streamer"])
        self.assertEqual(self.ran(), ["safetensors_cuda", "streamer_cuda"])
        # the same filter applies to tests that are named
        self.run_bench(gpus=1, repeats=1, loaders=["streamer"], tests="safetensors_cuda,streamer_cuda,streamer_cpu")
        self.assertEqual(self.ran(), ["streamer_cuda", "streamer_cpu"])

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
                     "- runai_model_streamer: 0.17.1", "- GPUs: 1 × H200",
                     "- Runs: median of 1 per loader\n",
                     "- Page cache, for the models on local disk: cold before every run"):
            self.assertIn(text.replace("Qwen", os.path.basename(self.dir.name)), page)
        self.assertTrue(page.rstrip().endswith("|"))  # the report ends with its table
        self.assertNotIn("s5cmd", page)
        # without the safetensors test there is nothing to compare with
        _, out = self.run_bench(tests="streamer_cuda", repeats=1)
        self.assertNotIn("speedup", out["results"][0])
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        self.assertIn("| Model Streamer | 5.00 | 8.0 |\n", page)  # and no speedup column
        self.assertNotIn("Speedup", page)

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
        self.assertIn("## Test on S3\n", page)
        self.assertIn("| Qwen | 1 × H200 | download + Safetensors | 35.00 (15.00 download + 20.00 load) |", page)
        self.assertIn("| | | Model Streamer | 5.00 | 8.0 | **7.00x** |", page)
        self.assertIn("Safetensors and InstantTensor read local files only. For them the model is first "
                      "downloaded to local disk with s5cmd, and then loaded", page)

    def test_without_s5cmd_only_the_streamer_reads_s3(self):
        a = argparse.Namespace(scratch_dir=None)
        with mock.patch.object(runner.shutil, "which", return_value=None):
            self.assertIn("s5cmd", runner.unavailable(a, "safetensors_cuda", remote=True))
            self.assertIsNone(runner.unavailable(a, "streamer_cuda", remote=True))
            self.assertIsNone(runner.unavailable(a, "safetensors_cuda"))

    def test_an_s3_model_without_a_local_copy_gets_a_reference_download(self):
        a = argparse.Namespace(model_dir="s3://b/Qwen", ref_dir=None, scratch_dir=None)
        with mock.patch.object(runner.shutil, "which", return_value="/bin/s5cmd"), \
                mock.patch("calibration.s3.s3_objects", return_value=[("s3://b/Qwen/m", 1 << 30)]), \
                mock.patch.object(runner.os, "makedirs"), \
                mock.patch.object(runner.shutil, "disk_usage", return_value=mock.Mock(free=10 << 30)), \
                mock.patch.object(runner, "download", return_value=(1.0, None)) as download, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.prepare_s3(a), os.path.join("bench_scratch", "ref"))
        self.assertEqual((a.scratch_dir, a.ref_dir), ("bench_scratch", os.path.join("bench_scratch", "ref")))
        download.assert_called_once_with("s3://b/Qwen", os.path.join("bench_scratch", "ref"), None)
        # too little room for the reference copy and a download is said before anything is fetched
        a = argparse.Namespace(model_dir="s3://b/Qwen", ref_dir=None, scratch_dir="/small")
        with mock.patch.object(runner.shutil, "which", return_value="/bin/s5cmd"), \
                mock.patch("calibration.s3.s3_objects", return_value=[("s3://b/Qwen/m", 8 << 30)]), \
                mock.patch.object(runner.os, "makedirs"), \
                mock.patch.object(runner.shutil, "disk_usage", return_value=mock.Mock(free=10 << 30)):
            with self.assertRaises(SystemExit) as stop:
                runner.prepare_s3(a)
        self.assertIn("needs twice its size", str(stop.exception))
        # a local copy given: nothing is downloaded
        a = argparse.Namespace(model_dir="s3://b/Qwen", ref_dir="/local", scratch_dir=None)
        self.assertIsNone(runner.prepare_s3(a))
        # the scratch directory is emptied before each download: it must not be the local copy
        a = argparse.Namespace(model_dir="s3://b/Qwen", ref_dir="/local/", scratch_dir="/local")
        with self.assertRaises(SystemExit) as stop:
            runner.prepare_s3(a)
        self.assertIn("same directory", str(stop.exception))

    def test_only_the_streamer_runs_by_default(self):
        self.run_bench(gpus=1, repeats=1, vllm=True, loaders=[], in_vllm=False)
        self.assertEqual(self.ran(), ["streamer_cpu", "streamer_cuda"])
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            self.assertNotIn("Speedup", f.read())  # nothing to compare with
        self.run_bench(gpus=8, nproc=2, repeats=1, vllm=True, loaders=[], in_vllm=False)
        self.assertEqual(self.ran(), ["streamer_dist"])
        # --vllm adds the same loads inside vLLM, --loaders the other loaders
        self.run_bench(gpus=1, repeats=1, vllm=True, loaders=[], in_vllm=True, device="cuda")
        self.assertEqual(self.ran(), ["streamer_cuda", "vllm_streamer"])
        self.run_bench(gpus=1, repeats=1, vllm=True, loaders=["safetensors", "streamer"], in_vllm=False, device="cuda")
        self.assertEqual(self.ran(), ["safetensors_cuda", "streamer_cuda"])
        # --tests names its tests whatever the defaults
        self.run_bench(gpus=1, repeats=1, loaders=[], in_vllm=False, tests="safetensors_cuda")
        self.assertEqual(self.ran(), ["safetensors_cuda"])

    def test_env_overrides_the_logging_defaults(self):
        code, _ = self.run_bench(tests="streamer_cpu", env=["RUNAI_STREAMER_LOG_LEVEL=DEBUG"])
        self.assertEqual(code, 0)
        self.assertEqual(self.calls[0][1]["RUNAI_STREAMER_LOG_LEVEL"], "DEBUG")
        self.assertEqual(self.calls[0][1]["RUNAI_STREAMER_LOG_TO_STDERR"], "1")

    def test_a_profile_from_another_s3_endpoint_is_not_reused(self):
        a = argparse.Namespace(model_dir="s3://b/Qwen")
        prof = {"host": {"hostname": profile.platform.node(), "s3": {"uri": "s3://b/x", "endpoint": "http://one"}},
                "ceilings_gibps": {"s3_get_parallel": 5.0}}
        with mock.patch("calibration.s3.s3_endpoint", return_value="http://one"):
            self.assertIsNone(profile.profile_mismatch(prof, a, True))
        with mock.patch("calibration.s3.s3_endpoint", return_value="http://two"):
            self.assertIn("endpoint", profile.profile_mismatch(prof, a, True))

    def test_a_rate_of_zero_is_near_no_ceiling(self):
        self.assertIsNone(profile.nearest(0.0, {"disk_direct_read_parallel": 10.0}))
        self.assertEqual(profile.nearest(8.0, {"a": 10.0, "b": 100.0}), "a")

    def test_dtype_is_compared_where_the_header_names_one(self):
        from benchmark import worker
        self.assertFalse(worker.differs(((2, 3), 12, "bfloat16"), ((2, 3), 12, "bfloat16")))
        self.assertTrue(worker.differs(((2, 3), 12, "float16"), ((2, 3), 12, "bfloat16")))
        self.assertTrue(worker.differs(((3, 2), 12, "bfloat16"), ((2, 3), 12, "bfloat16")))
        self.assertFalse(worker.differs(((2, 3), 12, "float16"), ((2, 3), 12, None)))

    def test_gpus_used_by_something_else_are_counted_and_reported(self):
        smi = lambda used: mock.patch.object(runner.subprocess, "run", return_value=mock.Mock(stdout=used))
        with mock.patch.object(runner, "gpu_memory_gib", return_value=[80.0, 80.0]):
            with smi("4\n4\n"):  # idle: a few MiB each
                self.assertEqual(runner.gpus_in_use("dist", 2), {})
                self.assertIsNone(runner.misfit("cuda", 61.0))
            with smi("40960\n4\n"):  # 40 GiB taken on GPU 0
                self.assertEqual(runner.gpus_in_use("cuda"), {0: 40.0})
                self.assertEqual(runner.gpus_in_use("cpu"), {})
                self.assertIn("of which 40 GiB is in use by something else", runner.misfit("cuda", 61.0))
                self.assertIsNone(runner.misfit("cuda", 20.0))
                self.assertIn("in use by something else", runner.misfit("dist", 100.0, 2))
            with smi("40960\n4\n"), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
                    mock.patch.object(runner, "gpu_memory_gib", return_value=[80.0]):
                self.assertEqual(runner.gpu_used_gib(), [4 / 1024])  # only the GPU the tests see
            with mock.patch.object(runner.subprocess, "run", side_effect=OSError):  # no nvidia-smi: not told
                self.assertIsNone(runner.gpu_used_gib())
                self.assertIsNone(runner.misfit("cuda", 61.0))
        with mock.patch.object(runner, "gpus_in_use", lambda group, nproc=None: {0: 40.0} if group == "cuda" else {}):
            _, out = self.run_bench(gpus=1, repeats=1, tests="streamer_cuda")
        self.assertEqual(out["environment"]["gpus_in_use"], {"0": 40.0})
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            self.assertIn("- GPUs already in use by something else during the run: GPU 0 (40.0 GiB)", f.read())

    def test_a_model_that_does_not_fit_the_gpus_is_not_run(self):
        with mock.patch.object(runner, "gpu_memory_gib", return_value=[80.0] * 8):
            self.assertIsNone(runner.misfit("cuda", 61.0))
            self.assertIn("one GPU has 80 GiB", runner.misfit("cuda", 100.0))
            self.assertIsNone(runner.misfit("cpu", 1000.0))  # host memory unknown: not checked
            with mock.patch.object(runner, "host_memory_gib", return_value=1000.0):
                self.assertIsNone(runner.misfit("cpu", 780.0))
                self.assertIn("1000 GiB of host memory", runner.misfit("cpu", 950.0))
            self.assertIsNone(runner.misfit("dist", 500.0, 8))
            self.assertIn("needs at least 4 ranks", runner.misfit("dist", 280.0, 2))
            self.assertNotIn("needs at least", runner.misfit("dist", 1000.0, 8))  # no rank count would do
        big = os.path.join(self.dir.name, "m.safetensors")
        with open(big, "r+b") as f:  # a sparse 2 GiB file with the same header
            f.truncate(2 << 30)
        with mock.patch.object(runner, "gpu_memory_gib", return_value=[1.0, 1.0]):
            # asked for nothing in particular: the GPU tests are left out and the rest runs
            code, out = self.run_bench(gpus=2, repeats=1)
            self.assertEqual((code, self.ran()), (0, ["safetensors_cpu", "streamer_cpu"]))
            # asked for the GPU: nothing to run, said with its own exit status
            self.assertEqual(self.run_bench(gpus=2, repeats=1, device="cuda")[0], 3)
            self.assertEqual(self.run_bench(gpus=2, repeats=1, nproc=2)[0], 3)
            self.assertIn("cannot run", self.run_bench(gpus=2, tests="streamer_cuda")[0])

    def test_nproc_runs_the_distributed_tests(self):
        code, out = self.run_bench(gpus=8, nproc=2, repeats=1)
        self.assertEqual(code, 0)
        self.assertEqual(self.ran(), ["safetensors_dist", "instanttensor_dist", DIST])
        self.assertTrue(all("torch.distributed.run" in c for c, _ in self.calls))
        self.assertEqual([s["group"] for s in out["results"]], ["dist"] * 3)
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        self.assertIn("| 2 × H200 | Safetensors |", page)

    def test_vllm_tests_time_the_weight_loading(self):
        code, out = self.run_bench(gpus=1, repeats=1, vllm=True, vllm_args={"trust_remote_code": True},
                                   tests="streamer_cuda,vllm_safetensors,vllm_instanttensor,vllm_streamer")
        self.assertEqual(code, 0)
        _, st, it, streamer = out["results"]
        self.assertEqual(st["runs"], [{"load_s": 40.0, "gibps": 1.0, "startup_s": 60.0, "model_load_s": 42.0,
                                       "gpu_gib": 40.2, "kv_cache_gib": 100.0}])
        # the KV cache vLLM fits afterwards: smaller where a loader left something on the GPU
        self.assertEqual((it["kv_cache_gib"], streamer["kv_cache_gib"]), (99.4, 100.0))
        self.assertEqual((streamer["load_s"], streamer["model_load_s"], streamer["group"]), (8.0, 10.0, "vllm"))
        # compared with safetensors in vLLM, not with the standalone safetensors test
        self.assertEqual([s["speedup"] for s in (st, it, streamer)], [1.0, 4.0, 5.0])
        self.assertNotIn("cpu_s_per_gib", streamer)
        by = {c[c.index("--child") + 1]: c for c, _ in self.calls}
        self.assertNotIn("--vllm-args", by["streamer_cuda"])
        self.assertEqual(json.loads(by["vllm_streamer"][by["vllm_streamer"].index("--vllm-args") + 1]),
                         {"trust_remote_code": True})
        with open(os.path.splitext(self.out)[0] + ".md") as f:
            page = f.read()
        # the loads inside vLLM have a table of their own, after the standalone ones
        alone, _, inside = page.partition("## Test in vLLM, from local disk")
        self.assertIn("| 1 × H200 | Model Streamer | 5.00 | 8.0 |\n", alone)
        self.assertNotIn("| Safetensors |", alone)
        self.assertIn("| 1 × H200 | Safetensors | 40.00 | 1.0 | 1.00x |", inside)
        self.assertIn("| | | Model Streamer | 8.00 | 5.0 | **5.00x** |", inside)
        self.assertNotIn("KV cache", page)  # kept in the results file only
        self.assertIn("- vllm: 0.34.0", page)
        # with tensor parallelism: the slowest rank's weight loading, and the GPU memory of all ranks
        code, out = self.run_bench(gpus=8, nproc=2, repeats=1, vllm=True, tests="vllm_dist_streamer")
        self.assertEqual(out["results"][0]["runs"], [{"load_s": 9.0, "gibps": 4.44, "startup_s": 28.0,
                                                      "model_load_s": 11.0, "gpu_gib": 40.2, "kv_cache_gib": 100.0}])
        self.assertTrue(all("torch.distributed.run" in c for c, _ in self.calls))
        # a rank that reported no time fails the run
        code, out = self.run_bench(launch=lambda cmd: ('RESULT {"startup_s": 1, "gib": 1, "tensors": 1, '
                                                       '"pkg": null, "errors": []}', "", None),
                                   gpus=1, repeats=1, vllm=True, tests="vllm_streamer")
        self.assertEqual(code, 1)
        self.assertIn("reported a weight loading time", out["results"][0]["runs"][0]["failed"])

    def test_vllm_tests_are_left_out_where_they_cannot_run(self):
        self.run_bench(gpus=1, repeats=1, device="cuda")  # vllm is not installed here
        self.assertFalse([t for t in self.ran() if t.startswith("vllm")])
        self.run_bench(gpus=1, repeats=1, device="cuda", vllm=True)
        self.assertEqual(self.ran()[-3:], ["vllm_safetensors", "vllm_instanttensor", "vllm_streamer"])
        a = argparse.Namespace(model_dir=self.dir.name)
        with mock.patch.object(runner.importlib.util, "find_spec", return_value=object()):
            self.assertIn("config.json", runner.unavailable(a, "vllm_streamer"))
            open(os.path.join(self.dir.name, "config.json"), "w").close()
            self.assertIsNone(runner.unavailable(a, "vllm_streamer"))
            self.assertIn("local directory", runner.unavailable(a, "vllm_streamer", remote=True))

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
        self.assertTrue(table.startswith("## Test on local disk"))  # every table of the page, under its heading
        with open(page) as f:
            self.assertIn(table, f.read())

    def test_a_setting_with_several_values_names_the_models(self):
        out = lambda model, env: {"model": {"path": model}, "settings": {"env": env}}
        self.assertEqual(markdown._settings([out("/m/A", {"X": "4", "Y": "1"}), out("/m/B", {"X": "4", "Y": "2"}),
                                             out("s3://b/A", {"Y": "1"})]), "X=4, Y=1 (A), 2 (B)")

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
            self.assertIn("| Model Streamer | failed | |\n", f.read())

    def test_cache_left_before_a_cold_run_is_a_warning(self):
        with mock.patch.object(runner, "evict", return_value=2 << 30):
            code, out = self.run_bench(tests="streamer_cpu", repeats=1)
        self.assertEqual(code, 0)
        self.assertEqual(out["results"][0]["runs"][0]["warnings"], ["2.0 GiB still in the page cache before the run"])
        # and the report does not call that run cold
        self.assertEqual(markdown._page_cache([out]), "not fully evicted before 1 run")
        self.assertEqual(markdown._page_cache([out, out]), "not fully evicted before 2 runs")  # counted, not merged
        out["results"][0]["runs"][0].pop("warnings")
        self.assertEqual(markdown._page_cache([out]), "cold before every run")
        out["model"]["path"] = "s3://b/Qwen"  # nothing is evicted or checked for an S3 model
        self.assertIsNone(markdown._page_cache([out]))

    def test_test_selection_errors(self):
        self.assertIn("needs --nproc", self.run_bench(tests=DIST)[0])
        self.assertIn("--nproc is for", self.run_bench(gpus=2, nproc=2, tests="streamer_cuda")[0])
        self.assertIn("unknown test", self.run_bench(tests="nope")[0])

    def test_an_s3_run_without_an_s3_ceiling_is_not_scored(self):
        self.assertEqual(profile.ceilings_for("streamer_cpu", PROFILE, remote=True)[0], None)  # no S3 ceiling


if __name__ == "__main__":
    unittest.main()
