# Model Streamer Benchmarks

`bench.py` measures how fast a model loads on your machine and compares it with
what the hardware can do. It loads the model with the streamer and with
`safetensors` baselines, each run in a fresh process on a cold page cache, and
checks the loaded tensors against `safetensors`. A run that returns wrong data
fails.

Each benchmark prints a summary table and writes `bench_results.json` and
`bench_results.html`: the median load time and GiB/s per test, CPU time per
GiB, peak memory, and the share of the machine's ceiling that was reached.

## Requirements

- Linux
- The packages in [`requirements.txt`](requirements.txt):
  `pip install -r benchmarks/requirements.txt`
  - `runai-model-streamer` (which brings `torch`) and `safetensors` are
    required; `bench.py` checks for them and says what is missing.
  - `vllm[instanttensor]` is for the vLLM tests. Without it they are left out
    and the rest still runs.
- torch built with NCCL and one GPU per rank, for the distributed test
- [`s5cmd`](https://github.com/peak/s5cmd) on `PATH`, only to calibrate S3

The benchmark measures the `runai_model_streamer` that this Python imports and
prints where it came from. To measure another build, install it or put its
directory first on `PYTHONPATH`.

## Layout

| Path | Purpose |
|---|---|
| `bench.py` | Command-line entry point |
| `calibrate.py` | Command-line entry point for calibration |
| `streamer_bench/runner.py` | Selects the tests and starts one measured process per run |
| `streamer_bench/worker.py` | The measured process: one load, timed and validated |
| `streamer_bench/files.py` | The model's files and their state in the page cache |
| `streamer_bench/profile.py` | Finds or makes the machine profile and scores a test against it |
| `streamer_bench/report.py` | Summary per test, the table and the results file |
| `streamer_bench/html.py` | The results as a self-contained HTML page |
| `streamer_bench/calibration/` | Measures the machine's ceilings, one module per stage: `memory`, `disk`, `gpu`, `s3`, and `host` for the machine's description |
| `requirements.txt` | The packages the benchmark loads with |
| `tests/` | Unit tests of the runner, the report and the calibration helpers |

## Running a benchmark

```bash
# every single-process test that applies on this machine, including vLLM
python3 benchmarks/bench.py /mnt/models/Qwen3-8B

# the distributed tests on 8 ranks of this node, one GPU each: the streamer, and vLLM with each loader
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --nproc 8

# one test, with a streamer setting
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --tests streamer_cuda \
    --env RUNAI_STREAMER_FS_STRATEGY=io_uring_buffered
```

| Option | Default | Meaning |
|---|---|---|
| `--nproc N` | off | Run the distributed tests on N ranks, one GPU each, in place of the single-process ones |
| `--tests a,b` | all that apply | Run only the named tests |
| `--env K=V` | none | Set a variable for the measured processes (repeatable) |
| `--warm` | cold | Load from the page cache instead of evicting first |
| `--ref-dir DIR` | none | Local copy of an S3 model, used for validation |
| `--profile FILE` | `machine_profile.json` | Machine profile made by `calibrate.py` |
| `--no-calibrate` | off | Never calibrate; the ceiling columns are then empty |
| `--repeats N` | 3 | Runs per test |
| `--out FILE` | `bench_results.json` | Results file; the HTML page gets the same name with `.html` |
| `--log-dir DIR` | none | Keep each run's output |

Runs are interleaved across the tests (every test once, then every test
again), so drift in the machine or the storage affects all of them alike.

## Tests

| Test | What it loads |
|---|---|
| `streamer_cpu` | Streamer to host memory, `owned=True` (the default) |
| `streamer_cpu_notowned` | The same with `owned=False` |
| `streamer_cuda` | Streamer to GPU |
| `safeopen_page_in` | `safetensors.safe_open`, one byte touched per page |
| `safeopen_to_cuda` | `safe_open` + `.to("cuda")`, the default loader's shape |
| `streamer_dist` | Streamer with `is_distributed=True` on `--nproc` ranks |
| `vllm_default` | A vLLM engine start with vLLM's default loader |
| `vllm_streamer` | The same with `load_format="runai_streamer"` |
| `vllm_instanttensor` | The same with `load_format="instanttensor"` |
| `vllm_dist_default`, `vllm_dist_streamer`, `vllm_dist_instanttensor` | The three vLLM tests with tensor parallelism over `--nproc` ranks |

Without `--tests`, the GPU tests run when a GPU is present, and only the
streamer tests run for a model on S3. A vLLM test is left out, with a note,
when what it needs is not installed or the model has no `config.json`.

In `streamer_dist` every rank streams all files and receives every tensor.
Each rank keeps a 1/N share of the tensors on its GPU, so device memory is
under real pressure, and drops the rest. The load time is measured from a
barrier before the load to a barrier after the last tensor, which makes it the
slowest rank's time. Each rank needs its own GPU, so `--nproc` cannot be more
than the number of GPUs on the machine.

### vLLM

The vLLM tests load the model in vLLM itself, once per loader: its default
loader, the streamer, and InstantTensor. Each is a fresh process on a cold
page cache like every other test. They need a GPU, `vllm` installed
(`pip install vllm`) and the model's `config.json` next to the weights;
`vllm_instanttensor` also needs `pip install 'vllm[instanttensor]'`.

With `--nproc N` the `vllm_dist_*` tests start vLLM with
`tensor_parallel_size=N` under torchrun, using vLLM's external launcher, so
every rank loads its share onto its own GPU. The streamer runs in its
distributed mode there (`model_loader_extra_config={"distributed": true}`),
which loads straight to the GPUs. The load time is the slowest rank's, and
`gpu_gib` is the sum over the ranks. The model's attention head count has to
be divisible by N.

vLLM's default loader reads the weights in one thread unless its
`enable_multithread_load` option is set, which this benchmark leaves off, as
vLLM does.

The load time is the one vLLM logs as "Model loading took", which covers
building the model and loading its weights onto the GPU. The results also
give `startup_s`, the whole engine start, and `vllm_speedup`, the default
loader's time over the streamer's. The engine starts in eager mode and without
a tokenizer to keep the start short; neither changes how the weights are
loaded. CPU time and host memory are not measured for these tests, and the
tensors are not compared with `safetensors`: a run is good when the engine
came up. Each run records `gpu_gib`, the GPU memory vLLM says the model took,
and carries a warning when that is well below the size of the weights.

## Machine ceilings

The ceilings come from a machine profile: the disk, host memory and
host-to-GPU rates that `calibrate.py` measures. On the first run `bench.py`
calibrates on the model's directory and keeps the profile in
`machine_profile.json`, which later runs on the same host and storage reuse.
Calibration writes 16 GiB of temporary files to that directory and removes
them, and uses about 40 GiB of memory; it uses less of both on a smaller
machine.

Calibration needs only the standard library; torch is used for the
host-to-GPU rates when it is installed. Each ceiling is the median of three
runs, in GiB/s:

| Ceiling | What is measured |
|---|---|
| `disk_direct_read_parallel` | O_DIRECT reads, one thread per file, into memory already faulted in |
| `disk_direct_read_fresh_dest` | The same into untouched memory, so the kernel faults it in during the read |
| `disk_direct_read_single` | One thread, one file |
| `disk_buffered_read_cold` | Buffered reads after evicting the page cache |
| `host_first_touch_1t`, `_nt` | First touch of fresh memory, with one and many threads |
| `host_memcpy_1t`, `_nt` | Memory-to-memory copy, with one and many threads |
| `h2d_pinned`, `h2d_pageable` | Copy to GPU 0 from pinned and from pageable memory |
| `s3_get_single` | One S3 object, one part in flight |
| `s3_get_parallel` | Every object at once, the best of a sweep over parts in flight and part size |

To calibrate separately, for example when the model directory is read-only,
run it yourself and pass the profile:

```bash
python3 benchmarks/calibrate.py --path /mnt/models/scratch --out machine_profile.json
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --profile machine_profile.json
```

Each test is scored against one ceiling:

| Test | Ceiling |
|---|---|
| Host tests and `streamer_dist` | `disk_direct_read_parallel` |
| Single-process GPU tests | The lower of that and `h2d_pinned` |
| Any test with `--warm` | `host_memcpy_nt` in place of the disk |
| Any test on S3 | `s3_get_parallel` in place of the disk |

The table also names the nearest ceiling: the stage in the profile whose rate
is closest to what the run achieved. It is a hint at what limited the run, not
a proof.

## S3 and network filesystems

For a model on S3, pass the `s3://bucket/prefix` and a local copy of the same
files with `--ref-dir`. The endpoint and credentials are taken from the
environment. S3 is not calibrated automatically, because the sweep reads the
whole model many times. Calibrate it yourself and pass the profile:

```bash
python3 benchmarks/calibrate.py --skip disk --s3-uri s3://bucket/prefix --out machine_profile.json
```

A model on NFS or another network filesystem is benchmarked like a local one.
The page cache is dropped on this machine only, so a cache on the file server
can still make a repeated run faster than a truly cold load. Drop the server's
cache between runs when you need server-cold numbers.

## Results

The results and the machine profile are written to the directory you run the
benchmark from. Their default names are in `.gitignore`.

`bench_results.html` shows the same results as a page: a bar per test against
its ceiling, a table of every number, and the runs. Below them it lists what
the numbers were measured on: the machine (kernel, CPUs, memory, GPUs, the
model's storage), the versions of the streamer, vLLM, torch and safetensors,
the settings, and the calibration with every ceiling. It is one file with no
external resources, so it can be copied off the machine and opened anywhere.
To make it again from a results file:

```bash
cd benchmarks && python3 -m streamer_bench.html /path/to/bench_results.json
```

`bench_results.json` holds the same once at the top (`host`, `environment`,
`versions`, `streamer`, `model`, `settings`, and `calibration`, which is the
machine profile without each stage's single runs), then one entry per test:

```json
{
  "test": "streamer_cuda",
  "ok": true,
  "load_s": 2.93,
  "gibps": 5.21,
  "gibps_min": 5.1,
  "gibps_max": 5.3,
  "cpu_s_per_gib": 0.21,
  "peak_rss_gib": 3.4,
  "fs_strategy": ["io_uring"],
  "ceiling": {"name": "h2d_pinned", "gibps": 6.0, "pct": 86.8, "nearest": "h2d_pinned"},
  "runs": [
    {"load_s": 2.931, "gibps": 5.21, "cpu_s": 3.2, "peak_rss_gib": 3.4}
  ]
}
```

- `ok` is true only when every run gave a result and none returned wrong data.
- `load_s` and `gibps` are medians over the runs.
- `ceiling` is absent, and `calibration` is null, when the benchmark ran
  without a profile.
- A vLLM test has `startup_s` in place of `cpu_s_per_gib` and `peak_rss_gib`.
- A run carries `errors` (wrong data) or `warnings` (for example a cache that
  was not cold) only when it has some. A run that gave no result is
  `{"failed": "..."}`.
- A `streamer_dist` run adds `rank_s` (each rank's time) and `peak_gpu_gib`.
  Its `cpu_s` is summed over the ranks and `peak_rss_gib` is the largest
  rank's.

With `--log-dir`, each run's log also holds the measured process's full
record, including page faults and per-thread system time.

## Local checks

```bash
python3 -m unittest discover -s benchmarks/tests
```

The tests stub the measured processes, so they need no GPU, torch or Linux.
