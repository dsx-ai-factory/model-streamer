# Model Streamer Benchmarks

`bench.py` measures how fast a model loads on your machine with three loaders,
`safetensors`, [InstantTensor](https://github.com/scitix/InstantTensor) and the
streamer, and compares each with what the hardware can do. Every run is a
fresh process on a cold page cache, and the loaded tensors are checked against
`safetensors`. A run that returns wrong data fails.

`suite.py` is a harness on top of it: it runs `bench.py` on several models and
puts the results in one table of load time, throughput and speedup.

## Requirements

- Linux
- The packages in [`requirements.txt`](requirements.txt):
  `pip install -r benchmarks/requirements.txt`
  - `runai-model-streamer` (which brings `torch`) and `safetensors` are
    required; `bench.py` checks for them and says what is missing.
  - `instanttensor` is for the InstantTensor tests. Without it they are left
    out and the rest still runs.
- torch built with NCCL and one GPU per rank, for the distributed tests
  - `s5cmd` is for a model on S3: it downloads the model for the loaders that
    read only local files, and calibrates S3.

The benchmark measures the `runai_model_streamer` that this Python imports and
prints where it came from. To measure another build, install it or put its
directory first on `PYTHONPATH`.

## Layout

| Path | Purpose |
|---|---|
| `bench.py` | Command-line entry point: one model |
| `suite.py` | Harness: several models through `bench.py`, one report |
| `calibrate.py` | Command-line entry point for calibration |
| `streamer_bench/__init__.py` | The loaders and the tests (`TESTS`) |
| `streamer_bench/runner.py` | Selects the tests and starts one measured process per run |
| `streamer_bench/worker.py` | The measured process: one load, timed and validated |
| `streamer_bench/files.py` | The model's files and their state in the page cache |
| `streamer_bench/profile.py` | Finds or makes the machine profile and scores a test against it |
| `streamer_bench/report.py` | Summary per test, the table and the results file |
| `streamer_bench/markdown.py` | Results as a Markdown report |
| `streamer_bench/calibration/` | Measures the machine's ceilings, one module per stage: `memory`, `disk`, `gpu`, `s3`, and `host` for the machine's description |
| `requirements.txt` | The packages the benchmark loads with |
| `tests/` | Unit tests of the runner, the report and the calibration helpers |

## Running the suite

```bash
python3 benchmarks/suite.py \
    --gpu /mnt/models/Qwen3-32B \
    --multi-gpu /mnt/models/DeepSeek-R1 \
    --cpu /mnt/models/Kimi-K3
```

Each option is one case: which model, and where it is loaded to. Any of them
can be left out, and each can be given more than once. A model is a directory
or an `s3://bucket/prefix`. Before anything is loaded, the benchmark checks
that the model fits the destination and says so if it does not.

| Option | Case |
|---|---|
| `--gpu MODEL` | The model loaded to one GPU |
| `--multi-gpu MODEL` | The model loaded to the GPUs of `--nproc` ranks (default: every GPU) |
| `--cpu MODEL` | The model loaded to host memory |

Every case is one `bench.py` run with each loader that applies. `--repeats`,
`--loaders`, `--env`, `--profile`, `--scratch-dir` and `--no-calibrate` are
passed on to it. The results go to `--out-dir` (default `bench_suite`): for
each case `<model>_<case>.json`, its report and its logs, and the combined
`report.md`, which has the test setting and this table:

| Model | Destination | Loader | Load time (s) | Throughput (GiB/s) | Speedup |
|---|---|---|---|---|---|
| Qwen3-32B | 1 × NVIDIA B200 | Safetensors | 5.67 | 10.8 | 1.00x |
| | | InstantTensor | 2.48 | 24.6 | 2.29x |
| | | Model Streamer | 1.60 | 38.1 | **3.54x** |

Speedup is against Safetensors to the same destination.

## Running one model

```bash
# to host memory and to one GPU, with each loader
python3 benchmarks/bench.py /mnt/models/Qwen3-8B

# to one GPU only
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --device cuda

# to the GPUs of 8 ranks of this node, one GPU each
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --nproc 8

# one test, with a streamer setting
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --tests streamer_cuda \
    --env RUNAI_STREAMER_FS_STRATEGY=io_uring_buffered
```

| Option | Default | Meaning |
|---|---|---|
| `--nproc N` | off | Run the distributed tests on N ranks, one GPU each, in place of the single-process ones |
| `--device cpu\|cuda` | both | Run the single-process tests to this destination only |
| `--loaders a,b` | all | Run only these loaders: `safetensors`, `instanttensor`, `streamer` |
| `--tests a,b` | all that apply | Run only the named tests |
| `--env K=V` | none | Set a variable for the measured processes (repeatable) |
| `--warm` | cold | Load from the page cache instead of evicting first |
| `--ref-dir DIR` | downloaded | Local copy of an S3 model, used for validation |
| `--scratch-dir DIR` | `bench_scratch` | Where an S3 model is downloaded to |
| `--profile FILE` | `machine_profile.json` | Machine profile made by `calibrate.py` |
| `--no-calibrate` | off | Never calibrate; the ceiling columns are then empty |
| `--repeats N` | 3 | Runs per test |
| `--out FILE` | `bench_results.json` | Results file; the Markdown report gets the same name with `.md` |
| `--log-dir DIR` | none | Keep each run's output |

Runs are interleaved across the tests (every test once, then every test
again), so drift in the machine or the storage affects all of them alike.

## Tests

A test is one loader bringing the whole model to one destination:

| Destination | Safetensors | InstantTensor | Model Streamer |
|---|---|---|---|
| Host memory | `safetensors_cpu` | not supported by InstantTensor | `streamer_cpu` |
| One GPU | `safetensors_cuda` | `instanttensor_cuda` | `streamer_cuda` |
| The GPUs of `--nproc` ranks | `safetensors_dist` | `instanttensor_dist` | `streamer_dist` |

| Loader | To host memory | To one GPU | To the ranks' GPUs |
|---|---|---|---|
| Safetensors | `safe_open`, then every page of each tensor is touched | `safe_open` + `.to("cuda")` per tensor | Every rank does the same load itself |
| InstantTensor | | `instanttensor.safe_open(files, device=...)` | The same with the ranks' process group |
| Model Streamer | `stream_files(files, device="cpu")` | `stream_files(files, device="cuda")` | `stream_files(files, device="cuda", is_distributed=True)` |

The tests are named in [`streamer_bench/__init__.py`](streamer_bench/__init__.py)
(`TESTS`). The loads themselves are the
`load` function in [`streamer_bench/worker.py`](streamer_bench/worker.py);
`single` and `distributed` wrap it with the timing and the validation
(`check`).

Without `--tests`, the GPU tests run when a GPU is present and the model fits:
one GPU has to hold the whole model, and with N ranks each GPU keeps a 1/N
share of it. A test that cannot run is left out with a note, for example an
InstantTensor test when `instanttensor` is not installed.

In a distributed test every rank receives every tensor on its GPU, as the
ranks of a tensor-parallel engine do. Each rank keeps a 1/N share of the
tensors, so device memory is under real pressure, and drops the rest. The
load time is measured from a barrier before the load to a barrier after the
last tensor, which makes it the slowest rank's time. Each rank needs its own
GPU, so `--nproc` cannot be more than the number of GPUs on the machine.

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
| Host tests and the distributed tests | `disk_direct_read_parallel` |
| Single-process GPU tests | The lower of that and `h2d_pinned` |
| Any test with `--warm` | `host_memcpy_nt` in place of the disk |
| Any test on S3 | `s3_get_parallel` in place of the disk |

The table also names the nearest ceiling: the stage in the profile whose rate
is closest to what the run achieved. It is a hint at what limited the run, not
a proof.

## S3 and network filesystems

For a model on S3, pass its `s3://bucket/prefix`. The endpoint and credentials
are taken from the environment, and `s5cmd` (in `requirements.txt`) has to be
on `PATH`:

```bash
python3 benchmarks/bench.py s3://bucket/Qwen3-32B
```

The streamer reads the model from S3 directly. Safetensors and InstantTensor
read local files only, so their tests first download the model with
[`s5cmd`](https://github.com/peak/s5cmd) and then load it, as it was
downloaded. Their load time is the two together, and `download_s` in the
results says how much of it the download was.

Two things are fetched for you, into the scratch directory (`bench_scratch`
in the current directory, or `--scratch-dir DIR`), which needs free space of
twice the model's size:

- A reference copy of the model, for the tensor names and for validation. It
  is downloaded once and removed after the run. `--ref-dir DIR` uses a local
  copy you already have instead; for `suite.py`, `--ref-root DIR` does the
  same with `DIR/<model name>` per model.
- The copy each download-then-load run makes, removed after each run.

On first use the machine is calibrated for S3 as well: a short sweep with
`s5cmd` over the first 64 GiB of the model. The profile then serves every
model in the same bucket. To calibrate more thoroughly, run it yourself and
pass the profile:

```bash
python3 benchmarks/calibrate.py --path /mnt/scratch --s3-uri s3://bucket/Qwen3-32B --out machine_profile.json
```

A model on NFS or another network filesystem is benchmarked like a local one.
The page cache is dropped on this machine only, so a cache on the file server
can still make a repeated run faster than a truly cold load. Drop the server's
cache between runs when you need server-cold numbers.

## Results

The results and the machine profile are written to the directory you run the
benchmark from. Their default names are in `.gitignore`.

`bench_results.md` is the report: the test setting (the hardware, the storage's
measured read rate, the versions of each loader), a table with a row per
loader giving its load time, throughput and speedup. Speedup is against
Safetensors to the same destination, and with several GPUs the load time is
the slowest rank's. To make it again from a results file, or one report from several:

```bash
cd benchmarks && python3 -m streamer_bench.markdown /path/to/bench_results.json
cd benchmarks && python3 -m streamer_bench.markdown a.json b.json --out report.md
```

`bench_results.json` holds everything that was measured. At the top are
`host`, `environment`, `versions`, `streamer`, `model`, `settings`, and
`calibration` (the machine profile without each stage's single runs). Then
comes one entry per test:

```json
{
  "test": "streamer_cuda",
  "loader": "streamer",
  "group": "cuda",
  "ok": true,
  "load_s": 2.93,
  "gibps": 5.21,
  "gibps_min": 5.1,
  "gibps_max": 5.3,
  "cpu_s_per_gib": 0.21,
  "peak_rss_gib": 3.4,
  "fs_strategy": ["io_uring"],
  "speedup": 3.61,
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
- `group` is the destination: `cpu`, `cuda` or `dist`. `speedup` is against the
  safetensors test of the same group, and is absent when that test did not run.
- A run carries `errors` (wrong data) or `warnings` (for example a cache that
  was not cold) only when it has some. A run that gave no result is
  `{"failed": "..."}`.
- A distributed run adds `rank_s` (each rank's time) and `peak_gpu_gib`.
  Its `cpu_s` is summed over the ranks and `peak_rss_gib` is the largest
  rank's.

With `--log-dir`, each run's log also holds the measured process's full
record, including page faults and per-thread system time.

## Local checks

```bash
python3 -m unittest discover -s benchmarks/tests
```

The tests stub the measured processes, so they need no GPU, torch or Linux.
