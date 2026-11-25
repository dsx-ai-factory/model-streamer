# Model Streamer Benchmarks

`bench.py` measures how fast the streamer loads a model on your machine and
compares it with what the hardware can do. On request it measures two other
loaders the same way, `safetensors` and
[InstantTensor](https://github.com/scitix/InstantTensor), and the same loads
inside vLLM. Every run is a fresh process on a cold page cache, and the
loaded tensors are checked against `safetensors`. A run that returns wrong
data fails.

## Requirements

- Linux
- The packages in [`requirements.txt`](requirements.txt):
  `pip install -r benchmarks/requirements.txt`
  - `runai-model-streamer` (which brings `torch`) and `safetensors` are
    required; `bench.py` checks for them and says what is missing.
  - `instanttensor` is for the InstantTensor tests. Without it they are left
    out and the rest still runs.
  - `s5cmd` is for a model on S3: it downloads the model for the loaders that
    read only local files, and calibrates S3.
  - `vllm` is for the vLLM tests. Without it they are left out and the rest
    still runs. It pins its own `torch`, so installing it can change the
    `torch` of that environment.
  - `huggingface_hub` is for `suite.sh`, to download a model that is missing.
- torch built with NCCL and one GPU per rank, for the distributed tests

The benchmark measures the `runai_model_streamer` that this Python imports and
prints where it came from. To measure another build, install it or put its
directory first on `PYTHONPATH`.

## Running the benchmark

```bash
# the streamer, to host memory and to one GPU
python3 benchmarks/bench.py /mnt/models/Qwen3-8B

# to one GPU only
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --device cuda

# to the GPUs of 8 ranks of this node, one GPU each
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --nproc 8

# compared with safetensors and InstantTensor (or name them: --loaders safetensors,streamer)
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --loaders all

# and the same loads inside vLLM: on one GPU, or with --nproc 8 in tensor parallel 8
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --loaders all --vllm

# one test, with a streamer setting
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --tests streamer_cuda \
    --env RUNAI_STREAMER_FS_STRATEGY=io_uring_buffered
```

| Option | Default | Meaning |
|---|---|---|
| `--nproc N` | off | Run the distributed tests on N ranks, one GPU each, in place of the single-process ones |
| `--device cpu\|cuda` | both | Run the single-process tests to this destination only |
| `--loaders a,b` | `streamer` | The loaders to measure: `safetensors`, `instanttensor`, `streamer`, or `all` |
| `--vllm` | off | Also time the same loads inside vLLM |
| `--tests a,b` | chosen by the options above | Run exactly the named tests |
| `--env K=V` | none | Set a variable for the measured processes (repeatable) |
| `--vllm-args JSON` | none | Extra vLLM engine options for the vLLM tests, for example `'{"trust_remote_code": true}'` |
| `--ref-dir DIR` | downloaded | Local copy of an S3 model, used for validation |
| `--scratch-dir DIR` | `bench_scratch` | Where an S3 model is downloaded to |
| `--profile FILE` | `machine_profile.json` | Machine profile made by `calibrate.py` |
| `--no-calibrate` | off | Never calibrate; the ceiling columns are then empty |
| `--repeats N` | 3 | Runs per test |
| `--out FILE` | `bench_results.json` | Results file; the Markdown report gets the same name with `.md` |
| `--log-dir DIR` | none | Keep each run's output |

Runs are interleaved across the tests (every test once, then every test
again), so drift in the machine or the storage affects all of them alike.

When another process already holds memory on a GPU the tests load to, the
benchmark warns before it starts and records it in the results, because the
numbers are then not those of an idle machine.

## Tests

A test is one loader bringing the whole model to one destination:

| Destination | Safetensors | InstantTensor | Model Streamer |
|---|---|---|---|
| Host memory | `safetensors_cpu` | not supported | `streamer_cpu` |
| One GPU | `safetensors_cuda` | `instanttensor_cuda` | `streamer_cuda` |
| The GPUs of `--nproc` ranks | `safetensors_dist` | `instanttensor_dist` | `streamer_dist` |

| Loader | To host memory | To one GPU | To the ranks' GPUs |
|---|---|---|---|
| Safetensors | `safe_open`, then every page of each tensor is touched | `safe_open` + `.to("cuda")` per tensor | Every rank does the same load itself |
| InstantTensor | | `instanttensor.safe_open(files, device=...)` | The same with the ranks' process group |
| Model Streamer | `stream_files(files, device="cpu")` | `stream_files(files, device="cuda:0")` | `stream_files(files, device="cuda:N", is_distributed=True)`, N being the rank's GPU |

The tests are named in [`benchmark/__init__.py`](benchmark/__init__.py)
(`TESTS`). The loads themselves are the
`load` function in [`benchmark/worker.py`](benchmark/worker.py);
`single` and `distributed` wrap it with the timing and the validation
(`check`).

By default only the streamer's tests run. `--loaders` adds the other loaders,
`--vllm` adds the vLLM tests below, and `--tests` names tests one by one.
Without `--tests`, a test runs when the model fits where it goes: host memory
or one GPU has to hold the whole model, and with N ranks each GPU keeps a 1/N
share of it. The GPU tests also need a GPU. A test that cannot run is left out with a note, for example an
InstantTensor test when `instanttensor` is not installed.

In a distributed test every rank receives every tensor on its GPU, as the
ranks of a tensor-parallel engine do. Each rank keeps a 1/N share of the
tensors, so device memory is under real pressure, and drops the rest. The
load time is measured from a barrier before the load to a barrier after the
last tensor, which makes it the slowest rank's time. Each rank needs its own
GPU, so `--nproc` cannot be more than the number of GPUs on the machine.

The clock starts where it does for a loader inside an engine:

- The job's NCCL group has no communicator yet, as in vLLM, which creates its
  group without a `device_id` and runs no collective before it loads the
  weights. A loader that sends on the job's group pays for setting up its
  communicator inside the measured time.
- NCCL itself has already started in every rank, as in vLLM, which builds a
  communicator for inference before it loads the weights. The test builds one
  on a group of its own before the clock. Without it each loader would also
  pay about 2 s for NCCL's first start in the process.

The barriers that line the ranks up run on a separate gloo group.

### vLLM tests

With `--vllm` the same loaders are also timed inside vLLM, which shows what a
server start gains from each:

| Destination | Safetensors | InstantTensor | Model Streamer |
|---|---|---|---|
| One GPU | `vllm_safetensors` | `vllm_instanttensor` | `vllm_streamer` |
| `--nproc` GPUs, tensor parallel | `vllm_dist_safetensors` | `vllm_dist_instanttensor` | `vllm_dist_streamer` |

Each test starts a vLLM engine on the model with the loader selected
(vLLM's `--load-format`), and the load
time is the time vLLM spends in its weight loading step, on the slowest rank.
The results also keep vLLM's own "Model loading took" time and the whole
start-up time.

The report gives these tests a table of their own. The results file also keeps
the KV cache vLLM then fits on each GPU (`kv_cache_gib`, the smallest over the
ranks). Whatever a loader leaves on the GPU comes out of it:
a loader that sends on vLLM's process group leaves that group's NCCL
communicator allocated, about 0.6 GiB per GPU on a B200.

- They need `vllm` installed and the model's `config.json`, and a model on
  local disk. Otherwise they are left out with a note.
- The vLLM used is the one installed in this Python; its version is in the
  report. A streamer build on `PYTHONPATH` is used in place of the one vLLM
  was installed with.
- The timer wraps vLLM's model loader classes. If a later vLLM renames them,
  these tests fail with "ranks reported no weight loading time" and the other
  tests are not affected.
- A model that needs engine options gets them through `--vllm-args`.

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
  copy you already have instead, which is what `suite.sh` does.
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
measured read rate, the versions of each loader), and tables with a row per
loader giving its load time, throughput and speedup: one for local disk, one
for the loads inside vLLM and one for S3. Speedup, shown when Safetensors was
one of the loaders, is against
Safetensors to the same destination, and with several GPUs the load time is
the slowest rank's. To make it again from a results file, or one report from several:

```bash
cd benchmarks && python3 -m benchmark.markdown /path/to/bench_results.json
cd benchmarks && python3 -m benchmark.markdown a.json b.json --out report.md
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

## Layout

| Path | Purpose |
|---|---|
| `bench.py` | Command-line entry point: one model |
| `suite.sh` | Our fixed run of three models through `bench.py` (see the end of this page) |
| `calibrate.py` | Command-line entry point for calibration |
| `benchmark/__init__.py` | The loaders and the tests (`TESTS`) |
| `benchmark/runner.py` | Selects the tests and starts one measured process per run |
| `benchmark/worker.py` | The measured process: one load, timed and validated |
| `benchmark/files.py` | The model's files and their state in the page cache |
| `benchmark/profile.py` | Finds or makes the machine profile and scores a test against it |
| `benchmark/report.py` | Summary per test, the table and the results file |
| `benchmark/markdown.py` | Results as a Markdown report |
| `calibration/` | Measures the machine's ceilings, one module per stage: `memory`, `disk`, `gpu`, `s3`, and `host` for the machine's description |
| `requirements.txt` | The packages the benchmark loads with |
| `tests/` | Unit tests of the runner, the report and the calibration helpers |

## Local checks

```bash
python3 -m unittest discover -s benchmarks/tests
```

The tests stub the measured processes, so they need no GPU, torch or Linux.

## The suite script

`suite.sh` is the fixed run we repeat ourselves: the same three models every
time through `bench.py`, from local disk and from S3, in one report. It is a
convenience around the commands above, not another way to measure.

```bash
benchmarks/suite.sh /mnt/models s3://bench
```

The cases, each with every loader, first from local disk and then from S3.
Where `vllm` is installed, the same loads then run inside vLLM, from local
disk only, as cases of their own:

| Model | Loaded to |
|---|---|
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) (61 GiB) | One GPU |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) (641 GiB) | Every GPU, one rank each |
| [Kimi-K3-REAP](https://huggingface.co/runrunway/Kimi-K3-REAP-448experts) (780 GiB) | Every GPU, and host memory |

The first argument is the directory the models are in. The second, optional,
is where they are on S3; without it only the local disk is used:

- **A model missing from the directory** is downloaded: from the bucket when
  it has the model, and from Hugging Face otherwise, at a pinned revision.
  A model that the directory has no room for is left out with a note, and the
  others still run.
- **The bucket is only read.** The S3 half runs for the models the bucket
  holds, each under its name (`s3://bench/Qwen3-32B/`). Nothing is uploaded: a
  model that is not there is left out of the S3 half with a note. The endpoint
  and credentials come from the environment (`AWS_ENDPOINT_URL`,
  `AWS_ACCESS_KEY_ID`, ...).
- **Calibration** happens on first use, for the disk and for S3, as with
  `bench.py`.

On a machine with one GPU the cases that load to every GPU are left out with a
note.

Anything after the directory and the bucket is passed on to every `bench.py`
run:

```bash
benchmarks/suite.sh /mnt/models s3://bench --repeats 1 --loaders safetensors,streamer
```

A few things can be set through the environment:

| Variable | Default | Meaning |
|---|---|---|
| `NPROC` | every GPU | Ranks for the multi-GPU cases |
| `OUT_DIR` | `bench_suite` | A directory named after the date is made in it |
| `SCRATCH_DIR` | `MODELS_DIR/bench_scratch` | Where an S3 model is downloaded to for the loaders that read only local files |
| `PYTHON` | `python3` | The Python that has the loaders installed |
| `S3_OPTIONS` | none | `bench.py` options for the S3 runs in one process, for example `--env RUNAI_STREAMER_OBJ_CONCURRENCY=4` |
| `S3_MULTI_OPTIONS` | none | The same for the S3 runs to every GPU, where the setting is per rank |

The results go to `OUT_DIR/<date>/`: for each case `<model>_<case>.json`
(`..._s3.json` for the S3 run), its report and its logs, and the combined
`report.md` with the test setting, a table for local disk and a table for S3:

| Model | Destination | Loader | Load time (s) | Throughput (GiB/s) | Speedup |
|---|---|---|---|---|---|
| Qwen3-32B | 1 × NVIDIA B200 | Safetensors | 5.67 | 10.8 | 1.00x |
| | | InstantTensor | 2.48 | 24.6 | 2.29x |
| | | Model Streamer | 1.60 | 38.1 | **3.54x** |

Speedup is against Safetensors to the same destination.
