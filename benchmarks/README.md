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
- Python with `torch`, `safetensors` and `runai-model-streamer` installed
- torch built with NCCL and one GPU per rank, for the distributed test
- [`s5cmd`](https://github.com/peak/s5cmd) on `PATH`, only to calibrate S3

The benchmark measures the `runai_model_streamer` that this Python imports and
prints where it came from. To measure another build, install it or put its
directory first on `PYTHONPATH`.

## Layout

| Path | Purpose |
|---|---|
| `bench.py` | Command-line entry point |
| `calibrate.py` | Measures the machine's ceilings into a profile; standard library only |
| `streamer_bench/runner.py` | Selects the tests and starts one measured process per run |
| `streamer_bench/worker.py` | The measured process: one load, timed and validated |
| `streamer_bench/files.py` | The model's files and their state in the page cache |
| `streamer_bench/profile.py` | Finds or makes the machine profile and scores a test against it |
| `streamer_bench/report.py` | Summary per test, the table and the results file |
| `streamer_bench/html.py` | The results as a self-contained HTML page |
| `tests/` | Unit tests of the runner and the report |

## Running a benchmark

```bash
# every single-process test that applies on this machine
python3 benchmarks/bench.py /mnt/models/Qwen3-8B

# the distributed load on 8 ranks of this node
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --nproc 8

# one test, with a streamer setting
python3 benchmarks/bench.py /mnt/models/Qwen3-8B --tests streamer_cuda \
    --env RUNAI_STREAMER_FS_STRATEGY=io_uring_buffered
```

| Option | Default | Meaning |
|---|---|---|
| `--nproc N` | off | Run the distributed test on N ranks; without `--tests`, only that test |
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

Without `--tests`, the GPU tests run when a GPU is present, and only the
streamer tests run for a model on S3.

In `streamer_dist` every rank streams all files and receives every tensor.
Each rank keeps a 1/N share of the tensors on its GPU, so device memory is
under real pressure, and drops the rest. The load time is measured from a
barrier before the load to a barrier after the last tensor, which makes it the
slowest rank's time.

## Machine ceilings

The ceilings come from a machine profile: the disk, host memory and
host-to-GPU rates that `calibrate.py` measures. On the first run `bench.py`
calibrates on the model's directory and keeps the profile in
`machine_profile.json`, which later runs on the same host and storage reuse.
Calibration writes 16 GiB of temporary files to that directory and removes
them, and uses about 40 GiB of memory; it uses less of both on a smaller
machine.

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
its ceiling, a table of every number, and the runs. It is one file with no
external resources, so it can be copied off the machine and opened anywhere.
To make it again from a results file:

```bash
cd benchmarks && python3 -m streamer_bench.html /path/to/bench_results.json
```

`bench_results.json` holds the host, the streamer, the model and the settings
once, then one entry per test:

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
- `ceiling` is absent when the benchmark ran without a profile.
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
