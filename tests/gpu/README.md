# Real-GPU tests

Streams synthetic safetensors models, and one small real model, to real GPUs with the model streamer and compares
every tensor with what the manifest recorded. CI otherwise runs on CPU and mock devices only, so this is where a real
CUDA copy, real NCCL and the real storage path are exercised together.

The `Nightly GPU tests` workflow (`.github/workflows/nightly-gpu.yaml`) runs every night at 00:17 UTC. A scheduled run
is skipped when nothing under `cpp/`, `py/`, `tests/gpu/` or `.devcontainer/` (documentation files aside) changed since
the last scheduled run that passed, so a failed run is repeated the next night. With `workflow_dispatch` it can be run on
demand, and a manual run is never skipped:
`gh workflow run nightly-gpu.yaml --ref <branch> [-f matrix=core] [-f seed=N] [-f rows=PREFIX,...]`.

## Where it runs

1. **`build-wheel`** (a CPU runner) builds `libstreamer.so` and the wheel of the streamer from the commit under test,
   in the repository's development image, and uploads the wheel.
2. **`gpu-tests`**, two legs in parallel on runner group `nv-gpu-amd64-t4-2gpu` (label `linux-amd64-gpu-t4-latest-2`:
   two T4s, 12 cores, 48 GB). Each leg runs `setup_and_run.sh` in a clean CUDA container
   (`nvidia/cuda:12.4.1-base-ubuntu22.04`: Ubuntu 22.04, no Python packages, no toolchain, GPU passthrough, 2 GiB of
   `/dev/shm` for NCCL and the io_uring seccomp setting). The script installs Python, pip, the PyTorch and numpy of the
   leg, and the wheel, the way a user would:

   | Leg | PyTorch | Rows of the synthetic models |
   |---|---|---|
   | `newest-torch` | the newest on PyPI (`torch`) | `full` (94 rows, under each of the four filesystem strategies), plus the self-check |
   | `oldest-torch` | the oldest the package supports (`torch==2.0.0`, with `numpy<2`) | `core` (14 rows, under each strategy) |

   Both legs also run the smoke test of the small real model. The versions in between are not run. The newest leg is
   what covers the newest dtypes; the oldest covers the code paths for a torch without `dtype.itemsize`, float8 or
   the unsigned 16/32/64 bit types.

   pip installs the newest torch that the image's Python can use (Python 3.10 on Ubuntu 22.04), so when a torch release
   stops supporting that Python, the newest leg would quietly test an older one. `env_check.py newest-torch` asks PyPI
   for the newest release and compares: if they differ it fails the leg and says which release needs which Python, and
   that the base container of the leg has to change. **The tests still run, with the older torch**: the leg is red
   because it is not testing the newest torch, not because a test failed, and the fix is the environment of the leg
   (`GPU_TEST_IMAGE` in the workflow). This is visible in four places: the check's output at the start of the log, a
   `NOT TESTING THE NEWEST TORCH` line at the end of the log with the verdict of the tests, an error annotation on the
   run, and a section at the top of the job summary. If PyPI cannot be reached the check says it could not verify and
   does not fail.

## What `setup_and_run.sh` does

1. installs the system Python and pip, PyTorch and numpy at the leg's versions, then the wheel, and checks that the
   streamer is imported from the installed wheel and that two GPUs and NCCL are there;
2. in the newest leg, checks that the container does not limit the newest torch; runs the CPU unit tests (generator,
   rows, real model download, the environment check);
3. generates four synthetic models into `.gpu-fixtures/` (not in the repo; the runner's VM is discarded after
   the job) and verifies them. The generator reads the streamer's dtype table, so it runs after the install. The models
   hold every dtype that the streamer's own table accepts with the installed PyTorch and that the generator knows (the
   script prints which ones are left out);
4. downloads the small real model (see below) and describes it, timing both;
5. fails if the fixtures are not on a real block device that serves `O_DIRECT` (the streamer reads a file from a mount
   that cannot, such as tmpfs, with its synchronous reader, quietly);
6. runs `test_synthetic_models.py` under `torchrun` with two ranks, once per filesystem strategy (`sync_buffered`,
   `libaio_direct`, `io_uring_direct`, `io_uring_buffered`), each alone with no `sync_buffered` fallback: a reader the
   host cannot serve fails the run instead of passing on the synchronous reader;
7. runs the self-check (newest leg, when asked) and the smoke test;
8. prints which rows ran, and a table of where the time went.

## The small real model (`real_model.py`)

A smoke test with a real model, with the streamer's default settings (no filesystem strategy chosen, no memory limit):
`Qwen/Qwen2.5-0.5B` (Apache-2.0, not gated, one 988 MB `model.safetensors`), pinned by revision, size and SHA-256, so
an upstream change fails the run. It is downloaded from the Hugging Face hub on every run, and the download is timed
(`real_model.download` in the timing table), to see what it adds to a run. `real_model.py` builds a manifest from the
bytes of the file (the CRC32 of every tensor), in the format of the synthetic models, so the same checks compare the
streamed tensors. Three rows (`GPU_MATRIX=smoke`): one solo load to the GPU, one distributed over the two ranks, and
one to host memory and then `.to()`.

## The models (`gen_models.py`)

| Model | Stress | Shape |
|---|---|---|
| `mixed` | every dtype, ranks 0-5, sizes at the 4 KiB / 64 KiB / 8 MiB boundaries, odd names | 4 uneven shards, about 1 GiB, headers padded differently so tensor starts are aligned, unaligned and misaligned for their dtype |
| `many_tensors` | about 100,000 tiny tensors, a header of several MB | 1 file, about 100 MB |
| `giant_tensor` | a tensor of exactly 2 GiB and one of 2004 MiB | 2 shards, about 4 GiB |
| `many_files` | 5,000 files, some header-only, some holding a single 1-byte tensor | about 90 MB |

Each model is written to `<out>/<name>/` with a manifest `<out>/<name>.manifest.json`. The generator takes
`--models mixed,many_files` to build only some of them.

Every dtype, every rank from 0 to 5 and every edge size is forced into the models; the rest is drawn from the
seed. The generator knows 18 dtypes: `F64`, `F32`, `F16`, `BF16`, the float8 types `F8_E4M3`, `F8_E5M2` and `F8_E8M0`,
`F4`, `C64`, the signed integers `I64` to `I8`, the unsigned `U64` to `U8`, and `BOOL`.
`--dtypes` narrows them; the test run passes the ones the streamer accepts with the installed PyTorch (a torch
before 2.8 has no `F4`, before 2.7 or so no `F8_E8M0`, before 2.3 no unsigned 16/32/64 bit types, before 2.1 no float8).
`F4` packs two 4-bit values into a byte, so the header records its shape in values and the torch tensor
(`float4_e2m1fn_x2`) has the last dimension halved (`packed_shape` in `gen_models.py`); an F4 tensor therefore has a
last dimension that is even and no scalar form. The streamer loads such tensors since
[#234](https://github.com/dsx-ai-factory/model-streamer/issues/234) was fixed, and the checks compare the torch shape
with `packed_shape` of the manifest's. A copy of the F4 tensor to CUDA is exercised here for the first time.
The expected dtype of a streamed tensor is taken from the streamer's own dtype table, so this test does not check
the table itself; [#267](https://github.com/dsx-ai-factory/model-streamer/issues/267) compares it with safetensors
in the CPU tests. The models hold no empty tensors (zero elements) for now, because the streamer yields them on the CPU
when a CUDA device is requested ([#260](https://github.com/dsx-ai-factory/model-streamer/issues/260)). Once
that is fixed, set `EMPTY_TENSORS = True` in `gen_models.py` to bring them back.

The seed is printed at the start of every run (`GPU_SEED`, default the workflow run id), and a seed reproduces
the models exactly under the same numpy version (the manifests record it; to rebuild an old model, install that
numpy and run the generator with the same seed). The manifest of a model lists, per tensor, the name, dtype,
shape, file position and CRC32 of the bytes, computed while the data was generated. The gzipped manifests are
uploaded with the results (`manifests/`), so a failing tensor can be looked up without regenerating anything.

The generator checks its own output before anything is streamed (headers, offsets, CRC32 read back from disk,
a few tensors regenerated from the seed, and coverage of all dtypes/ranks/edge sizes), so a failure in the
streaming test is never a failure of the fixture. It needs only numpy and runs on its own:

```bash
python3 tests/gpu/gen_models.py generate --out /tmp/fixtures --seed 1 --scale 0.05 --files 300
python3 -m unittest tests/gpu/test_gen_models.py
```

## The test (`test_synthetic_models.py`)

The rows are defined in `matrix.py` (no GPU needed; `test_matrix.py` tests it). A row is one load: a model,
distributed or not, a partition policy, `owned`, a destination (`cuda`, or `cpu_to`: load to host memory, then
`.to()` each tensor) and a memory limit. Its id says all of it: `mixed.solo.owned.cuda.tight`,
`many_files.dist.files.owned.cuda.tight`. `GPU_MATRIX` picks the rows:

- `core`: 14 rows, a quick set (the oldest-torch leg runs it). One option at a time on the `mixed` model (borrowed
  tensors, `chunks`, host then `.to()`, the deep and default limits), and one solo and one distributed load of every
  other model.
- `full` (the newest-torch leg): every valid combination, 94 rows. A distributed row needs `cuda`; the `files` policy
  needs more than one file; the deep limit needs a model that can use it.
- `smoke`: three loads of the small real model, with the default memory limit.

The memory limit decides whether the host buffers, the ring the streamer reads into, are reused. With
N = min(18, limit // B) buffers of B = max(largest range, limit // 18) bytes:

- `tight`: about a sixteenth of the model, never below twice the largest range (a tensor or a file header) times
  the ranks sharing the limit: two buffers for `mixed`, five for `many_tensors`, one for the other two. The
  model passes through the ring many times, and every buffer is reused.
- `deep`: 18 largest tensors per rank, so all 18 buffers exist and the model still passes through them more than
  once. Only `mixed` and `many_tensors` can have this.
- `default`: the streamer's 40 GB. A model that small fits in the ring whole and no buffer is ever reused.

After a load the test checks, for every tensor: name (no missing, extra or duplicate), dtype, shape, device and
the CRC32 of its bytes copied back from the GPU. Rows with `owned=True` hold every tensor without cloning until
the stream ends and check them all then, which tests the ownership guarantee. Rows with `owned=False` check each
tensor when it is yielded. A row that asks for distributed streaming fails if the streamer silently fell back
to non-distributed.

After every row the test drops its own references, frees the GPU cache and checks that `torch.cuda.memory_allocated()`
is back to its value before the run (8 MiB of slack). Anything above that is reported as a GPU memory leak and
fails the row. The ranks agree on the outcome of a row over a separate CPU-only (gloo) group, so the test adds no
NCCL traffic or GPU work next to the streamer's. Host memory is not checked here; that belongs in a longevity test.

Missing GPUs or NCCL fail the job; nothing skips. A row that found wrong tensors does not stop the run, so one
run reports every failing row. A distributed row in which a rank raised does stop it, since the collectives may
be out of step.

When a rank fails inside a distributed row, the streamer's NCCL group never completes its next collective, and after
the streamer's timeout (`RUNAI_STREAMER_DIST_TIMEOUT`, 300 s here) the NCCL watchdog aborts the other rank
(`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`). The whole `torchrun` of that strategy ends, so the rows after the failing one
are not run for that strategy; the other strategies still run. A drill on 2026-10-20 showed this (a rank raising
mid-row ended with `Watchdog caught collective operation timeout` and exit code -6 on the other rank).
The run is bounded, but it cannot report the row from inside the process, so the results file lists a row as
started (`"ok": null`) before it runs, and `setup_and_run.sh` prints `== rows that ran`: for every results file how many
of the expected rows finished, and in which row the run ended. An incomplete strategy fails the job.

The results of each row are written as soon as it ends, in `rows-<strategy>-rank<N>.json`:
`{"strategy", "expected_rows", "rows": [...]}`.

## Self-check of the checks

A check that never fails could be silently broken, so the run can prove that the CRC32 check and the GPU memory check
fire. With `GPU_SELF_CHECK=true`, `setup_and_run.sh` runs two rows once more (`sync_buffered` only, about 2 minutes)
with a fault injected on purpose (`GPU_DRILL=KIND:ROW_ID,...` in `test_synthetic_models.py`): `corrupt` flips a bit of
one tensor after it arrived, and `leak` keeps a 32 MiB GPU tensor alive after the row. That pass succeeds only if
both are reported, and fails if one goes unnoticed. Its results are in `self-check/`.

The workflow turns it on, in the newest-torch leg, for every manual run, and for a scheduled run when the harness changed
since the last scheduled run that passed (`test_synthetic_models.py`, `matrix.py`, `phase_timer.py`, `real_model.py`,
`setup_and_run.sh` or the workflow itself), so nobody has to remember to run it.

The script raises the open files limit to the hard limit before the test, because the `many_files` model has 5,000 files and
the streamer keeps a file open until its submission ends
([#261](https://github.com/dsx-ai-factory/model-streamer/issues/261)).

## Timing

Every phase is timed (system and Python package installs, generation, the real model download, setup, load, verify)
and written to `test-results/gpu/timing-*.jsonl`; the summary table is in the job summary of each leg and in
`timing-summary.txt`. It is there to find slow parts of the run, not to benchmark the streamer. The impact of the real
model download is the `real_model.download` row.

## Not covered

The Python path only. The C++ direct-to-GPU path (pinned staging buffers, `register` mode, several devices on
one streamer) is reached through the C API and its gtests, not through Python.
