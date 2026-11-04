# The streaming suite

Streams synthetic safetensors models, and one small real model, to real GPUs with the model streamer and compares every
tensor with what the manifest recorded. CI otherwise runs on CPU and mock devices only, so this is where a real CUDA copy,
real NCCL and the real storage path are exercised together.

It is one suite of the real-GPU tests (`tests/gpu/README.md` says how suites run, what the harness gives them, and how to
add another). It is the one that runs by default (`GPU_SUITES=streaming`), and the only one that CI runs.

## What is in this directory

| File | What it is |
|---|---|
| `suite.sh` | what the runner does for this suite: prepare (generate the models, fetch the real model, check the storage), run (once per filesystem strategy, the self-check, the smoke test), and what it expects to have written |
| `gen_models.py` | the generator of the four synthetic models, and their manifests |
| `dtype_check.py` | the dtypes that the generator writes and the installed streamer accepts; fails on a dtype the generator does not know |
| `direct_io_probe.py` | probes that the fixtures directory serves `O_DIRECT` reads |
| `real_model.py` | the small real model: download, pin, manifest |
| `matrix.py` | the rows (one load each) and the memory limit of each; no torch, unit tested on the CPU |
| `test_synthetic_models.py` | the test that runs on the GPUs, under `torchrun` with the ranks: `Rows` and `SelfChecks` |
| `drills.py` | what the self-checks accept as the report of an injected fault |
| `test_gen_models.py`, `test_matrix.py`, `test_real_model.py`, `test_drills.py` | CPU-only unit tests of the code above (`suite_unit_tests` in `suite.sh`: the run starts with them) |

The harness (`tests/gpu/harness/gpu_harness`) gives this suite its ranks and results (`Session`), the timings, and the bytes
and CRC32 of a tensor (`tensors.py`). Nothing in the harness knows this suite.

## How it runs (`suite.sh`)

1. generates four synthetic models into `.gpu-fixtures/streaming/` (in the workspace, not in the repo; the runner groups'
   documentation says the NVKS GPU runners run each job in an ephemeral virtual machine, so nothing stays behind for the
   next job, and the script does not clean up after itself) and verifies them. The generator reads the streamer's dtype
   table, so it runs after the install. The models hold every dtype that the streamer's own table accepts with the
   installed PyTorch and that the generator knows (the script prints which ones are left out);
2. downloads the small real model (see below) and describes it, timing both;
3. fails if the fixtures are not on a real block device that serves `O_DIRECT` (the streamer reads a file from a mount
   that cannot, such as tmpfs, with its synchronous reader, quietly);
4. runs the `Rows` test case of `test_synthetic_models.py` (a `unittest` module) under `torchrun` with two ranks, once per
   filesystem strategy (`sync_buffered`, `libaio_direct`, `io_uring_direct`, `io_uring_buffered`), each alone with no
   `sync_buffered` fallback: a reader the host cannot serve fails the run instead of passing on the synchronous reader;
5. runs the self-check (newest leg, when asked) and the smoke test.

The variables of the suite:

| Variable | Meaning |
|---|---|
| `GPU_MATRIX` | `core` (default), `full` or `smoke`: which rows `Rows` runs |
| `GPU_ROWS` | comma-separated row id prefixes: run only the rows whose id starts with one of them. A prefix that matches no row fails the run (a typo), unless `GPU_ALLOW_EMPTY_ROWS=true`: a leg that has only some of the rows (the oldest-torch leg has the core rows) then has nothing to run and passes, saying so |
| `GPU_SELF_CHECK` | `true` also runs the self-check |
| `GPU_REAL_SMOKE` | `false` leaves out the download of the real model and the smoke test |
| `GPU_SEED` | the seed of the models (the harness sets it: default the time; the workflow passes the run id) |
| `GPU_FIXTURES` | where the models are written (the harness gives the suite a directory of its own) |

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
| `mixed` | every dtype, ranks 0-5, sizes at the 4 KiB / 64 KiB / 8 MiB boundaries, odd names, sizes log-uniform up to 64 MiB | 4 uneven shards, about 1 GiB, headers padded differently so tensor starts are aligned, unaligned and misaligned for their dtype |
| `many_tensors` | about 100,000 tensors in a mix of sizes: a fifth are 4-byte scalar scales, a few are tiny (1 to 128 elements), 70% are of 1 to 16 KB (norms, biases, blocks of scales), the rest up to 2 MiB; about 100 more of up to 32 MiB, drawn like those of `mixed`; a header of 11 MB, and an embedding of at least twice the header as its largest tensor | 1 file, 1.0 to 1.2 GB depending on the seed (median tensor 2.6 KB) |
| `giant_tensor` | a tensor of exactly 2 GiB and one of 2004 MiB, among about 50 tensors of the layers of a model (log-uniform up to 64 MiB) and small ones | 2 shards, about 4.5 GB |
| `many_files` | 500 files, some header-only, some holding a single 1-byte tensor, most holding 8 to 40 tensors of up to 2 MiB (about 2.7 MiB a file), in some seeds a few files with one to three tensors of up to 10 MiB | about 1.4 GB, 11,500 tensors |

A real model has a header of kilobytes (about 100 bytes a tensor) and tensors of megabytes to gigabytes, so the largest
tensor of every model is not smaller than its largest header (the streamer reads a header as one range; `matrix.py`
refuses a model that breaks this, and `test_gen_models.py` checks it).

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
in the CPU tests. The models hold empty tensors (zero elements): about 2% of the drawn shapes have a zero dimension, and
the mixed model has two mandatory ones. They were left out while the streamer yielded them on the CPU when a CUDA
device was requested ([#260](https://github.com/dsx-ai-factory/model-streamer/issues/260), fixed), and a run on GPU
is what confirms the fix. `EMPTY_TENSORS = False` in `gen_models.py` leaves them out again.

The seed is printed at the start of every run, and a seed reproduces the models exactly under the same numpy version (the
manifests record it; to rebuild an old model, install that numpy and run the generator with the same seed). The manifest
of a model lists, per tensor, the name, dtype, shape, file position and CRC32 of the bytes, computed while the data was
generated. The gzipped manifests are uploaded with the results (`streaming/manifests/`), so a failing tensor can be looked
up without regenerating anything.

The generator checks its own output before anything is streamed (headers, offsets, CRC32 read back from disk,
a few tensors regenerated from the seed, and coverage of all dtypes/ranks/edge sizes), so a failure in the
streaming test is never a failure of the fixture. It needs only numpy and runs on its own:

```bash
python3 tests/gpu/suites/streaming/gen_models.py generate --out /tmp/fixtures --seed 1 --scale 0.05 --files 300
python3 -m unittest tests/gpu/suites/streaming/test_gen_models.py
```

## The test (`test_synthetic_models.py`)

A `unittest` module, run as `torchrun --nproc_per_node=2 tests/gpu/suites/streaming/test_synthetic_models.py Rows` (or
`SelfChecks`, see below). `Rows` loops over the rows with a `subTest` each: a failing row is recorded with its id and
errors, and the next row runs, so one run reports every failing row, and unittest sets the exit status. The only row that
stops the loop is one after which a rank raised in a distributed row (see below).

The rows are defined in `matrix.py` (no GPU needed; `test_matrix.py` tests it). A row is one load: a model,
distributed or not, a partition policy, `owned`, a destination (`cuda`, or `cpu_to`: load to host memory, then
`.to()` each tensor) and a memory limit. Its id says all of it: `mixed.solo.owned.cuda.tight`,
`many_files.dist.files.owned.cuda.tight`. `GPU_MATRIX` picks the rows:

- `core`: 14 rows, a quick set (the oldest-torch leg runs it). One option at a time on the `mixed` model (borrowed
  tensors, `chunks`, host then `.to()`, the roomy and default limits), and one solo and one distributed load of every
  other model.
- `full` (the newest-torch leg): every valid combination, 104 rows. A distributed row needs `cuda`; the `files` policy
  needs more than one file; the roomy limit is for the models that it is below (not `giant_tensor`).
- `smoke`: three loads of the small real model, with the default memory limit.

On one GPU (`GPU_WORLD=1`) the distributed rows are left out: two ranks cannot share a GPU, and they stream between two
ranks over NCCL. What runs is 44 of the 104 rows of `full`, 8 of the 14 of `core` and 2 of the 3 of `smoke`: all four
models, every dtype, `owned` and borrowed tensors, both destinations, the three limits, the four filesystem strategies, the
checks and the self-checks. The log and the summary say how many rows were left out (`ONE GPU: N distributed rows were not
run`), so a run on one GPU is not taken for a run of every row. Selecting only distributed rows on one GPU fails.

The memory limit (`RUNAI_STREAMER_MEMORY_LIMIT`, a node total divided between the ranks, with no rank getting less than
the largest tensor) bounds what the streamer holds of the model at a time. The levels are defined by what a user can
see of it: the model's size and its largest tensor, not the arrangement of the streamer's buffers.

- `tight`: the largest tensor of the model for each rank, which is the least that the streamer gives a rank (a solo
  load with less is refused, and the streamer raises the share of a rank to that): the load gets through the whole
  model with room for one tensor at a time.
- `roomy`: a quarter of the model, but not less than twice the tight limit: room for several tensors at a time, and
  still far below the model.
- `default`: the streamer's 40 GB. A model that small fits whole and nothing is reused.

A limit below the size of the model cannot hold the model, so whatever the streamer does with it, it has to reuse
memory: that is what the limits are for. The rows do not claim anything about how many buffers there are, and no test
reads the streamer's ring sizing; that has its own unit tests in the streamer.

After a load the test checks, for every tensor: name (no missing, extra or duplicate), dtype, shape, device and
the CRC32 of its bytes copied back from the GPU. Rows with `owned=True` hold every tensor without cloning until
the stream ends and check them all then, which tests the ownership guarantee. (On a row that copies to the device,
the tensor held is the one the streamer yielded on the host, which is checked again at the end next to its copy.)
Rows with `owned=False` check each tensor when it is yielded. A row that asks for distributed streaming fails if the
streamer silently fell back to non-distributed.

After every row the test drops its own references, frees the GPU cache and checks that `torch.cuda.memory_allocated()`
is back to its value before the run (8 MiB of slack). Anything above that is reported as a GPU memory leak and
fails the row. That counts what the torch allocator holds: memory that the streamer's C++ side allocates itself
(`cudaMalloc`, pinned memory) is not seen. The ranks agree on the outcome of a row over a separate CPU-only (gloo)
group (the `Session` of the harness), so the test adds no NCCL traffic or GPU work next to the streamer's. Host memory
is not checked here; that belongs in a longevity test.

Missing GPUs or NCCL fail the job; nothing skips. A row that found wrong tensors does not stop the run, so one
run reports every failing row. A distributed row in which a rank raised does stop it, since the collectives may
be out of step. A limit that cannot be set for a model (a header larger than the largest tensor, a roomy limit that the
model fits in) is a failure of the row, found from the manifest before the load.

When a rank fails inside a distributed row, the streamer's NCCL group never completes its next collective, and after
the streamer's timeout (`RUNAI_STREAMER_DIST_TIMEOUT`, 300 s here) the NCCL watchdog aborts the other rank
(`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`). The whole `torchrun` of that strategy ends, so the rows after the failing one
are not run for that strategy; the other strategies still run. A drill on 2026-10-20 showed this (a rank raising
mid-row ended with `Watchdog caught collective operation timeout` and exit code -6 on the other rank).
The run is bounded, but it cannot report the row from inside the process, so the results file lists a row as
started (`"ok": null`) before it runs, and the summary of the run (`== cases that ran`) says, for every results file, how
many of the expected rows finished and in which row the run ended, and it checks that each of the ranks wrote its results
file (a rank that died before its first write leaves only the other's). An incomplete strategy fails the job.

The time caps for this suite are for a hang, and generous: 60 minutes per strategy (about 10 on a healthy runner, 18 on a
slow day), 20 for the self-check and for the smoke test, 20 for the generation and 35 for the real model (an attempt to
download it also has a deadline of 10 minutes, so that a connection that trickles data ends). A leg takes about 40 minutes
(newest) or 8 (oldest) when healthy. A download of the real model that fails fails the run at the end, and the tests of the
synthetic models still run.

The results of each row are written as soon as it ends, in `streaming/results-<strategy>-rank<N>.json`, in the format
that every suite writes (`tests/gpu/README.md`, "Results"): a row is a case, named by its id, with `ok`, `seconds`,
`errors` and the GPU memory held after it.

## Self-check of the checks

A check that never fails could be silently broken, so the run can prove that the CRC32 check and the GPU memory check
fire. `test_synthetic_models.py` is a `unittest` module with two test cases, `Rows` (every row, one subtest each, so that
a failing row does not stop the others) and `SelfChecks`; the exit status is unittest's, so there is no pass/fail logic
of our own to test. With `GPU_SELF_CHECK=true`, `suite.sh` runs `SelfChecks` once (`sync_buffered` only, about 10
seconds), which injects two faults and asserts, as ordinary tests, that each is reported as what it is and with no other
error: `corrupt` flips a bit of one tensor after it arrived (the CRC32 error must name that tensor), and `leak` keeps a
32 MiB GPU tensor alive after the row (the GPU memory error must be there, with at least those 32 MiB held). A fault that
goes unnoticed fails an assertion, and the run. Its results are in `streaming/self-check/`; `drills.py` has what the
assertions accept, and a case with an injected fault (`"fault"` in its results) is not counted as a failure in the summary.

The workflow turns it on, in the newest-torch leg, for every manual run, and for a scheduled run when the code that it
exercises changed since the last scheduled run that passed (this suite's test, `drills.py`, `gen_models.py`, `matrix.py`,
`real_model.py` and `suite.sh`, the harness's `gpu_harness`, `common.sh` and `suites.sh`, `tests_entrypoint.sh`, or the
workflow itself), so nobody has to remember to run it.

## The open files limit

The streamer keeps a file open until its submission ends
([#261](https://github.com/dsx-ai-factory/model-streamer/issues/261)), so a load holds descriptors in proportion to
its files and requests. The container's soft limit of 1024 is too low for `many_files` (500 files) and `many_tensors`
(one file of about 100,000 tensors).

`test_synthetic_models.py` raises the soft limit of its own process to 262,144, or to the hard limit if that is lower,
and prints it (`open files limit: 1024 -> 262144`). No other process is raised.

## Not covered

The Python path only. The C++ direct-to-GPU path (pinned staging buffers, `register` mode, several devices on
one streamer) is reached through the C API and its gtests, not through Python.
