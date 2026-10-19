# Real-GPU tests

Streams synthetic safetensors models to real GPUs with the model streamer and compares every tensor with what
the generator recorded. CI otherwise runs on CPU and mock devices only, so this is where a real CUDA copy,
real NCCL and the real storage path are exercised together.

The `Nightly GPU tests` workflow (`.github/workflows/nightly-gpu.yaml`) runs the `core` rows every night at
00:17 UTC and all the rows (`full`) every Sunday at 01:47 UTC, on runner group `nv-gpu-amd64-t4-2gpu` (label
`linux-amd64-gpu-t4-latest-2`: two T4s, 12 cores, 48 GB). With `workflow_dispatch` either can be run on demand:
`gh workflow run nightly-gpu.yaml --ref <branch> [-f matrix=full] [-f seed=N]`.

## What it does

`setup_and_run.sh`, inside `.devcontainer/gpu/devcontainer.json` (the prebuilt devcontainer image with GPU
passthrough, 2 GiB of `/dev/shm` for NCCL and the existing io_uring seccomp setting):

1. installs CUDA PyTorch (2.4.1, the newest build for the image's Python 3.8) and the Python requirements,
   builds the real `libstreamer.so` and proves io_uring works here;
2. generates four synthetic models into `.gpu-fixtures/` (not in the repo; the runner's VM is discarded after
   the job), in parallel with the build, and verifies them;
3. runs `test_synthetic_models.py` under `torchrun` with two ranks, once per filesystem strategy
   (`sync_buffered`, `libaio_direct`, `io_uring_direct`, `io_uring_buffered`);
4. prints a table of where the time went.

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
seed. The models hold no empty tensors (zero elements) for now, because the streamer yields them on the CPU
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

- `core` (nightly): 14 rows. One option at a time on the `mixed` model (borrowed tensors, `chunks`, host then
  `.to()`, the deep and default limits), and one solo and one distributed load of every other model.
- `full` (weekly): every valid combination, 94 rows. A distributed row needs `cuda`; the `files` policy needs more
  than one file; the deep limit needs a model that can use it.

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
be out of step. The results of each row are written as soon as it ends.

The script raises the open files limit to the hard limit before the test, because the `many_files` model has 5,000 files and
the streamer keeps a file open until its submission ends
([#261](https://github.com/dsx-ai-factory/model-streamer/issues/261)).

## Timing

Every phase is timed (setup, generation, write, verify, load, per-tensor check, teardown) and written to
`test-results/gpu/timing-*.jsonl`; the summary table is in the job summary and in `timing-summary.txt`. It is
there to find slow parts of the run, not to benchmark the streamer.

## Not covered

The Python path only. The C++ direct-to-GPU path (pinned staging buffers, `register` mode, several devices on
one streamer) is reached through the C API and its gtests, not through Python.
