# Real-GPU tests

End-to-end tests of the model streamer on real GPUs. CI otherwise runs on CPU and mock devices only, so this is where a real
CUDA copy, real NCCL, the real storage path and the services that the streamer talks to are exercised together.

The code is in three parts, and a test is only ever in the last:

| Part | What it is | Where |
|---|---|---|
| **Environment** | the containers that a run happens in: a clean CUDA container, and the services that a suite needs (object storage) | `compose.yaml`, `tests_entrypoint.sh` |
| **Harness** | what every test shares: sets the container up, runs the suites with time caps and a budget, writes and reads results, reports the verdict. Contains no test of the streamer | `harness/` |
| **Suites** | the tests: one directory per kind of test, each with its own README | `suites/<name>/` |

```
tests/gpu/
  README.md                 this file
  compose.yaml              the default environment: `tests` (a clean CUDA container) and the services of the suites
  tests_entrypoint.sh       the entry of that environment: sets the container up, runs the suites, reports
  run.sh                    runs the environment, with the services of the selected suites: on your machine, and in CI
  harness/
    common.sh               time caps, the budget, timed steps, what to say when the script ends early
    suites.sh               finds the suites, runs them (prepare, run, expect, collect), reports
    summarize.py            reads the results files: which cases ran, which failed, where a run ended
    clean_results.sh        removes what an earlier run left in the results directory
    env_check.py            checks that the container does not limit the newest torch
    gpu_check.py            checks that the container has the GPUs and NCCL, and prints the versions
    seaweedfs-s3.json       the keys of the object storage service
    gpu_harness/            the python library of the suites (see "What the harness gives a suite")
    test_*.py               CPU unit tests of the harness
  suites/
    streaming/              the streamer loads models to the GPUs, byte for byte (the default suite)
    object_storage/         the streamer reads object storage: an example of a suite that is not about a model
  test_run.py               CPU unit tests of run.sh
  test_suites.py            CPU unit tests of the runner, and of what every suite must be
```

| Suite | What it tests | Needs | CI |
|---|---|---|---|
| [`streaming`](suites/streaming/README.md) | the streamer loads synthetic models and a small real model to the GPUs, under every filesystem strategy, memory limit and option, and every tensor is compared with its manifest | 2 GPUs (1 runs what needs one) | nightly |
| [`object_storage`](suites/object_storage/README.md) | features of the streamer on object storage (stream, list, pull, errors) against a SeaweedFS service | the `seaweedfs` service, the S3 plugin of the streamer | manual runs (`suites=object_storage`) |

## Where it runs

The `Nightly GPU tests` workflow (`.github/workflows/nightly-gpu.yaml`) runs every night at 00:17 UTC, whether or not anything
changed. With `workflow_dispatch` it can be run on demand:
`gh workflow run nightly-gpu.yaml --ref <branch> [-f matrix=core] [-f seed=N] [-f rows=PREFIX,...] [-f gpus=1]
[-f suites=streaming,object_storage]` (`matrix` and `rows` are the streaming suite's; `suites` defaults to `streaming`, and
a run that asks for `object_storage` also builds the S3 plugin and starts the SeaweedFS service, in the newest-torch leg).
A `rows` filter that matches no row of the newest-torch leg fails it, as a typo would;
the oldest-torch leg has only the core rows and passes, saying that it has nothing to run (`GPU_ALLOW_EMPTY_ROWS`). `gpus=1`
(default 2, and always 2 on the schedule) runs on one GPU of the runner and leaves out what needs two, which the log and
the summary say.

The wheel is built in the repository's published development image (`devcontainer:latest`), which only changes after a
change of `.devcontainer/` merges and the image is published. A run that happens before the publish builds with the old
image, and the next night builds with the new one.

The README of the repository shows a badge with the result of the latest scheduled run.

1. **`build-wheel`** (a CPU runner) builds `libstreamer.so` and the wheel of the streamer from the commit under test,
   in the repository's development image, and uploads the wheel (and, for a run that asks for the `object_storage` suite,
   the wheel of the S3 plugin, built the same way with `COMPONENT=s3`, in the same artifact).
2. **`gpu-tests`**, two legs in parallel on runner group `nv-gpu-amd64-t4-2gpu` (label `linux-amd64-gpu-t4-latest-2`:
   two T4s, 12 cores, 48 GB). Each leg downloads the wheels, makes sure that the docker compose plugin is there (a pinned
   release is installed if the runner has none), and runs `tests/gpu/run.sh`, which runs `tests_entrypoint.sh` in a clean CUDA container
   (`nvidia/cuda:12.6.3-base-ubuntu24.04` with Python 3.12 for the newest torch, which needs Python 3.11 or later;
   `nvidia/cuda:12.4.1-base-ubuntu22.04` with Python 3.10 for the oldest, since torch 2.0.0 has no wheel for 3.12. No
   Python packages, no toolchain, GPU passthrough, 2 GiB of `/dev/shm` for NCCL and the io_uring seccomp setting). The
   script installs Python, pip, the PyTorch and numpy of the leg, and the wheel, the way a user would:

   | Leg | PyTorch | Streaming rows |
   |---|---|---|
   | `newest-torch` | the newest on PyPI (`torch`) | `full` (104 rows, under each of the four filesystem strategies), plus the self-check on a manual run or when the harness changed |
   | `oldest-torch` | the oldest the package supports (`torch==2.0.0`, with `numpy<2`) | `core` (14 rows, under each strategy) |

   Both legs also run the smoke test of the small real model. The versions in between are not run. The newest leg is
   what covers the newest dtypes; the oldest covers the code paths for a torch without `dtype.itemsize`, float8 or
   the unsigned 16/32/64 bit types.

   pip installs the newest torch that the image's Python can use (Python 3.12 on Ubuntu 24.04), so when a torch release
   stops supporting that Python, the newest leg would quietly test an older one. `harness/env_check.py newest-torch` asks
   PyPI for the newest release and compares: if they differ it fails the leg and says which release needs which Python, and
   that the base container of the leg has to change (`GPU_TEST_IMAGE` in the workflow). **The tests still run, with the
   older torch**, and their results are a separate matter: the environment never explains a failing test, so a leg with
   this flag is red for one reason if the tests passed (the environment) and for two if some failed (the failing tests,
   to be looked into regardless, and the environment). The flag is visible in four places: the check's output at the
   start of the log, a `NOT TESTING THE NEWEST TORCH` line at the end of the log that gives the verdict of the tests,
   an error annotation on the run, and a section at the top of the job summary. If PyPI cannot be reached the check
   says it could not verify and does not fail. If PyPI's answer has a shape the check does not know, it is blind and says
   so in a warning annotation (not a failure).

The NVKS GPU runners run each job in an ephemeral virtual machine (the runner groups' documentation), so nothing stays
behind for the next job, and the scripts do not clean up after themselves. The log is written by the workflow
(`tests/gpu/run.sh ... | tee`), not by a `tee` inside the container, which is killed with the script and can lose the last lines,
where the verdict is. The wheel is downloaded outside the checkout and mounted read-only at `/wheel`.

## Running it: `run.sh`

`run.sh` runs the environment of `compose.yaml` (the clean CUDA container, with its image, `/dev/shm`, seccomp setting and
mounts) and the services that the selected suites need. The workflow runs it too, so a run on your machine and a run in
CI are the same thing apart from the machine. It needs Docker with the compose plugin (2.20 or later: `compose.yaml` uses
`depends_on` with `required: false`; `run.sh` checks), the NVIDIA container toolkit and an NVIDIA GPU (the environment
refuses to run without one), and about 9 GB of disk for the models of the streaming suite, on a filesystem that serves
`O_DIRECT` (not a tmpfs).

```bash
# a wheel that you built (a file, or a glob in the file name that matches exactly one), on every GPU that the machine has, at most two
tests/gpu/run.sh --wheel ~/dist/runai_model_streamer-*.whl

# a released version, on one GPU, the streaming rows that mention mixed
tests/gpu/run.sh --streamer-spec 'runai-model-streamer==0.17.0' --gpus 1 --rows mixed

# the object storage suite (it starts the SeaweedFS service, and removes it at the end)
tests/gpu/run.sh --suites object_storage --streamer-spec 'runai-model-streamer==0.17.0' \
    --s3-spec 'runai-model-streamer-s3==0.17.0'

# the oldest torch, every streaming row, the self-checks too
tests/gpu/run.sh --wheel PATH --matrix full --self-check --torch-spec 'torch==2.0.0' --numpy-spec 'numpy<2' \
    --image nvidia/cuda:12.4.1-base-ubuntu22.04
```

From the root of the repository, `make` has short forms of the common ones (`make gpu-help` lists them):

```bash
make gpu-test WHEEL=~/dist/runai_model_streamer-0.17.0-py3-none-any.whl   # also WHEEL_DIR=DIR, STREAMER_SPEC=SPEC
make gpu-test-1gpu STREAMER_SPEC='runai-model-streamer==0.17.0'
make gpu-test-object-storage WHEEL='A.whl B.whl'     # the streamer and the S3 plugin (paths without spaces)
make gpu-dry-run WHEEL=PATH                         # what would run, without running it
make gpu-unit-tests                                 # the CPU unit tests of all this: no GPU, no container
```

`SUITES`, `GPUS`, `MATRIX`, `ROWS` and `EXTRA` (any other option of `run.sh`) are variables of the same targets.
`harness/unit_tests.sh` is the one list of those unit tests: `tests_entrypoint.sh` runs the same script.

- The streamer comes from one place: `--wheel` (repeat it for more than one wheel), `--wheel-dir DIR` (a directory of
  wheels, mounted as it is: what CI downloads the artifact into), or `--streamer-spec` (a requirement specifier that pip
  installs: a version, a range, or just the name). `--wheel` copies the files alone into a temporary directory. Either way
  the directory is mounted read-only at `/wheel`, where `tests_entrypoint.sh` looks for the wheels.
- `--suites a,b` picks the suites (default `streaming`). A suite that needs services has a compose profile of its own name
  (`object_storage` starts `seaweedfs`): a service with a profile starts only when the profile is asked for, and `run.sh`
  asks for the profile of every selected suite (a suite without services has none, which compose does not mind). The
  services go with the run: `run.sh` ends with `docker compose down --volumes`, also when it is interrupted (Ctrl-C).
- `--gpus N` is the GPUs to use: with a number given, the machine must have that many, or the run fails (CI always gives
  it). `--gpus 1` runs on one GPU (`GPU_WORLD=1`): two ranks cannot share a GPU, so a suite leaves out what needs two, and
  says how much (the streaming suite: the 60 distributed rows of the 104).
- `--compose-file PATH` runs another environment (see "Environments"); `--no-chown` leaves the files that the container
  wrote as they are. `--project NAME` names the compose project, to clean up after a run that was killed (CI does it with
  the project's labels). By hand, every compose command needs the variables of `compose.yaml`, and `down` needs the profile
  of the services that it is to remove:
  `GPU_WHEEL_HOST_DIR=/tmp GPU_REPO=$PWD docker compose -p NAME -f tests/gpu/compose.yaml --profile object_storage down --volumes`.
- `--dry-run` prints what would run. The results are in `test-results/gpu/` of the checkout; each run removes the results
  of the earlier one first (`harness/clean_results.sh`: the results files, the summaries and the directories of the suites,
  not `run.log`), so a run of one GPU after a run of two is not counted as two ranks. The container runs as root, so the
  script gives what it wrote back to you at the end, and Python does not write `__pycache__` into the checkout.

## What `tests_entrypoint.sh` does

1. installs the system Python and pip, PyTorch and numpy at the leg's versions, then the streamer (the wheel in
   `/wheel`, or `GPU_STREAMER_SPEC`), and checks that the streamer is imported from the installed package and that the
   GPUs (two, or one with `GPU_WORLD=1`) and NCCL are there;
2. in the newest leg, checks that the container does not limit the newest torch;
3. runs the CPU unit tests of the harness (the unit tests of a suite run with the suite);
4. runs the suites that `GPU_SUITES` selects, one after the other (`harness/suites.sh`): for each, its unit tests, its
   `suite_prepare`, its `suite_run`, and then collects what it expects, its artifacts and its timings;
5. prints which cases ran, with the names of the failed ones (`harness/summarize.py`, also kept in `cases-summary.txt`
   and shown at the top of the workflow's job summary), a table of where the time went, and the verdict. The exit status
   is 0 only if everything passed and every result that a suite expected is there.

## What the harness gives a suite

A suite is a directory under `suites/` with a `suite.sh`. The runner sources it, from the root of the checkout, and calls the
functions that it defines. While they run, `GPU_SUITE` is the name of the suite, `GPU_RESULTS` a directory of its own under
the results directory, `GPU_FIXTURES` a directory of its own for what it generates, and `WORLD` the GPUs of the run.

| In `suite.sh` | |
|---|---|
| `suite_description="..."` | one line: what it tests (required) |
| `suite_unit_tests=(tests/gpu/suites/NAME/test_x.py ...)` | CPU-only tests of the suite's own code, run first; a failure fails the run, and the suite still runs |
| `suite_prepare()` | once, before the tests: fixtures, services to wait for. A failure skips the suite's tests and fails the run. It runs where `set -e` does not apply: end every command with `|| return` |
| `suite_run()` | the tests (required), built from the steps below |
| `suite_expect()` | prints the result files that the suite must have written, as globs relative to its results directory (required: a suite that did not run, or a run that wrote nothing, is missed instead of being green) |
| `suite_collect()`, `suite_timing()` | keep artifacts for debugging, and name extra timing files |

The steps that `suite_run` is made of (`harness/common.sh`). Each prints START and END, records its time, is cut off by its
cap (generous, and cut to what is left of the budget, so the run always gets to say what ran), and fails the run if it fails
while the other steps still run:

| Step | |
|---|---|
| `run_unittest NAME MINUTES TEST...` | ordinary unittest modules in one process (`python3 -m gpu_harness.run_unittest`); every test is a case in the results. The way to run a test of a feature |
| `run_torchrun NAME MINUTES SCRIPT [ARG...]` | a script under `torchrun` with `WORLD` ranks, one GPU each, for a test whose ranks work together (the streaming suite's `Rows`) |
| `run_step NAME MINUTES COMMAND...` | any command; `VAR=value run_step ...` or `run_step ... env -u VAR ...` for its environment |
| `phase NAME COMMAND...` | a timed preparation, for `suite_prepare` |

The Python library (`harness/gpu_harness`, on `PYTHONPATH`; it imports nothing from a suite):

| Module | |
|---|---|
| `run_unittest` | runs unittest modules and writes their results; a skipped test is a failure (a GPU test that cannot run is not a pass) |
| `session` | `Session`: for tests whose ranks work together: checks the ranks and GPUs, starts the NCCL group and a CPU-only group on which the ranks agree on a case's outcome, writes the results after every case |
| `results`, `world` | the results files, and `gpu_world()` (`GPU_WORLD`) |
| `object_store` | `ObjectStore`: a client of the object storage of the environment |
| `safetensors_file` | writes a small safetensors file, for a test that needs a file and not a model |
| `tensors` | the bytes of a tensor and their CRC32 (a tensor is compared by its bytes, never by value) |
| `phase_timer` | timing of the phases |

### Results

Every suite writes the same files, so that one summary reads them all: `results-<run>-rank<N>.json` under the suite's results
directory, one per rank of a run (`gpu_harness/results.py`):

```json
{"suite": "streaming", "run": "sync_buffered", "world": 2, "expected_cases": 104,
 "skipped_cases": 0, "skipped_note": "",
 "cases": [{"case": "mixed.solo.owned.cuda.tight", "ok": true, "seconds": 1.2}, ...]}
```

A case is written with `"ok": null` from the moment it starts, so that a process that is killed in the middle of it (the NCCL
watchdog does that when a rank fails inside a distributed case) leaves a file that names the case. `harness/summarize.py`
prints, per file, how many of the expected cases finished and failed and where the run ended, and fails the run if a
file is incomplete, if a rank of the run has no file, or if an expected result is not there at all. A suite that left cases
out on purpose says so (`skipped_note`), and the line is printed at the end, so a run on one GPU is not taken for a full one.

## Adding a test

Decide what you are adding:

**A new case in an existing suite** (a row, a model, a limit; a test of another feature of object storage): add it in that suite,
following its README. Nothing else changes.

**A new kind of test: a new suite.** One that looks at something else, end to end, and not necessarily at the content of a
model: a feature of the streamer that needs a service, a failure mode, a configuration. Copy `suites/object_storage`:

1. `suites/<name>/test_<something>.py`: an ordinary `unittest` module (one process: no ranks, no `torchrun`). Put what the
   tests share (a client, a prefix of its own, cleanup) in `setUpClass` and `setUp`, and use the harness's helpers instead of
   writing them again (`ObjectStore`, `small_model`, `tensor_crc32`). Add the harness to the path with the three lines that
   the other tests start with. If the ranks of the test have to work together, use a `Session` and `run_torchrun` instead
   (see the streaming suite's `test_synthetic_models.py`).
2. `suites/<name>/suite.sh`: the description, `suite_run` (`run_unittest basic 15 tests/gpu/suites/<name>/test_<something>.py`),
   and `suite_expect` (`echo "results-basic-rank0.json"`). Add `suite_prepare` for what the tests need first.
3. `suites/<name>/README.md`: what it tests and what it needs, like the two that are there.
4. If it needs a service, put it in `compose.yaml` with `profiles: [<name>]`, the name of the suite, and a healthcheck, and
   list it under `tests.depends_on` with `condition: service_healthy` and `required: false` (see `seaweedfs`): `run.sh`
   starts the profile of every suite that is selected, and compose starts only `tests` and what it depends on, so a
   service that is not listed there never starts. `test_suites.py` checks it.
5. CPU unit tests for helper code that the suite has (`suite_unit_tests`), if it has any.
6. Run it: `tests/gpu/run.sh --suites <name> ...`; and `python3 -m unittest tests/gpu/test_suites.py`, which checks
   the contract of every suite: a `suite.sh` that is valid bash and describes the suite and defines `suite_run` and `suite_expect`,
   a README, a test, unit tests that exist, profiles that are named after suites, no suite that knows another, no harness
   file that knows a suite.
7. To run it in CI (the workflow's `suites` input), the job has to build and give it what it needs: see how the S3 plugin wheel
   is built and uploaded next to the streamer's in `nightly-gpu.yaml`.

What a suite must not do: reach into another suite, or into the harness's internals (use `gpu_harness`); skip when it cannot
run (a missing GPU or service is a failure: say how to get it); write results anywhere but `GPU_RESULTS`; take a time that
is not capped (use the steps).

**A new environment**, for tests that need another kind of container (a toolchain for C++ tests, other services, another
image): see below. A new environment still runs the same suites.

## Environments

An environment is a compose file with a service named `tests`, which runs an *entry*: a script that sets the container up and then
runs the suites. `compose.yaml` is the default one: `tests` is a clean CUDA container, and its entry, `tests_entrypoint.sh`,
installs Python, torch and the streamer; the services of a suite (`seaweedfs`) have a compose profile of the suite's name.

To run in another environment, copy `compose.yaml`, change what differs (the image, the services, the command), and
run it with `tests/gpu/run.sh --compose-file my-compose.yaml ...`. What the compose file has to keep: the service
called `tests`; the checkout mounted at `/workspace`, and the wheels at `/wheel` (`GPU_REPO`, `GPU_WHEEL_HOST_DIR`); the
`GPU_*` variables listed under `environment` (they are how `run.sh` hands over its options); and a `command` that is the
entry. A different setup of the container needs its own entry: write it like `tests_entrypoint.sh`, which is short on purpose:
set the variables (`WORLD` and `GPU_WORLD`, the same number: the first is what `common.sh` launches with, the second what the
library reads; `RESULTS`, `FIXTURES`, `GPU_SEED`, `WHEEL_DIR` for the suites that install from the wheels, `SETUP_TIMING`,
and `PYTHONPATH` with `tests/gpu/harness`), and the environment that every suite runs under (NCCL's error handling,
unbuffered output, the streamer's log on stderr: copy the exports of `tests_entrypoint.sh` before `run_suites`), remove what
an earlier run left in the results
directory (`bash tests/gpu/harness/clean_results.sh "$RESULTS" $(ls tests/gpu/suites)`: without it `summarize.py` counts the
files of the earlier run as this one's), source `harness/common.sh` and `harness/suites.sh`, set the container up, then

```bash
select_suites || exit 2
run_suites
report_suites
finished=true
exit "$status"
```

so that the suites, the caps, the budget, the results and the verdict are the same in every environment.

## Variables

| Variable | Meaning |
|---|---|
| `GPU_SUITES` | the suites to run, comma-separated (default `streaming`) |
| `GPU_WORLD` | `2` (default) or `1`: the GPUs of the run |
| `GPU_TORCH_SPEC`, `GPU_NUMPY_SPEC` | what to install (default the newest torch, any numpy) |
| `GPU_STREAMER_SPEC` | install the streamer from a requirement specifier instead of the wheel in `/wheel` |
| `GPU_S3_SPEC` | the S3 plugin to install when there is no wheel of it (the object_storage suite) |
| `GPU_DEVICE` | where the object_storage suite streams to: `cuda:0` (default, and a missing GPU is a failure), or `cpu` when the suite's test is run by hand outside the environment, on a machine without a GPU |
| `GPU_SEED` | the seed of what is generated (default the time) |
| `GPU_EXPECT_NEWEST_TORCH` | `true` checks that the container does not limit the newest torch (the newest-torch leg) |
| `GPU_BUDGET_MINUTES` | the budget of the whole run (285: the job's timeout less 15 minutes) |
| `GPU_RESULTS`, `GPU_FIXTURES` | the base directories (`test-results/gpu`, `.gpu-fixtures`) that the entry reads if they are set where it runs (compose.yaml does not forward them: they are for an environment of your own); a suite gets its own under them, and these two names are what it sees |
| `GPU_MATRIX`, `GPU_ROWS`, `GPU_SELF_CHECK`, `GPU_ALLOW_EMPTY_ROWS`, `GPU_REAL_SMOKE` | the streaming suite's, see its README |

## Time

Every step is timed (system and Python package installs, generation, the real model download, setup, load, verify, each
test) and written to `timing-*.jsonl` next to the results; the summary table is in the job summary of each leg and in
`timing-summary.txt`. It is there to find slow parts of the run, not to benchmark the streamer. Every step that can hang has
a cap, generous, several times what the step takes on a healthy runner. The whole run has a budget, `GPU_BUDGET_MINUTES`
(285), which is the job's `timeout-minutes` (300) less 15: a cap is cut to what is left of the budget less 5 minutes, so the
script always gets to say what ran, whatever the caps add up to, and neither needs adjusting when the other changes (raise
the budget with the job's timeout). A setup step that fails (the installs) ends the script at once, and it says so at the
end of the log, with an error annotation: `THE RUN ENDED EARLY`. A suite that fails to prepare, a failing step, or a failing
unit test fails the run at the end, and the other suites and steps still run: their results are worth having.

## Not covered

The Python path only. The C++ direct-to-GPU path (pinned staging buffers, `register` mode, several devices on
one streamer) is reached through the C API and its gtests, not through Python.
