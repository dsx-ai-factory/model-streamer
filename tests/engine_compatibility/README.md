<!--
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine compatibility check

Checks that the streamer wheel under test installs and serves inside the official images of the
inference engines that consume the package: vLLM (`vllm/vllm-openai`) and SGLang
(`lmsysorg/sglang`). Both engines depend on `runai-model-streamer` and load models through it
(`load_format=runai_streamer`), so a change on our side can break them without any test in this
repository noticing. It has happened: a glibc upgrade in the development image produced a wheel
whose `libstreamer.so` needed glibc symbols newer than the vLLM image ships, and nothing failed
until vLLM users hit it.

Only the engines' public surfaces are exercised - pip and the serve command. Engine internals move
between releases and are not called. The steps, per engine image:

1. **resolver** - `pip install --dry-run` of the wheel *with* its dependencies, constrained to the
   image's versions of the wheel's direct dependencies ([make_constraints.py](make_constraints.py),
   so pip cannot pass by planning an upgrade of an engine pin, while the image's own transitive
   version choices stay its business): catches a dependency conflict a new pin of ours would
   introduce. A conflict fails the run but the later steps still run, so one run reports every
   signal.
2. **install** - uninstall the bundled streamer (pip skips a local wheel whose version is already
   installed), then `pip install` the wheel the way a user installs it (missing dependencies are
   added, satisfied pins stay), and assert pip reports the wheel's own version: the engine's pinned
   streamer is really replaced by the build under test.
3. **engine-import** - the engine still imports: catches the install having replaced a dependency
   with a version the engine cannot use.
4. **serve** (`--serve`, needs a GPU of compute capability 8.0+) - [serve_check.py](serve_check.py)
   snapshot-downloads `Qwen/Qwen2.5-0.5B` at the revision the GPU harness pins (serving needs the
   config and tokenizer files too, not only the weights), starts `vllm serve` /
   `sglang.launch_server` with `load_format=runai_streamer` - the engine's own loader streaming
   real weights through the installed wheel, which is what loads `libstreamer.so`, so a glibc or
   ABI mismatch fails here, with a hint - waits for `/health`, asks `/v1/completions` for one
   completion, and requires the streamer's `[RunAI Streamer]` marker in the server log: a server
   that quietly fell back to the engine's default loader fails, because then nothing was tested.
   SGLang's kernels need sm80 or newer, which is why this does not run on T4s.

**Scope: the core streamer, local files only.** The engine images install
`runai-model-streamer[s3,gcs,azure]`, and only the core package is replaced by the wheel under
test: after the install, the dev core sits next to the released object-storage plugins - a
combination the package's own extras pins forbid, and not one a user would have. That is harmless
for the local-file serving this check runs, but it means the object-storage path is not covered.
Building and installing all four wheels is a possible follow-up.

## Running it locally

Build the wheel (in the devcontainer: `make ci-build-cpp && make ci-build-python`), or download
the `gpu-test-wheel` artifact of a nightly run (`gh run download --name gpu-test-wheel --dir DIR`),
then:

```bash
tests/engine_compatibility/run.sh --engine vllm --image vllm/vllm-openai:latest \
  --wheel-dir py/runai_model_streamer/dist --serve
```

`--engine sglang --image lmsysorg/sglang:latest` for SGLang. Without `--serve`, only the install
checks run, which need no GPU. `--wheel-dir` must hold exactly one `runai_model_streamer-*.whl`,
anywhere under it (the search is recursive: the CI artifact nests the wheel under
`runai_model_streamer/dist/`). The images are 10-20 GB; the first run pulls them.

## Where it runs in CI

The `engine-compatibility` job of [nightly-gpu.yaml](../../.github/workflows/nightly-gpu.yaml), on
the `nv-gpu-amd64-l40g-1gpu` runners, one matrix leg per engine, against the wheel the run's
`build-wheel` job built from the commit under test. The engine images are `:latest` on purpose -
the job exists to catch breakage against what users install today - and each run logs the pulled
image's digest, so a red run names the exact engine build; a manual run can pin another tag through
the `vllm_image` / `sglang_image` inputs.
