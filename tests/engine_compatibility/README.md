<!--
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Engine compatibility check

Checks that the streamer wheel under test installs and runs inside the official images of the
inference engines that consume the package: vLLM (`vllm/vllm-openai`) and SGLang
(`lmsysorg/sglang`). Both engines depend on `runai-model-streamer` and load models through it
(`load_format=runai_streamer`), so a change on our side can break them without any test in this
repository noticing. It has happened: a glibc upgrade in the development image produced a wheel
whose `libstreamer.so` needed glibc symbols newer than the vLLM image ships, and nothing failed
until vLLM users hit it.

No GPU is needed. The check streams a tiny safetensors file it generates itself, which is what
loads the native library (importing the package does not) - a glibc or ABI mismatch fails right
there. The steps, per engine image (details in [run.sh](run.sh)):

1. **resolver** - `pip install --dry-run` of the wheel *with* its dependencies, constrained to the
   image's versions of the wheel's direct dependencies (so pip cannot pass by planning an upgrade
   of an engine pin, while the image's own transitive version choices stay its business): catches a
   dependency conflict a new pin of ours would introduce.
2. **install** - uninstall the bundled streamer (pip skips a local wheel whose version is already
   installed), then `pip install` the wheel the way a user installs it (missing dependencies are
   added, satisfied pins stay), and assert pip reports the wheel's own version: the engine's pinned
   streamer is really replaced by the build under test.
3. **check** - [check.py](check.py): stream a generated file through `SafetensorsStreamer` into CPU
   tensors and compare, then import the engine itself (catches the install having broken the
   engine's environment).

A resolver conflict fails the run but the later steps still run, so one run reports every signal.

## The serve check (`--serve`, needs a GPU of compute capability 8.0+)

With `--serve`, after the steps above pass, [serve_check.py](serve_check.py) runs the engine end to
end through its public command - the only stable surface, unlike engine internals, which move
between releases: it snapshot-downloads `Qwen/Qwen2.5-0.5B` at the revision the GPU harness pins
(serving needs the config and tokenizer files too, not only the weights), starts `vllm serve` /
`sglang.launch_server` with `load_format=runai_streamer`, waits for `/health`, asks
`/v1/completions` for one completion, and requires the streamer's `[RunAI Streamer]` marker in the
server log: a server that quietly fell back to the engine's default loader fails, because then
nothing was tested. SGLang's kernels need sm80 or newer, which is why this does not run on T4s.

**Scope: the core streamer, local files only.** The engine images install
`runai-model-streamer[s3,gcs,azure]`, and only the core package is replaced by the wheel under
test: after the install, the dev core sits next to the released object-storage plugins - a
combination the package's own extras pins forbid, and not one a user would have. That is harmless
for the local-file stream this check runs, but it means the object-storage path is not covered.
Building and installing all four wheels is a possible follow-up.

## Running it locally

Build the wheel (in the devcontainer: `make ci-build-cpp && make ci-build-python`), or download
the `gpu-test-wheel` artifact of a nightly run (`gh run download --name gpu-test-wheel --dir DIR`),
then:

```bash
tests/engine_compatibility/run.sh --engine vllm --image vllm/vllm-openai:latest \
  --wheel-dir py/runai_model_streamer/dist
```

`--engine sglang --image lmsysorg/sglang:latest` for SGLang, and `--serve` for the serve check on
a machine with an sm80+ GPU. `--wheel-dir` must hold exactly one `runai_model_streamer-*.whl`,
anywhere under it (the search is recursive: the CI artifact nests the wheel under
`runai_model_streamer/dist/`). The images are 10-20 GB; the first run pulls them.

## Where it runs in CI

Two jobs of [nightly-gpu.yaml](../../.github/workflows/nightly-gpu.yaml), each one matrix leg per
engine, against the wheel the run's `build-wheel` job built from the commit under test:
`engine-compatibility` (the install checks, on `ubuntu-latest`) and `engine-serve` (the same plus
`--serve`, on the `nv-gpu-amd64-l40g-1gpu` runners). The engine images are `:latest` on purpose -
the jobs exist to catch breakage against what users install today - and each run logs the pulled
image's digest, so a red run names the exact engine build; a manual run can pin another tag through
the `vllm_image` / `sglang_image` inputs.
