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

1. **resolver** - `pip install --dry-run` of the wheel *with* its dependencies against the engine's
   pinned packages: catches a dependency conflict a new pin of ours would introduce.
2. **install** - `pip install --force-reinstall --no-deps`, then assert pip reports the wheel's own
   version: the engine's pinned streamer is really replaced by the build under test.
3. **check** - [check.py](check.py): stream a generated file through `SafetensorsStreamer` into CPU
   tensors and compare, then import the engine itself (catches the install having broken the
   engine's environment).

A resolver conflict fails the run but the later steps still run, so one run reports every signal.

## The serve check (`--serve`, vLLM only, no GPU)

With `--serve` and vLLM's official CPU image (`public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo`),
after the steps above pass, [serve_check.py](serve_check.py) runs the loader end to end: it
snapshot-downloads `Qwen/Qwen2.5-0.5B` at the revision the GPU harness pins (the config and
tokenizer files too - serving needs more than the weights), starts `vllm serve` with
`load_format=runai_streamer` - vLLM's own loader streaming the real weights through the installed
wheel into CPU tensors - waits for `/health`, asks `/v1/completions` for one completion, and
requires the streamer's `[RunAI Streamer]` marker in the server log: a server that quietly fell
back to vLLM's default loader fails, because then nothing was tested.

The prebuilt CPU image needs `avx512f`; on a machine without it the serve step prints a SKIP and
the run passes on the earlier steps alone (CI runner VMs are a mixed fleet). SGLang has no official
CPU serving image, so `--serve` is vLLM-only.

## Running it locally

Build the wheel (in the devcontainer: `make ci-build-cpp && make ci-build-python`), or download
the `gpu-test-wheel` artifact of a nightly run (`gh run download --name gpu-test-wheel --dir DIR`),
then:

```bash
tests/engine_compatibility/run.sh --engine vllm --image vllm/vllm-openai:latest \
  --wheel-dir py/runai_model_streamer/dist
```

`--engine sglang --image lmsysorg/sglang:latest` for SGLang. `--wheel-dir` must hold exactly one
`runai_model_streamer-*.whl`, anywhere under it (the search is recursive: the CI artifact nests the
wheel under `runai_model_streamer/dist/`). The images are 10-20 GB; the first run pulls them.

## Where it runs in CI

The `engine-compatibility` job of [nightly-gpu.yaml](../../.github/workflows/nightly-gpu.yaml), on
`ubuntu-latest`, against the wheel the run's `build-wheel` job built from the commit under test:
one leg per engine image (`vllm`, `sglang`), and the `vllm-cpu-serve` leg, which adds `--serve` on
the vLLM CPU image. The engine images are `:latest` on purpose - the job exists to catch breakage
against what users install today - and each run logs the pulled image's digest, so a red run names
the exact engine build; a manual run can pin another tag through the `vllm_image` / `sglang_image`
/ `vllm_cpu_image` inputs.
