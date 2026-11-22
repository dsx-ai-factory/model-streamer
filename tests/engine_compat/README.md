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

## Running it locally

Build the wheel (in the devcontainer: `make ci-build-cpp && make ci-build-python`), or download
the `gpu-test-wheel` artifact of a nightly run (`gh run download --name gpu-test-wheel --dir DIR`),
then:

```bash
tests/engine_compat/run.sh --engine vllm --image vllm/vllm-openai:latest \
  --wheel-dir py/runai_model_streamer/dist
```

`--engine sglang --image lmsysorg/sglang:latest` for SGLang. `--wheel-dir` must hold exactly one
`runai_model_streamer-*.whl`. The images are 10-20 GB; the first run pulls them.

## Where it runs in CI

The `engine-compat` job of [nightly-gpu.yaml](../../.github/workflows/nightly-gpu.yaml), on
`ubuntu-latest`, one matrix leg per engine, against the wheel the run's `build-wheel` job built
from the commit under test. The engine images are `:latest` on purpose - the job exists to catch
breakage against what users install today - and each run logs the pulled image's digest, so a red
run names the exact engine build; a manual run can pin another tag through the `vllm_image` /
`sglang_image` inputs.
