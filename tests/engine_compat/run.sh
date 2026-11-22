#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that the streamer wheel under test installs and runs inside an official inference-engine
# image (vLLM or SGLang). No GPU is needed: the wheel's native library is exercised by streaming a
# tiny generated safetensors file to CPU tensors (tests/engine_compat/check.py). Catches what broke
# before without CI noticing: a wheel whose libstreamer.so needs a newer glibc than the engine image
# ships, a dependency pin of the streamer that conflicts with the engine's, a packaging mistake.
#
# Usage:
#   tests/engine_compat/run.sh --engine vllm   --image vllm/vllm-openai:latest --wheel-dir DIR
#   tests/engine_compat/run.sh --engine sglang --image lmsysorg/sglang:latest  --wheel-dir DIR
#   tests/engine_compat/run.sh --engine vllm --serve --wheel-dir DIR \
#     --image public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest
#
# --serve (vLLM only, with vLLM's official CPU image): after the steps below, vLLM serves the
# pinned small real model with load_format=runai_streamer - its own loader streaming real weights
# through the wheel into CPU tensors - and answers one completion; the server log must show the
# streamer's marker (tests/engine_compat/serve_check.py). No GPU anywhere. The prebuilt CPU image
# needs avx512f: on a machine without it the serve part reports a SKIP instead of running.
#
# DIR holds exactly one runai_model_streamer-*.whl (the gpu-test-wheel artifact in CI, py/
# runai_model_streamer/dist locally). The steps, inside the engine container:
#   1. resolver  pip install --dry-run of the wheel WITH its dependencies, against the engine's
#                installed set: a conflict fails the run, but the later steps still run, so one run
#                reports every signal.
#   2. install   pip install --force-reinstall --no-deps, then assert that pip reports the version
#                encoded in the wheel's filename: proof the wheel under test replaced the engine's
#                pinned streamer.
#   3. check     tests/engine_compat/check.py: stream a generated safetensors file (dlopens
#                libstreamer.so), then import the engine itself.

set -o errexit -o nounset -o pipefail

usage() { sed -n '5,25p' "$0"; }

engine='' image='' wheel_dir='' serve=0
while [ $# -gt 0 ]; do
  case "$1" in
    --engine)    engine="${2:?--engine needs a value}"; shift 2 ;;
    --image)     image="${2:?--image needs a value}"; shift 2 ;;
    --wheel-dir) wheel_dir="${2:?--wheel-dir needs a value}"; shift 2 ;;
    --serve)     serve=1; shift ;;
    --help|-h)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$engine" in
  vllm|sglang) ;;
  *) echo "--engine must be vllm or sglang (got '${engine:-none}')" >&2; exit 2 ;;
esac
if [ "$serve" = 1 ] && [ "$engine" != vllm ]; then
  echo "--serve is vLLM-only (SGLang has no official CPU serving image)" >&2; exit 2
fi
[ -n "$image" ] || { echo "--image is required" >&2; exit 2; }
[ -d "$wheel_dir" ] || { echo "--wheel-dir '$wheel_dir' is not a directory" >&2; exit 2; }

# Exactly one streamer wheel, found the way tests/gpu/tests_entrypoint.sh finds it - recursively,
# because the gpu-test-wheel artifact keeps the build's directories (the wheel is nested under
# runai_model_streamer/dist/). Two wheels would make the tested version ambiguous.
# Not mapfile: this script runs on the host, which may be a Mac with bash 3.2 (the harness scripts
# run inside Linux containers and can assume bash 4).
wheels=()
while IFS= read -r found; do wheels+=("$found"); done < <(find "$wheel_dir" -name 'runai_model_streamer-*.whl')
if [ ${#wheels[@]} -ne 1 ]; then
  echo "expected exactly one runai_model_streamer-*.whl under $wheel_dir, found ${#wheels[@]}: ${wheels[*]:-none}" >&2
  exit 2
fi
wheel_name=$(basename "${wheels[0]}")
wheel_mount_dir=$(cd "$(dirname "${wheels[0]}")" && pwd)

repo_root=$(cd "$(dirname "$0")/../.." && pwd)

# The pull, and the digest behind the floating tag, so that a red run names the exact engine build.
docker pull "$image"
digest=$(docker inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null || true)
echo "ENGINE-COMPAT IMAGE: ${digest:-$image}"

# --entrypoint: both official images set a serving ENTRYPOINT (vllm serve / sglang launch_server),
# so the command would otherwise become its arguments. The container is the engine's own
# environment; the script below runs with its python3/pip, whatever they are.
# --shm-size for the serve run: torch wants more than docker's 64 MiB default.
serve_args=''
if [ "$serve" = 1 ]; then serve_args='--shm-size 2g'; fi

# shellcheck disable=SC2086  # serve_args is word-split on purpose
docker run --rm \
  --entrypoint bash \
  --volume "$wheel_mount_dir:/wheel:ro" \
  --volume "$repo_root:/workspace:ro" \
  --env ENGINE="$engine" \
  --env WHEEL="/wheel/$wheel_name" \
  --env SERVE="$serve" \
  --env PIP_BREAK_SYSTEM_PACKAGES=1 \
  --env PIP_ROOT_USER_ACTION=ignore \
  --env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  $serve_args \
  "$image" \
  /workspace/tests/engine_compat/container_steps.sh
