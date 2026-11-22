#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that the streamer wheel under test installs and runs inside an official inference-engine
# image (vLLM or SGLang). No GPU is needed: the wheel's native library is exercised by streaming a
# tiny generated safetensors file to CPU tensors (tests/engine_compatibility/check.py). Catches what broke
# before without CI noticing: a wheel whose libstreamer.so needs a newer glibc than the engine image
# ships, a dependency pin of the streamer that conflicts with the engine's, a packaging mistake.
#
# Usage:
#   tests/engine_compatibility/run.sh --engine vllm   --image vllm/vllm-openai:latest --wheel-dir DIR
#   tests/engine_compatibility/run.sh --engine sglang --image lmsysorg/sglang:latest  --wheel-dir DIR
#
# DIR holds exactly one runai_model_streamer-*.whl (the gpu-test-wheel artifact in CI, py/
# runai_model_streamer/dist locally). The steps, inside the engine container:
#   1. resolver  pip install --dry-run of the wheel WITH its dependencies, against the engine's
#                installed set: a conflict fails the run, but the later steps still run, so one run
#                reports every signal.
#   2. install   pip install of the wheel, the way a user installs it (missing dependencies are
#                added, satisfied pins stay), then assert that pip reports the version encoded in
#                the wheel's filename: proof the wheel under test replaced the engine's pinned
#                streamer.
#   3. check     tests/engine_compatibility/check.py: stream a generated safetensors file (dlopens
#                libstreamer.so), then import the engine itself.

set -o errexit -o nounset -o pipefail

usage() { sed -n '5,25p' "$0"; }

engine='' image='' wheel_dir=''
while [ $# -gt 0 ]; do
  case "$1" in
    --engine)    engine="${2:?--engine needs a value}"; shift 2 ;;
    --image)     image="${2:?--image needs a value}"; shift 2 ;;
    --wheel-dir) wheel_dir="${2:?--wheel-dir needs a value}"; shift 2 ;;
    --help|-h)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$engine" in
  vllm|sglang) ;;
  *) echo "--engine must be vllm or sglang (got '${engine:-none}')" >&2; exit 2 ;;
esac
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
echo "ENGINE-COMPATIBILITY IMAGE: ${digest:-$image}"

# --entrypoint: both official images set a serving ENTRYPOINT (vllm serve / sglang launch_server),
# so the command would otherwise become its arguments. The container is the engine's own
# environment; the script below runs with its python3/pip, whatever they are.
docker run --rm \
  --entrypoint bash \
  --volume "$wheel_mount_dir:/wheel:ro" \
  --volume "$repo_root:/workspace:ro" \
  --env ENGINE="$engine" \
  --env WHEEL="/wheel/$wheel_name" \
  --env PIP_BREAK_SYSTEM_PACKAGES=1 \
  --env PIP_ROOT_USER_ACTION=ignore \
  --env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$image" \
  /workspace/tests/engine_compatibility/container_steps.sh
