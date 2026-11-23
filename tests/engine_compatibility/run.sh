#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that the streamer wheel under test installs and runs inside an official inference-engine
# image (vLLM or SGLang). No GPU needed. See tests/engine_compatibility/README.md for the steps.
#
# Usage:
#   tests/engine_compatibility/run.sh --engine vllm   --image vllm/vllm-openai:latest --wheel-dir DIR [--serve]
#   tests/engine_compatibility/run.sh --engine sglang --image lmsysorg/sglang:latest  --wheel-dir DIR [--serve]
#
# DIR holds exactly one runai_model_streamer-*.whl, anywhere under it (the gpu-test-wheel artifact
# in CI, py/runai_model_streamer/dist locally). --serve needs a GPU of compute capability 8.0 or
# newer: after the install checks, the engine serves the pinned small real model with
# load_format=runai_streamer and answers one completion (tests/engine_compatibility/serve_check.py).

set -o errexit -o nounset -o pipefail

usage() { sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

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
[ -n "$image" ] || { echo "--image is required" >&2; exit 2; }
[ -d "$wheel_dir" ] || { echo "--wheel-dir '$wheel_dir' is not a directory" >&2; exit 2; }

# Recursive, as tests/gpu/tests_entrypoint.sh does: the CI artifact nests the wheel. No mapfile:
# this runs on the host, which may be a Mac with bash 3.2.
wheels=()
while IFS= read -r found; do wheels+=("$found"); done < <(find "$wheel_dir" -name 'runai_model_streamer-*.whl')
if [ ${#wheels[@]} -ne 1 ]; then
  echo "expected exactly one runai_model_streamer-*.whl under $wheel_dir, found ${#wheels[@]}: ${wheels[*]:-none}" >&2
  exit 2
fi
wheel_name=$(basename "${wheels[0]}")
wheel_mount_dir=$(cd "$(dirname "${wheels[0]}")" && pwd)

repo_root=$(cd "$(dirname "$0")/../.." && pwd)

# The digest behind the floating tag, so that a red run names the exact engine build.
docker pull "$image"
digest=$(docker inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null || true)
echo "ENGINE-COMPATIBILITY IMAGE: ${digest:-$image}"

# A serve run gets the GPUs, shared memory for torch, and a fixed name, so that a cancelled CI job
# can remove a container that would otherwise go on holding the GPU.
serve_args=''
if [ "$serve" = 1 ]; then serve_args="--gpus all --shm-size 2g --name engine-compatibility-serve-$engine"; fi

# --entrypoint: both official images set a serving ENTRYPOINT.
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
  /workspace/tests/engine_compatibility/container_steps.sh
