#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that the streamer wheel under test installs and runs inside an official inference-engine
# image (vLLM or SGLang). No GPU needed. See tests/engine_compatibility/README.md for the steps.
#
# Usage:
#   tests/engine_compatibility/run.sh --engine vllm   --image vllm/vllm-openai:latest --wheel-dir DIR
#   tests/engine_compatibility/run.sh --engine sglang --image lmsysorg/sglang:latest  --wheel-dir DIR
#
# DIR holds exactly one runai_model_streamer-*.whl, anywhere under it (the gpu-test-wheel artifact
# in CI, py/runai_model_streamer/dist locally).

set -o errexit -o nounset -o pipefail

usage() { sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

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

# --entrypoint: both official images set a serving ENTRYPOINT.
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
