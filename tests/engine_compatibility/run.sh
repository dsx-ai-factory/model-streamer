#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that the streamer wheel under test installs and serves inside an official inference-engine
# image (vLLM or SGLang). See tests/engine_compatibility/README.md for the steps.
#
# Usage:
#   tests/engine_compatibility/run.sh --engine vllm   --image vllm/vllm-openai:latest --wheel-dir DIR [--serve]
#   tests/engine_compatibility/run.sh --engine sglang --image lmsysorg/sglang:latest  --wheel-dir DIR [--serve]
#
# runs every step in one container. CI runs each step as a workflow step of its own instead, so a
# red run names the failed check in the job's step list:
#
#   run.sh --engine E --image I --wheel-dir DIR [--serve] --start   # pull, print the digest, start the container
#   run.sh --engine E --step resolver|install|import|serve          # one step inside it
#   run.sh --engine E --stop                                        # remove the container
#
# DIR holds exactly one runai_model_streamer-*.whl, anywhere under it (the gpu-test-wheel artifact
# in CI, py/runai_model_streamer/dist locally). --serve needs a GPU of compute capability 8.0 or
# newer: the engine serves the pinned small real model with load_format=runai_streamer and answers
# one completion (tests/engine_compatibility/serve_check.py).

set -o errexit -o nounset -o pipefail

usage() { sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

engine='' image='' wheel_dir='' serve=0 mode=run step=''
while [ $# -gt 0 ]; do
  case "$1" in
    --engine)    engine="${2:?--engine needs a value}"; shift 2 ;;
    --image)     image="${2:?--image needs a value}"; shift 2 ;;
    --wheel-dir) wheel_dir="${2:?--wheel-dir needs a value}"; shift 2 ;;
    --serve)     serve=1; shift ;;
    --start)     mode=start; shift ;;
    --step)      mode=step; step="${2:?--step needs a value}"; shift 2 ;;
    --stop)      mode=stop; shift ;;
    --help|-h)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$engine" in
  vllm|sglang) ;;
  *) echo "--engine must be vllm or sglang (got '${engine:-none}')" >&2; exit 2 ;;
esac
container="engine-compatibility-$engine"

if [ "$mode" = stop ]; then
  docker rm -f "$container" >/dev/null 2>&1 || true
  exit 0
fi

if [ "$mode" = step ]; then
  exec docker exec "$container" /workspace/tests/engine_compatibility/container_steps.sh "$step"
fi

# run and start need the image and the wheel.
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

# --entrypoint: both official images set a serving ENTRYPOINT. The container carries the
# environment the steps need, so that --step can docker exec into it. A serve run gets the GPUs
# and shared memory for torch.
docker_args=(
  --name "$container"
  --entrypoint ''
  --volume "$wheel_mount_dir:/wheel:ro"
  --volume "$repo_root:/workspace:ro"
  --env "ENGINE=$engine"
  --env "WHEEL=/wheel/$wheel_name"
  --env "SERVE=$serve"
  --env PIP_BREAK_SYSTEM_PACKAGES=1
  --env PIP_ROOT_USER_ACTION=ignore
  --env PIP_DISABLE_PIP_VERSION_CHECK=1
)
if [ "$serve" = 1 ]; then docker_args+=(--gpus all --shm-size 2g); fi

if [ "$mode" = start ]; then
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker run --detach "${docker_args[@]}" "$image" sleep infinity >/dev/null
  echo "started $container"
else
  docker run --rm "${docker_args[@]}" "$image" bash /workspace/tests/engine_compatibility/container_steps.sh all
fi
