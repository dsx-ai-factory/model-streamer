#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The steps that run INSIDE the engine image. Started by tests/engine_compatibility/run.sh, which documents
# them; not meant to be run on a host. Needs ENGINE (vllm|sglang) and WHEEL (path to the streamer
# wheel) in the environment, and the repository mounted at /workspace.

set -o nounset -o pipefail

status=0

echo "=== environment ==="
python3 --version
python3 -m pip --version
echo "engine image streamer (before): $(python3 -m pip show runai-model-streamer 2>/dev/null | sed -n 's/^Version: //p' || true)"

# The version encoded in the wheel's filename (runai_model_streamer-<version>-...whl).
wheel_version=$(basename "$WHEEL" | cut -d- -f2)

echo "=== step 1: resolver check (pip install --dry-run, with dependencies) ==="
# Constrained to the image's versions of the wheel's direct dependencies: unconstrained, the dry
# run passes by planning an upgrade of an engine pin; constraining every installed package trips on
# the images' own intentional inconsistencies (e.g. vLLM overrides torch's NCCL pin).
constraints=/tmp/engine-compatibility-constraints.txt
if ! python3 /workspace/tests/engine_compatibility/make_constraints.py "$WHEEL" > "$constraints"; then
  echo "ENGINE-COMPATIBILITY FAIL [resolver]: could not derive the constraints from the wheel and the image"
  status=1
else
  echo "constraining to the image's versions of: $(tr '\n' ' ' < "$constraints")"
  if ! python3 -m pip install --dry-run --constraint "$constraints" "$WHEEL"; then
    echo "ENGINE-COMPATIBILITY FAIL [resolver]: the wheel's dependencies conflict with the $ENGINE image's packages"
    status=1
  else
    echo "ENGINE-COMPATIBILITY OK [resolver]"
  fi
fi

echo "=== step 2: install the wheel over the engine's pinned streamer ==="
# Uninstall first: pip skips a local wheel whose version is already installed.
python3 -m pip uninstall --yes runai-model-streamer 2>/dev/null || true
# With dependencies: an image that does not ship the streamer has none of its dependencies either,
# and --no-deps would leave the install broken.
if ! python3 -m pip install "$WHEEL"; then
  echo "ENGINE-COMPATIBILITY FAIL [install]: pip could not install the wheel"
  exit 1
fi
installed=$(python3 -m pip show runai-model-streamer | sed -n 's/^Version: //p')
if [ "$installed" != "$wheel_version" ]; then
  echo "ENGINE-COMPATIBILITY FAIL [install]: installed version '$installed' is not the wheel's '$wheel_version'"
  exit 1
fi
echo "ENGINE-COMPATIBILITY OK [install]: runai-model-streamer $installed"

echo "=== step 3: stream through the native library, then import $ENGINE ==="
if ! python3 /workspace/tests/engine_compatibility/check.py "$ENGINE"; then
  status=1
fi

exit "$status"
