#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The steps that run INSIDE the engine image. Started by tests/engine_compat/run.sh, which documents
# them; not meant to be run on a host. Needs ENGINE (vllm|sglang) and WHEEL (path to the streamer
# wheel) in the environment, and the repository mounted at /workspace.

set -o nounset -o pipefail

status=0

echo "=== environment ==="
python3 --version
python3 -m pip --version
echo "engine image streamer (before): $(python3 -m pip show runai-model-streamer 2>/dev/null | sed -n 's/^Version: //p' || true)"

# The version the wheel's filename encodes (runai_model_streamer-<version>-py3-...whl), normalized
# the way pip reports it ('+' survives, '_' in the local part does not occur in our versions).
wheel_version=$(basename "$WHEEL" | cut -d- -f2)

echo "=== step 1: resolver check (pip install --dry-run, with dependencies) ==="
if ! python3 -m pip install --dry-run "$WHEEL"; then
  echo "ENGINE-COMPAT FAIL [resolver]: the wheel's dependencies conflict with the $ENGINE image's packages"
  status=1
else
  echo "ENGINE-COMPAT OK [resolver]"
fi

echo "=== step 2: install the wheel over the engine's pinned streamer ==="
# --no-deps: swap only the streamer, leaving the engine's pinned stack as its users run it. The
# resolver signal came from step 1.
if ! python3 -m pip install --force-reinstall --no-deps "$WHEEL"; then
  echo "ENGINE-COMPAT FAIL [install]: pip could not install the wheel"
  exit 1
fi
installed=$(python3 -m pip show runai-model-streamer | sed -n 's/^Version: //p')
if [ "$installed" != "$wheel_version" ]; then
  echo "ENGINE-COMPAT FAIL [install]: installed version '$installed' is not the wheel's '$wheel_version'"
  exit 1
fi
echo "ENGINE-COMPAT OK [install]: runai-model-streamer $installed"

echo "=== step 3: stream through the native library, then import $ENGINE ==="
if ! python3 /workspace/tests/engine_compat/check.py "$ENGINE"; then
  status=1
fi

exit "$status"
