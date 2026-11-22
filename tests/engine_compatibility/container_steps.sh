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

# The version the wheel's filename encodes (runai_model_streamer-<version>-py3-...whl), normalized
# the way pip reports it ('+' survives, '_' in the local part does not occur in our versions).
wheel_version=$(basename "$WHEEL" | cut -d- -f2)

echo "=== step 1: resolver check (pip install --dry-run, with dependencies) ==="
if ! python3 -m pip install --dry-run "$WHEEL"; then
  echo "ENGINE-COMPATIBILITY FAIL [resolver]: the wheel's dependencies conflict with the $ENGINE image's packages"
  status=1
else
  echo "ENGINE-COMPATIBILITY OK [resolver]"
fi

echo "=== step 2: install the wheel over the engine's pinned streamer ==="
# With its dependencies, the way a user installs it: pip's only-if-needed upgrade strategy leaves
# the engine's satisfied pins (torch, numpy) alone and adds only what is missing - an image that
# does not ship the streamer has none of its dependencies either, and --no-deps would leave the
# install broken (no humanize). What pip changes beyond the streamer, the log of this step says.
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
