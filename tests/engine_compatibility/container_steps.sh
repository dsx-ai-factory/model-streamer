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
python3 - "$WHEEL" > "$constraints" <<'PY'
import re, subprocess, sys, zipfile

def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()

names = set()
with zipfile.ZipFile(sys.argv[1]) as wheel:
    metadata = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
    for line in wheel.read(metadata).decode(errors="replace").splitlines():
        if line.lower().startswith("requires-dist:"):
            match = re.match(r"\s*([A-Za-z0-9._-]+)", line.split(":", 1)[1])
            if match:
                names.add(canonical(match.group(1)))
names.discard("runai-model-streamer")
frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
for line in frozen.splitlines():
    if "==" in line and canonical(line.split("==")[0].strip()) in names:
        print(line)
PY
echo "constraining to the image's versions of: $(tr '\n' ' ' < "$constraints")"
if ! python3 -m pip install --dry-run --constraint "$constraints" "$WHEEL"; then
  echo "ENGINE-COMPATIBILITY FAIL [resolver]: the wheel's dependencies conflict with the $ENGINE image's packages"
  status=1
else
  echo "ENGINE-COMPATIBILITY OK [resolver]"
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
