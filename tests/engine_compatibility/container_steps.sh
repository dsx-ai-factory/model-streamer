#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The steps that run INSIDE the engine image. Started by tests/engine_compatibility/run.sh, which
# documents them; not meant to be run on a host. Needs ENGINE (vllm|sglang) and WHEEL (path to the
# streamer wheel) in the environment, and the repository mounted at /workspace.
#
# Takes the step to run - environment, resolver, install, import, serve - so that CI can run each
# as a workflow step of its own and a red run names the failed check in the job's step list; or
# 'all' (the default), which runs every step in order and keeps going past a resolver failure, so
# one run reports every signal.

set -o nounset -o pipefail

environment() {
  python3 --version
  python3 -m pip --version
  echo "engine image streamer (before): $(python3 -m pip show runai-model-streamer 2>/dev/null | sed -n 's/^Version: //p' || true)"
}

resolver() {
  # Constrained to the image's versions of the wheel's direct dependencies: unconstrained, the dry
  # run passes by planning an upgrade of an engine pin; constraining every installed package trips
  # on the images' own intentional inconsistencies (e.g. vLLM overrides torch's NCCL pin).
  local constraints=/tmp/engine-compatibility-constraints.txt
  if ! python3 /workspace/tests/engine_compatibility/make_constraints.py "$WHEEL" > "$constraints"; then
    echo "ENGINE-COMPATIBILITY FAIL [resolver]: could not derive the constraints from the wheel and the image"
    return 1
  fi
  echo "constraining to the image's versions of: $(tr '\n' ' ' < "$constraints")"
  if ! python3 -m pip install --dry-run --constraint "$constraints" "$WHEEL"; then
    echo "ENGINE-COMPATIBILITY FAIL [resolver]: the wheel's dependencies conflict with the $ENGINE image's packages"
    return 1
  fi
  echo "ENGINE-COMPATIBILITY OK [resolver]"
}

install_wheel() {
  # The version encoded in the wheel's filename (runai_model_streamer-<version>-...whl).
  local wheel_version installed
  wheel_version=$(basename "$WHEEL" | cut -d- -f2)
  # Uninstall first: pip skips a local wheel whose version is already installed.
  python3 -m pip uninstall --yes runai-model-streamer 2>/dev/null || true
  # With dependencies: an image that does not ship the streamer has none of its dependencies
  # either, and --no-deps would leave the install broken.
  if ! python3 -m pip install "$WHEEL"; then
    echo "ENGINE-COMPATIBILITY FAIL [install]: pip could not install the wheel"
    return 1
  fi
  installed=$(python3 -m pip show runai-model-streamer | sed -n 's/^Version: //p')
  if [ "$installed" != "$wheel_version" ]; then
    echo "ENGINE-COMPATIBILITY FAIL [install]: installed version '$installed' is not the wheel's '$wheel_version'"
    return 1
  fi
  echo "ENGINE-COMPATIBILITY OK [install]: runai-model-streamer $installed"
}

import_engine() {
  # Catches the install having replaced a dependency with a version the engine cannot use.
  if ! python3 -c "import $ENGINE as m; print('ENGINE-COMPATIBILITY OK [engine-import]:', '$ENGINE', getattr(m, '__version__', 'unknown'))"; then
    echo "ENGINE-COMPATIBILITY FAIL [engine-import]: $ENGINE no longer imports after the install"
    return 1
  fi
}

serve() {
  # The engine's own serve command loads real weights through the installed streamer - what dlopens
  # libstreamer.so, so a glibc/ABI mismatch of the wheel fails here.
  python3 /workspace/tests/engine_compatibility/serve_check.py "$ENGINE"
}

case "${1:-all}" in
  environment) environment ;;
  resolver)    resolver ;;
  install)     install_wheel ;;
  import)      import_engine ;;
  serve)       serve ;;
  all)
    status=0
    environment
    resolver || status=1
    # No point going on over a broken install; a server started over one would blame the wrong step.
    install_wheel || exit 1
    import_engine || status=1
    if [ "${SERVE:-0}" = 1 ] && [ "$status" = 0 ]; then
      serve || status=1
    fi
    exit "$status"
    ;;
  *) echo "unknown step '${1}': environment, resolver, install, import, serve, or all" >&2; exit 2 ;;
esac
