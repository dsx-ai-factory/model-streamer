#!/usr/bin/env bash
set -euo pipefail

# Python 3.12 is exercised by ci-test-python. Also test the same wheel and exact
# dependencies on the other supported stable interpreters bundled by manylinux.
for tag in cp313-cp313 cp314-cp314; do
    env_dir=$(mktemp -d)
    "/opt/python/${tag}/bin/python3" -m venv "$env_dir"
    "$env_dir/bin/python3" -m pip install --require-hashes --only-binary=:all: -r .devcontainer/dependencies/requirements.lock
    "$env_dir/bin/python3" -m pip install --no-deps py/runai_model_streamer/dist/*.whl
    library=$("$env_dir/bin/python3" -c 'from runai_model_streamer.libstreamer import DEFAULT_STREAMER_LIBRARY; print(DEFAULT_STREAMER_LIBRARY)')
    PATH="$env_dir/bin:$PATH" STREAMER_LIBRARY="$library" make -C py test
    "$env_dir/bin/python3" -m pip check
done
