# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The object storage suite: features of the streamer that need an object store, tested end to end against the SeaweedFS
# service of the environment. Sourced by tests/gpu/harness/suites.sh (see there for what a suite.sh may define). It is an
# example of a suite that is not about the content of a model: tests/gpu/suites/object_storage/README.md.

suite_description="the streamer reads from object storage (a SeaweedFS service): stream, list and pull, end to end"
# Its service, seaweedfs, is in compose.yaml under the profile object_storage (the name of the suite): run.sh starts it.

# The streamer reads object storage through a plugin of its own, a second package (runai-model-streamer-s3) that is built
# apart from the streamer (COMPONENT=s3 of the Makefile): from a wheel of it in GPU_WHEEL_DIR (/wheel) next to that of the
# streamer, else from GPU_S3_SPEC (a requirement specifier: runai-model-streamer-s3==X), else the newest on PyPI, which may
# not be the version of the streamer under test, and the log says so.
object_storage_install_s3_plugin() {
    local wheels
    mapfile -t wheels < <(find "$WHEEL_DIR" -name 'runai_model_streamer_s3-*.whl' 2>/dev/null)
    if [ "${#wheels[@]}" -eq 1 ]; then
        echo "installing the S3 plugin from ${wheels[0]}"
        timeout --signal=TERM --kill-after=30s "$(cap 5)s" python3 -m pip install --no-cache-dir "${wheels[0]}" || return
    elif [ "${#wheels[@]}" -gt 1 ]; then
        echo "ERROR: expected one runai_model_streamer_s3 wheel in $WHEEL_DIR, found ${#wheels[@]}: ${wheels[*]}" >&2
        return 1
    else
        echo "WARNING: no S3 plugin wheel in $WHEEL_DIR: installing ${GPU_S3_SPEC:-runai-model-streamer-s3 (the newest on PyPI)}, which may not be the version of the streamer under test"
        timeout --signal=TERM --kill-after=30s "$(cap 5)s" python3 -m pip install --no-cache-dir "${GPU_S3_SPEC:-runai-model-streamer-s3}" || return
    fi
}

# boto3 is the client that the tests put objects with; it is not a dependency of the streamer, so it is installed here.
suite_prepare() {
    export PIP_BREAK_SYSTEM_PACKAGES=1
    object_storage_install_s3_plugin || return
    timeout --signal=TERM --kill-after=30s "$(cap 5)s" python3 -m pip install --no-cache-dir boto3 || return
    # The service is a container of its own: wait until it answers, and say how to get one if the environment has none.
    timeout --signal=TERM --kill-after=10s "$(cap 5)s" python3 - <<'PY'
import sys

from gpu_harness.object_store import ObjectStore, ObjectStoreNotAvailable

try:
    store = ObjectStore.from_environment()
    store.wait_until_ready(seconds=90)
    print("object storage: %s, bucket %s" % (store.endpoint, store.bucket))
except ObjectStoreNotAvailable as error:
    print("ERROR: %s" % error, file=sys.stderr)
    sys.exit(1)
PY
}

suite_run() {
    run_unittest basic 15 tests/gpu/suites/object_storage/test_object_storage.py
}

suite_expect() {
    echo "results-basic-rank0.json"
}
