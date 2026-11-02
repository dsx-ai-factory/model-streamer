# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The streaming suite: the streamer loads models to the GPUs, and every tensor is compared with what the manifest recorded.
# Sourced by tests/gpu/harness/suites.sh (see there for what a suite.sh may define). What it tests, and how, is in
# tests/gpu/suites/streaming/README.md.

suite_description="the streamer loads synthetic models and a small real model to the GPUs, byte for byte"
suite_services=()
suite_unit_tests=(
    tests/gpu/suites/streaming/test_gen_models.py
    tests/gpu/suites/streaming/test_matrix.py
    tests/gpu/suites/streaming/test_drills.py
    tests/gpu/suites/streaming/test_real_model.py
)

streaming_dir=tests/gpu/suites/streaming
# Each reader alone, with no sync_buffered fallback: a list the host cannot serve is an error in the streamer, so a run
# under a reader that is not available fails instead of passing on the synchronous reader. (The streamer lists in the
# Makefile's test_strategies keep sync_buffered last for hosts of any kind.)
streaming_strategies=(sync_buffered libaio_direct io_uring_direct io_uring_buffered)
smoke_expected=false

# The models hold every dtype that the streamer accepts with the installed PyTorch and that the generator knows. The
# streamer's own table decides, so the test follows it when it gains a dtype, and a dtype that this PyTorch lacks is
# listed here as not covered rather than failing the load of every file that holds it. The generator needs the
# installed streamer for that table, so it runs after the install. Generation takes under a minute.
streaming_generate_models() {
    local dtypes
    dtypes=$(python3 - <<'PY'
import sys
sys.path.insert(0, "tests/gpu/suites/streaming")
import gen_models
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype

streamer = set(safetensors_to_torch_dtype)
covered = sorted(streamer & set(gen_models.DTYPES))
print("streamer dtypes: %s" % " ".join(sorted(streamer)), file=sys.stderr)
print("not generated:   %s" % (" ".join(sorted(set(gen_models.DTYPES) - streamer)) or "-"), file=sys.stderr)
unknown = sorted(streamer - set(gen_models.DTYPES) - gen_models.KNOWN_GAPS)
print("not known to the generator: %s" % (" ".join(unknown) or "-"), file=sys.stderr)
if unknown:
    print("ERROR: the streamer accepts %s, which the generator cannot write: add it to DTYPES in "
          "tests/gpu/suites/streaming/gen_models.py" % " ".join(unknown), file=sys.stderr)
    sys.exit(1)
print(",".join(covered))
PY
    ) || return
    rm -rf "$GPU_FIXTURES"
    mkdir -p "$GPU_FIXTURES"
    timeout --signal=TERM --kill-after=30s "$(cap 20)s" \
        python3 "$streaming_dir/gen_models.py" generate --out "$GPU_FIXTURES" --seed "${GPU_SEED:?}" --dtypes "$dtypes" \
        2>&1 | tee "$GPU_RESULTS/generate.log"
    return "${PIPESTATUS[0]}"
}

# The direct I/O strategies are only tested if the fixtures are on a real block device that serves O_DIRECT. The
# streamer reads a file from a mount that cannot (tmpfs, for one) with its synchronous reader, quietly, so on such
# a storage a direct strategy would pass without ever being used. Fail instead.
streaming_check_fixture_storage() {
    local source fstype
    echo "fixtures in $GPU_FIXTURES: $(stat -f -c %T "$GPU_FIXTURES")"
    df -h "$GPU_FIXTURES" | tail -1
    source=$(df --output=source "$GPU_FIXTURES" | tail -1) || return
    fstype=$(stat -f -c %T "$GPU_FIXTURES") || return
    echo "fixtures storage: $source ($fstype)"
    case "$source" in
        /dev/*) ;;
        *) echo "ERROR: the fixtures are on $source ($fstype), not on a block device; the direct I/O strategies would not be tested" >&2
           return 1 ;;
    esac
    timeout --signal=TERM --kill-after=10s "$(cap 5)s" python3 - "$GPU_FIXTURES" <<'PY' || { echo "ERROR: the fixtures storage does not serve O_DIRECT reads" >&2; return 1; }
import mmap
import os
import sys

path = os.path.join(sys.argv[1], ".direct-io-probe")
with open(path, "wb") as probe:
    probe.write(bytes(4096))
try:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        if os.preadv(fd, [mmap.mmap(-1, 4096)], 0) != 4096:
            raise OSError("short O_DIRECT read")
    finally:
        os.close(fd)
finally:
    os.unlink(path)
PY
}

suite_prepare() {
    echo "== generate synthetic models"
    phase generate_models streaming_generate_models || return

    # The small real model, downloaded on every run: real_model.py times the download and the manifest separately
    # (timing-real.jsonl), to show what the download adds to a run. It only feeds the smoke test, so a download that
    # fails (the hub is slow or down) fails the run at the end, and does not stop the rows of the synthetic models.
    smoke_expected=false
    if [ "${GPU_REAL_SMOKE:-true}" = true ]; then
        echo "== download the small real model"
        if phase fetch_real_model timeout --signal=TERM --kill-after=30s "$(cap 35)s" \
            python3 "$streaming_dir/real_model.py" fetch --out "$GPU_FIXTURES"; then
            smoke_expected=true
        else
            status=1
            echo "ERROR: the real model could not be fetched: the smoke test will not run (the run fails at the end)"
        fi
    fi
    streaming_check_fixture_storage
}

suite_run() {
    export GPU_MATRIX="${GPU_MATRIX:-core}"
    local strategy

    # The time caps are for a hang: a strategy takes about 10 minutes when the runner is healthy (18 on a slow day), and
    # its cap is 60. A cap is cut to what is left of the run's budget (see cap, in common.sh), so the caps never need to
    # add up to less than the job's timeout for the script to get to print which rows ran.
    for strategy in "${streaming_strategies[@]}"; do
        echo "== streaming with filesystem strategy: $strategy"
        RUNAI_STREAMER_FS_STRATEGY="$strategy" run_torchrun "$strategy" 60 "$streaming_dir/test_synthetic_models.py" Rows
    done

    # Self-check of the test's own checks: when the test harness changed (the workflow decides), run the SelfChecks test
    # case, which injects a corrupted tensor and a leaked GPU tensor and asserts that the CRC32 check and the GPU memory
    # check report them, as themselves and with no other error. sync_buffered is enough: the faults do not depend on how
    # the file is read. It is an ordinary test case: its failure is unittest's, like that of a row.
    if [ "${GPU_SELF_CHECK:-false}" = true ]; then
        echo "== self-check: faults injected on purpose, the checks must report them"
        GPU_RESULTS="$GPU_RESULTS/self-check" RUNAI_STREAMER_FS_STRATEGY=sync_buffered \
            run_torchrun self_check 20 "$streaming_dir/test_synthetic_models.py" SelfChecks
    fi

    # The small real model, loaded with the streamer's default settings (no filesystem strategy chosen): what a user
    # runs. Without GPU_ROWS: a filter of the rows of the synthetic models would leave no smoke row.
    if [ "$smoke_expected" = true ]; then
        echo "== smoke test: the small real model, default settings"
        GPU_MATRIX=smoke GPU_RESULTS="$GPU_RESULTS/smoke" run_step real_smoke 20 \
            env -u RUNAI_STREAMER_FS_STRATEGY -u GPU_ROWS "${TORCHRUN[@]}" "$streaming_dir/test_synthetic_models.py" Rows
    fi
}

suite_expect() {
    local strategy
    for strategy in "${streaming_strategies[@]}"; do
        echo "results-$strategy-rank*.json"
    done
    if [ "${GPU_SELF_CHECK:-false}" = true ]; then
        echo "self-check/results-sync_buffered-rank*.json"
    fi
    if [ "$smoke_expected" = true ]; then
        echo "smoke/results-default-rank*.json"
    fi
}

# The manifests list every tensor of every model; keep them with the results to debug a failure.
suite_collect() {
    local m
    mkdir -p "$GPU_RESULTS/manifests"
    for m in mixed many_tensors giant_tensor many_files real_smoke; do
        if [ -f "$GPU_FIXTURES/$m.manifest.json" ]; then
            gzip -c "$GPU_FIXTURES/$m.manifest.json" > "$GPU_RESULTS/manifests/$m.manifest.json.gz" || true
        fi
    done
}

suite_timing() {
    echo "$GPU_FIXTURES/timing-gen.jsonl"
    echo "$GPU_FIXTURES/timing-real.jsonl"
}
