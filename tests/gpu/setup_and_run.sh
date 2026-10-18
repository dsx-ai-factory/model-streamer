#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Real-GPU test run, inside the GPU devcontainer on a host with two GPUs:
#   1. set up: CUDA PyTorch, the real libstreamer, a proof that io_uring works here
#   2. generate the four synthetic models (in parallel with the build) and check them
#   3. stream them to the GPUs once per filesystem strategy and compare every tensor
# Every step is timed; the table at the end shows where the time went.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RESULTS=${GPU_RESULTS:-test-results/gpu}
FIXTURES=${GPU_FIXTURES:-.gpu-fixtures}
SEED=${GPU_SEED:-$(date +%s)}
export GPU_RESULTS="$RESULTS" GPU_FIXTURES="$FIXTURES"
mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/run.log") 2>&1

SETUP_TIMING="$RESULTS/timing-setup.jsonl"
: > "$SETUP_TIMING"

# record NAME START END: append the wall time of a phase to the timing file and print it.
record() {
    local secs
    secs=$(awk -v s="$2" -v e="$3" 'BEGIN { printf "%.2f", e - s }')
    printf '{"phase": "%s", "seconds": %s}\n' "$1" "$secs" >> "$SETUP_TIMING"
    echo "[$(date +%T)] END   $1  ${secs}s"
}

# phase NAME COMMAND...: run a command, print START and END lines, and record its wall time.
phase() {
    local name=$1; shift
    local start rc=0
    echo "[$(date +%T)] START $name"
    start=$(date +%s.%N)
    "$@" || rc=$?
    record "$name" "$start" "$(date +%s.%N)"
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%T)] FAILED $name (exit $rc)"
    fi
    return "$rc"
}

echo "== environment"
date -u
uname -r
echo "seed: $SEED"
python3 --version
nvidia-smi
echo "ulimit -n: $(ulimit -n)   ulimit -l: $(ulimit -l)"
free -g | head -2
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
mkdir -p "$FIXTURES"
echo "fixtures in $FIXTURES: $(stat -f -c %T "$FIXTURES")"
df -h "$FIXTURES" | tail -1

install_python_packages() {
    # The shared Ubuntu 20.04 devcontainer uses Python 3.8. Keep a compatible CUDA build; make
    # install/test would replace it with CPU-only PyTorch.
    python3 -m pip install 'torch==2.4.1' --index-url https://download.pytorch.org/whl/cu124
    python3 -m pip install -r py/runai_model_streamer/requirements.dev
}
phase install_python_packages install_python_packages

python3 - <<'PY'
import torch
import torch.distributed as dist

print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("GPU tests require two visible CUDA GPUs")
if not dist.is_nccl_available():
    raise RuntimeError("GPU tests require an NCCL-enabled PyTorch build")
for index in range(2):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
PY

echo "== generator self-test"
phase generator_unit_tests python3 -m unittest tests/gpu/test_gen_models.py

echo "== generate synthetic models, in the background while libstreamer builds"
rm -rf "$FIXTURES"
mkdir -p "$FIXTURES"
python3 tests/gpu/gen_models.py generate --out "$FIXTURES" --seed "$SEED" > "$RESULTS/generate.log" 2>&1 &
GENERATOR_PID=$!

build_libstreamer() {
    (
        cd cpp
        bazel build --config=x86_64 //streamer:libstreamer.so
        # Prove io_uring is usable rather than accepting a fallback or skipped test.
        bazel test --config=x86_64 //posix_io/io_uring_engine:io_uring_engine_test \
            --test_env=RUNAI_STREAMER_REQUIRE_IO_URING=1 --test_output=errors
    )
    python3 -m pip install --no-deps -e py/runai_model_streamer
}
phase build_libstreamer build_libstreamer

wait_for_generator() {
    if ! wait "$GENERATOR_PID"; then
        cat "$RESULTS/generate.log"
        echo "fixture generation failed"
        return 1
    fi
    cat "$RESULTS/generate.log"
}
phase wait_for_generator wait_for_generator

export STREAMER_LIBRARY="$PWD/cpp/bazel-bin/streamer/libstreamer.so"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export RUNAI_STREAMER_LOG_TO_STDERR=1

status=0
for strategy in \
    sync_buffered \
    libaio_direct,sync_buffered \
    io_uring_direct,sync_buffered \
    io_uring_buffered,sync_buffered
do
    echo "[$(date +%T)] START run.$strategy"
    echo "== streaming with filesystem strategy: $strategy"
    start=$(date +%s.%N)
    if ! RUNAI_STREAMER_FS_STRATEGY="$strategy" \
        timeout --signal=TERM --kill-after=30s 45m \
        torchrun --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
        tests/gpu/test_synthetic_models.py; then
        status=1
        echo "[$(date +%T)] FAILED run.$strategy"
    fi
    record "run.$strategy" "$start" "$(date +%s.%N)"
done

# The manifests list every tensor of every model; keep them with the results to debug a failure.
mkdir -p "$RESULTS/manifests"
for m in A B C D; do gzip -c "$FIXTURES/$m.manifest.json" > "$RESULTS/manifests/$m.manifest.json.gz" || true; done

echo "== where the time went"
python3 tests/gpu/phase_timer.py "$RESULTS"/timing-*.jsonl "$FIXTURES"/timing-gen.jsonl | tee "$RESULTS/timing-summary.txt"

exit "$status"
