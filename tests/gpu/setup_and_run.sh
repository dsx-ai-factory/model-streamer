#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Real-GPU test run, inside the GPU devcontainer on a host with two GPUs:
#   1. set up: CUDA PyTorch, the real libstreamer, a proof that io_uring works here
#   2. generate the four synthetic models, with every dtype the streamer accepts here, and check them
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
echo "matrix: ${GPU_MATRIX:-core}"
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
    # phase runs this function where `set -e` does not apply, so every command ends in `|| return`.
    python3 -m pip install 'torch==2.4.1' --index-url https://download.pytorch.org/whl/cu124 || return
    python3 -m pip install -r py/runai_model_streamer/requirements.dev || return
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
phase generator_unit_tests python3 -m unittest tests/gpu/test_gen_models.py tests/gpu/test_matrix.py

build_libstreamer() {
    # Called through phase, where `set -e` does not apply (not inside the subshell either): every command ends in
    # `|| return` or `|| exit`, or a failed build or io_uring check would be hidden by the install that follows.
    (
        cd cpp || exit
        bazel build --config=x86_64 //streamer:libstreamer.so || exit
        # Prove io_uring is usable rather than accepting a fallback or skipped test.
        bazel test --config=x86_64 //posix_io/io_uring_engine:io_uring_engine_test \
            --test_env=RUNAI_STREAMER_REQUIRE_IO_URING=1 --test_output=errors || exit
    ) || return
    python3 -m pip install --no-deps -e py/runai_model_streamer || return
}
phase build_libstreamer build_libstreamer

export STREAMER_LIBRARY="$PWD/cpp/bazel-bin/streamer/libstreamer.so"

# The models hold every dtype that the streamer accepts with the installed PyTorch and that the generator knows. The
# streamer's own table decides, so the test follows it when it gains a dtype, and a dtype that this PyTorch lacks is
# listed here as not covered rather than failing the load of every file that holds it. (F4 is known to the streamer
# with PyTorch 2.8 or later but not to the generator, see gen_models.py.) Generation takes under a minute.
generate_models() {
    local dtypes
    dtypes=$(python3 - <<'PY'
import sys
sys.path.insert(0, "tests/gpu")
import gen_models
from runai_model_streamer.safetensors_streamer.safetensors_pytorch import safetensors_to_torch_dtype

streamer = set(safetensors_to_torch_dtype)
covered = sorted(streamer & set(gen_models.DTYPES))
print("streamer dtypes: %s" % " ".join(sorted(streamer)), file=sys.stderr)
print("not generated:   %s" % (" ".join(sorted(set(gen_models.DTYPES) - streamer)) or "-"), file=sys.stderr)
unknown = sorted(streamer - set(gen_models.DTYPES) - gen_models.KNOWN_GAPS)
print("not known to the generator: %s" % (" ".join(unknown) or "-"), file=sys.stderr)
if unknown:
    print("ERROR: the streamer accepts %s, which the generator cannot write: add it to DTYPES in tests/gpu/gen_models.py"
          % " ".join(unknown), file=sys.stderr)
    sys.exit(1)
print(",".join(covered))
PY
    ) || return
    rm -rf "$FIXTURES"
    mkdir -p "$FIXTURES"
    python3 tests/gpu/gen_models.py generate --out "$FIXTURES" --seed "$SEED" --dtypes "$dtypes" \
        2>&1 | tee "$RESULTS/generate.log"
    return "${PIPESTATUS[0]}"
}
echo "== generate synthetic models"
phase generate_models generate_models

# The direct I/O strategies are only tested if the fixtures are on a real block device that serves O_DIRECT. The
# streamer reads a file from a mount that cannot (tmpfs, for one) with its synchronous reader, quietly, so on such
# a storage a direct strategy would pass without ever being used. Fail instead.
check_fixture_storage() {
    local source fstype
    source=$(df --output=source "$FIXTURES" | tail -1) || return
    fstype=$(stat -f -c %T "$FIXTURES") || return
    echo "fixtures storage: $source ($fstype)"
    case "$source" in
        /dev/*) ;;
        *) echo "ERROR: the fixtures are on $source ($fstype), not on a block device; the direct I/O strategies would not be tested" >&2
           return 1 ;;
    esac
    python3 - "$FIXTURES" <<'PY' || { echo "ERROR: the fixtures storage does not serve O_DIRECT reads" >&2; return 1; }
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
check_fixture_storage || exit 1

export GPU_MATRIX="${GPU_MATRIX:-core}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export RUNAI_STREAMER_LOG_TO_STDERR=1

# The many_files model has 5,000 files, and the container starts with a soft limit of 1024 open files. The streamer keeps a
# file open until its whole submission ends (https://github.com/dsx-ai-factory/model-streamer/issues/261), so
# the soft limit is raised to the hard limit for the test. Remove this once that is fixed.
echo "open files limit before: soft $(ulimit -Sn), hard $(ulimit -Hn)"
ulimit -Sn "$(ulimit -Hn)" || echo "could not raise the open files limit"
echo "open files limit now: soft $(ulimit -Sn)"

# Each reader alone, with no sync_buffered fallback: a list the host cannot serve is an error in the streamer, so
# a run under a reader that is not available fails instead of passing on the synchronous reader. (The streamer
# lists in the Makefile's test_strategies keep sync_buffered last for hosts of any kind.)
status=0
for strategy in \
    sync_buffered \
    libaio_direct \
    io_uring_direct \
    io_uring_buffered
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

# Self-check of the test's own checks: when the test harness changed (the workflow decides), run two rows once more with
# a fault injected on purpose, a corrupted tensor and a leaked GPU tensor. This pass succeeds only if the CRC32 check and
# the GPU memory check report them. sync_buffered is enough: the faults do not depend on how the file is read.
if [ "${GPU_SELF_CHECK:-false}" = true ]; then
    echo "[$(date +%T)] START run.self_check"
    echo "== self-check: faults injected on purpose, the checks must report them"
    start=$(date +%s.%N)
    if ! GPU_ROWS="mixed.solo.owned.cuda.tight,mixed.solo.borrowed.cuda.tight" \
        GPU_DRILL="corrupt:mixed.solo.owned.cuda.tight,leak:mixed.solo.borrowed.cuda.tight" \
        GPU_RESULTS="$RESULTS/self-check" RUNAI_STREAMER_FS_STRATEGY=sync_buffered \
        timeout --signal=TERM --kill-after=30s 15m \
        torchrun --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
        tests/gpu/test_synthetic_models.py; then
        status=1
        echo "[$(date +%T)] FAILED run.self_check"
    fi
    record "run.self_check" "$start" "$(date +%s.%N)"
fi

# Which rows ran. A rank that fails inside a distributed row makes the NCCL watchdog abort the other one, which ends
# that strategy's run: the rows after it never start. They must not hide behind the strategies that did finish.
echo "== rows that ran"
python3 - "$RESULTS" sync_buffered libaio_direct io_uring_direct io_uring_buffered <<'PY' || status=1
import glob
import json
import os
import sys

results, strategies = sys.argv[1], sys.argv[2:]
incomplete = False
for path in sorted(glob.glob(os.path.join(results, "**", "rows-*-rank*.json"), recursive=True)):
    data = json.load(open(path))
    rows = data["rows"]
    finished = [r for r in rows if r["ok"] is not None]
    unexpected = [r["row"] for r in finished if not r["ok"] and not r.get("drill")]
    line = "%s: %d of %d rows finished, %d failed" % (os.path.relpath(path, results), len(finished),
                                                       data["expected_rows"], len(unexpected))
    killed = [r["row"] for r in rows if r["ok"] is None]
    if killed:
        line += "; the run ended inside row " + killed[0]
    if len(finished) < data["expected_rows"]:
        incomplete = True
        line += "  <-- INCOMPLETE"
    print(line)
for strategy in strategies:
    if not glob.glob(os.path.join(results, "rows-%s-rank*.json" % strategy)):
        incomplete = True
        print("%s: no results at all  <-- INCOMPLETE" % strategy)
sys.exit(1 if incomplete else 0)
PY

# The manifests list every tensor of every model; keep them with the results to debug a failure.
mkdir -p "$RESULTS/manifests"
for m in mixed many_tensors giant_tensor many_files; do gzip -c "$FIXTURES/$m.manifest.json" > "$RESULTS/manifests/$m.manifest.json.gz" || true; done

echo "== where the time went"
python3 tests/gpu/phase_timer.py "$RESULTS"/timing-*.jsonl "$FIXTURES"/timing-gen.jsonl | tee "$RESULTS/timing-summary.txt"

exit "$status"
