#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p test-results/gpu
exec > >(tee test-results/gpu/run.log) 2>&1

nvidia-smi
# The shared Ubuntu 20.04 devcontainer uses Python 3.8. Keep a compatible
# CUDA build; make install/test would replace it with CPU-only PyTorch.
python3 -m pip install 'torch==2.4.1' --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r py/runai_model_streamer/requirements.dev

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

(
    cd cpp
    bazel build --config=x86_64 //streamer:libstreamer.so
    # Prove io_uring is usable rather than accepting a fallback or skipped test.
    bazel test --config=x86_64 //posix_io/io_uring_engine:io_uring_engine_test \
        --test_env=RUNAI_STREAMER_REQUIRE_IO_URING=1 --test_output=errors
)
python3 -m pip install --no-deps -e py/runai_model_streamer
export STREAMER_LIBRARY="$PWD/cpp/bazel-bin/streamer/libstreamer.so"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export RUNAI_STREAMER_DIST_TIMEOUT=60
export RUNAI_STREAMER_LOG_TO_STDERR=1

for strategy in \
    sync_buffered \
    libaio_direct,sync_buffered \
    io_uring_direct,sync_buffered \
    io_uring_buffered,sync_buffered
do
    echo "Testing GPU streaming with filesystem strategy: $strategy"
    RUNAI_STREAMER_FS_STRATEGY="$strategy" \
        timeout --signal=TERM --kill-after=30s 15m \
        torchrun --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
        tests/gpu/run_tests.py
done
