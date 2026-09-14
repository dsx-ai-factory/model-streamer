# Nightly GPU tests

The `Nightly GPU tests` workflow runs on pull requests, daily at **00:17 UTC**,
and through manual `workflow_dispatch` runs. It requires access to runner group
`nv-gpu-amd64-t4-2gpu`, label `linux-amd64-gpu-t4-latest-2`.
Scheduled runs begin once the workflow is on the repository's default branch.

The GPU devcontainer reuses the build Dockerfile and enables GPU passthrough,
2 GiB of shared memory for NCCL, and the existing io_uring seccomp setting.
PyTorch 2.4.1 with CUDA 12.4 is pinned for the Dockerfile's Python 3.8 runtime.
Do not run the CPU `make install` or `make test` targets in this environment:
they explicitly install CPU-only PyTorch.

Run inside `.devcontainer/devcontainer.gpu.json` on a Linux host with two GPUs:

```bash
bash tests/gpu/run.sh
```

The script builds the real C++ streamer and checks io_uring availability, then
runs two ranks with NCCL across the four filesystem strategy configurations.
It tests CPU-to-CUDA copies, direct CUDA tensor output, repeated loads, all three
distributed partition policies, and a rank with no local tensor to read.
Each rank checks every tensor's name, shape, dtype, values, and CUDA device,
including empty tensors and scalars. Synthetic fixtures require no model download
or cloud credentials. The host-buffer budget is 4 MiB per node and the distributed
staging-buffer minimum is 1 MiB, exercising reuse within T4 memory limits.

Missing GPUs/NCCL and unexpected CPU fallback fail the job. Per-strategy execution
is limited to 15 minutes, collective operations to 60 seconds, and the whole job
to 90 minutes. `test-results/gpu/run.log` is uploaded even when the test step fails.
These are correctness tests; they do not measure production storage throughput
or validate inference workloads.
