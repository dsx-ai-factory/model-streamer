## Overview

Model Streamer is a Python SDK designed to enhance model loading times, supporting several model formats (e.g., SafeTensors) as well as various storage types (network file systems, S3, Disk, etc.).

The Streamer uses multiple threads to read tensors concurrently from a file in some file or object storage to a dedicated buffer in the CPU memory. Every tensor is given an identifier that subsequently is used by the application to load the tensor to the GPU memory. This way the application can load tensors from the CPU memory to the GPU memory while other tensors are being read from storage to the CPU memory.

The model streaming utilizes OS-level concurrency to read data from local file systems, remote file systems, or object stores. The package employs a highly performant C++ layer to ensure maximum performance and minimum model load times, which is crucial for auto-scaling inference servers and keeping GPU idle times low. In addition to performance, a Python wrapper provides simple APIs and easy integration into an existing codebase.

Currently, the Streamer supports PyTorch applications only.

## Table of Contents

- [Installation](src/installation.md)
- [Usage](src/usage.md)
- [Environment Variables](src/env-vars.md)
- [Benchmarks](src/benchmarks.md)
