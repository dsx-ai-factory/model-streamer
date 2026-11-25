#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Official CUDA 12.8 component archives and SHA-256 pins:
# https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json
if (( $# != 1 )); then
    echo "Usage: $0 <amd64|arm64>" >&2
    exit 2
fi

case "$1" in
    amd64)
        CUDA_PLATFORM=linux-x86_64
        CUDA_SHA256=5bd3ac35ea8e8ab880e595d5054ee373abf6d9e53dcb8cef0a5c75358dbc0ae2
        ;;
    arm64)
        CUDA_PLATFORM=linux-sbsa
        CUDA_SHA256=b94814dacc8e4e69c8909ac08ae9f8d6831c1a141358a6e9f60583481a49ae2b
        ;;
    *)
        echo "Unsupported CUDA headers architecture: $1" >&2
        exit 1
        ;;
esac

CUDA_ARCHIVE="cuda_cudart-${CUDA_PLATFORM}-12.8.57-archive"
curl -fsSL -o /tmp/cuda-cudart.tar.xz \
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/${CUDA_PLATFORM}/${CUDA_ARCHIVE}.tar.xz"
echo "${CUDA_SHA256}  /tmp/cuda-cudart.tar.xz" | sha256sum -c -
mkdir -p /usr/local/cuda
tar -xJf /tmp/cuda-cudart.tar.xz -C /usr/local/cuda --strip-components=1 \
    "${CUDA_ARCHIVE}/include" "${CUDA_ARCHIVE}/LICENSE"
rm /tmp/cuda-cudart.tar.xz
