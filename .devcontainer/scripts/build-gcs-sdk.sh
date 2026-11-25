#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build for the image architecture supplied by Docker's TARGETARCH.
if (( $# != 1 )); then
    echo "Usage: $0 <amd64|arm64>" >&2
    exit 2
fi
case "$1" in
    amd64)
        arch=x86_64
        bazel_arch=x86_64
        ;;
    arm64)
        arch=aarch64
        bazel_arch=arm64
        ;;
    *)
        echo "Unsupported SDK architecture: $1" >&2
        exit 2
        ;;
esac

cd /build/gcs

# Install the SDK's pinned Bazel binary for the native image architecture.
bazel_version=$(cat .bazelversion)
bazel_url="https://github.com/bazelbuild/bazel/releases/download/${bazel_version}/bazel-${bazel_version}-linux-${bazel_arch}"
curl -fsSL -o /usr/local/bin/bazel "${bazel_url}"
# The checksum comes from the same release, so it detects a corrupt download, not a tampered release.
bazel_sha256=$(curl -fsSL "${bazel_url}.sha256" | awk '{print $1; exit}')
if [[ ! "${bazel_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Cannot read the Bazel ${bazel_version} checksum from ${bazel_url}.sha256" >&2
    exit 1
fi
echo "${bazel_sha256}  /usr/local/bin/bazel" | sha256sum -c -
chmod +x /usr/local/bin/bazel

python3 -B -m unittest discover -s /build/gcs -p test_install.py

CC=/usr/bin/gcc CXX=/usr/bin/g++ bazel build //:sdk --config="$arch" --jobs=4
python3 install.py bazel-bin/sdk.json "$(bazel info execution_root)" "/opt/gcs-sdk/$arch" \
    --build-inputs /build/gcs --compiler /usr/bin/g++
bazel shutdown
