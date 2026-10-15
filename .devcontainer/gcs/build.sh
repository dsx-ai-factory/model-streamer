#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd /build/gcs
for arch in x86_64 aarch64; do
    bazel build //:sdk --config="$arch" --jobs=4
    python3 install.py bazel-bin/sdk.json "$(bazel info execution_root)" "/opt/gcs-sdk/$arch"
done
bazel shutdown
