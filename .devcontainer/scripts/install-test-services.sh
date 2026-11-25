#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 <amd64|arm64>" >&2
    exit 2
fi
target_arch=$1
case "$target_arch" in
    amd64) weed_sha256=31fb804858885f9e7f18b6d3b1da09e824baac3e6a55b5a62c3c4c77e6ed6d7d ;;
    arm64) weed_sha256=ba5c9def9ff98f78becdf309a50a9f906922f6383a7c4c772d01ede3cc012e16 ;;
    *)
        echo "Unsupported test-services architecture: $target_arch" >&2
        exit 2
        ;;
esac

# SeaweedFS provides the S3-compatible service for local tests.
curl -fsSL -o weed.tar.gz "https://github.com/seaweedfs/seaweedfs/releases/download/4.47/linux_${target_arch}.tar.gz"
echo "${weed_sha256}  weed.tar.gz" | sha256sum -c -
tar zxf weed.tar.gz
chmod +x weed
mv weed /usr/local/bin/weed
rm -f weed.tar.gz

# Local Google Cloud Storage emulator (separate from the GCS client SDK).
curl -fsSL -o fake-gcs-server_1.52.2.tar.gz "https://github.com/fsouza/fake-gcs-server/releases/download/v1.52.2/fake-gcs-server_1.52.2_Linux_${target_arch}.tar.gz"
mkdir -p fake-gcs-server
tar zxf fake-gcs-server_1.52.2.tar.gz -C fake-gcs-server
chmod +x fake-gcs-server/fake-gcs-server
mv fake-gcs-server/fake-gcs-server /usr/local/bin/fake-gcs-server
rm -rf fake-gcs-server

# Azurite provides the local Azure storage emulator and requires Node.js.
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g azurite
rm -rf /var/lib/apt/lists/*
