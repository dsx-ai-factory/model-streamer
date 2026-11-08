#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build for the image architecture supplied by Docker's TARGETARCH.
if (( $# != 1 )); then
    echo "Usage: $0 <amd64|arm64>" >&2
    exit 2
fi
case "$1" in
    amd64) arch=x86_64 ;;
    arm64) arch=aarch64 ;;
    *)
        echo "Unsupported architecture: $1" >&2
        exit 2
        ;;
esac

git clone --recurse-submodules --depth 1 --branch "1.11.584" https://github.com/aws/aws-sdk-cpp /opt/aws-sdk-cpp
cd /opt/aws-sdk-cpp

cmake -DBUILD_ONLY="s3-crt" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_PREFIX_PATH="/opt/${arch}-ssl;/opt/${arch}-curl" \
      -DCMAKE_INSTALL_PREFIX="/opt/${arch}-aws" \
      -DFORCE_SHARED_CRT=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DENABLE_TESTING=OFF \
      -DAUTORUN_UNIT_TESTS=OFF \
      -DUSE_CRT_HTTP_CLIENT=ON \
      -DZLIB_LIBRARY="/opt/${arch}-zlib/lib/libz.a" \
      -DZLIB_INCLUDE_DIR="/opt/${arch}-zlib/include" \
      -Dcrypto_LIBRARY="/opt/${arch}-ssl/lib/libcrypto.a" \
      -Dcrypto_INCLUDE_DIR="/opt/${arch}-ssl/include" \
      -DCURL_LIBRARY="/opt/${arch}-curl/lib/libcurl.a" \
      -DCURL_INCLUDE_DIR="/opt/${arch}-curl/include" \
      -DOPENSSL_CRYPTO_LIBRARY="/opt/${arch}-ssl/lib/libcrypto.a" \
      -DOPENSSL_SSL_LIBRARY="/opt/${arch}-ssl/lib/libssl.a" \
      -DOPENSSL_INCLUDE_DIR="/opt/${arch}-ssl/include"
make -j4
make install
rm -rf /opt/aws-sdk-cpp
