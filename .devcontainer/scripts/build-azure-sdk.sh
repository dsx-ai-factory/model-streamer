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

# Use the prebuilt libraries in /opt instead of Azure's automatic vcpkg toolchain.
export AZURE_SDK_DISABLE_AUTO_VCPKG=1

git clone --recurse-submodules https://github.com/Azure/azure-sdk-for-cpp /opt/azure-sdk-cpp
cd /opt/azure-sdk-cpp
git checkout "azure-storage-blobs_${AZURE_STORAGE_BLOBS_VERSION:?}"

# Helper function to build Azure SDK components
build_azure_sdk_components() {
    local arch=$1 prefix=$2 source=$3 extra_opts=$4
    # CMAKE_BUILD_TYPE is REQUIRED, not a nicety. With it unset cmake adds no optimisation flag at
    # all and gcc falls back to its own default of -O0. Measured before this line: every function
    # in libazure-core.a and libazure-storage-blobs.a opened with a frame-pointer prologue and
    # spilled its arguments to the stack, which is what -O0 does and -O1 upwards does not.
    #
    # aws-sdk-cpp gets away with the same omission because it sets a default build type in its own
    # CMakeLists. azure-sdk-for-cpp does not, so it has to be stated here.
    local common_opts="-DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_INSTALL_PREFIX=${prefix} \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DCURL_LIBRARY=${prefix%azure}curl/lib/libcurl.a \
        -DCURL_INCLUDE_DIR=${prefix%azure}curl/include \
        -DOPENSSL_ROOT_DIR=${prefix%azure}ssl \
        -DOPENSSL_INCLUDE_DIR=${prefix%azure}ssl/include \
        -DOPENSSL_CRYPTO_LIBRARY=${prefix%azure}ssl/lib/libcrypto.a \
        -DOPENSSL_SSL_LIBRARY=${prefix%azure}ssl/lib/libssl.a \
        -DLIBXML2_LIBRARY=${prefix%azure}xml2/lib/libxml2.a \
        -DLIBXML2_INCLUDE_DIR=${prefix%azure}xml2/include/libxml2"
    cmake -B build-${arch}-${source##*/} -S ${source} \
        ${common_opts} \
        -DCMAKE_PREFIX_PATH="${prefix%azure}ssl;${prefix%azure}curl;${prefix%azure}xml2${extra_opts:+;}${extra_opts}" \
        ${@:5}
    cmake --build build-${arch}-${source##*/} -j4
    cmake --install build-${arch}-${source##*/}
}

build_azure_sdk_components "$arch" "/opt/${arch}-azure" sdk/core/azure-core "" -DBUILD_TRANSPORT_CURL=ON -DDISABLE_AZURE_CORE_OPENTELEMETRY=ON
build_azure_sdk_components "$arch" "/opt/${arch}-azure" sdk/storage/azure-storage-common "/opt/${arch}-azure"
build_azure_sdk_components "$arch" "/opt/${arch}-azure" sdk/storage/azure-storage-blobs "/opt/${arch}-azure"
build_azure_sdk_components "$arch" "/opt/${arch}-azure" sdk/identity/azure-identity "/opt/${arch}-azure"

rm -rf /opt/azure-sdk-cpp

# Expose the native SDK to tools using the standard include and library paths.
if [ -d /opt/${arch}-azure ]; then
    ln -sf /opt/${arch}-azure/include/azure /usr/local/include/azure
    ln -sf /opt/${arch}-azure/lib/* /usr/local/lib/
fi
