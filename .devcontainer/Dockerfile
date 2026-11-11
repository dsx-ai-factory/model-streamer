# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:20.04 AS base

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# `make build` runs `setup.py bdist_wheel` directly, with no PEP 517 isolation, so it never
# reads py/*/pyproject.toml's setuptools pin. Keep this in sync with that pin by hand. Exact
# pin, not a floor, to match this file's other dependencies. Newest versions still compatible
# with this image's Python 3.8 - setuptools 76.0.0 needs >=3.9, wheel 0.46.0 needs >=3.9.
# Upgrade Ubuntu's pip before setuptools to avoid legacy version-parsing failures.
# pip 25.1 drops Python 3.8 support.
RUN python3 -m pip install --upgrade "pip==25.0.1" \
    && python3 -m pip install "setuptools==75.3.4" "wheel==0.45.1"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential binutils \
    && rm -rf /var/lib/apt/lists/*

ARG TARGETARCH

# ThreadSanitizer (`bazel test --config=tsan`).
#
# Ubuntu 20.04 ships libtsan.so for gcc-9 but NOT libtsan_preinit.o, so -fsanitize=thread fails to
# link. This is that file, which is all of GCC's own libsanitizer/tsan/tsan_preinit.c: it registers
# __tsan_init in .preinit_array so the runtime starts before any constructor runs.
#
# The `test -f libtsan.so` is not a formality: -print-file-name echoes its argument back when the
# file is not there, so without it the object would land in the build directory and --config=tsan
# would keep failing with the same confusing error. The second test never shadows a real
# libtsan_preinit.o, in case a later base image ships one. Retain the AMD64 workaround
# in the native AMD64 image.
RUN if [ "$TARGETARCH" = amd64 ]; then \
    printf '%s\n' \
        'extern void __tsan_init(void);' \
        'static void __local_tsan_preinit(void) { __tsan_init(); }' \
        '__attribute__((section(".preinit_array"), used))' \
        'static void (*tsan_preinit)(void) = __local_tsan_preinit;' \
        > /tmp/tsan_preinit.c \
    && TSAN_DIR="$(dirname "$(gcc -print-file-name=libtsan.so)")" \
    && test -f "$TSAN_DIR/libtsan.so" \
    && { test -f "$TSAN_DIR/libtsan_preinit.o" \
         || gcc -c -fPIC /tmp/tsan_preinit.c -o "$TSAN_DIR/libtsan_preinit.o"; } \
    && rm /tmp/tsan_preinit.c; \
    fi

RUN pip install \
    boto3 \
    cpplint==1.5.4 \
    pytest

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    git \
    curl \
    xz-utils \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Local storage services used by the integration tests.
COPY .devcontainer/scripts/install-test-services.sh /tmp/install-test-services.sh
RUN bash /tmp/install-test-services.sh "$TARGETARCH"

# CUDA headers only, for compile-time type checking of the driver API. No runtime and no
# libraries: libcuda.so.1 is loaded with dlopen, so the build has no link-time CUDA dependency
# and the image needs no driver.
COPY .devcontainer/scripts/install-cuda-headers.sh /tmp/install-cuda-headers.sh
RUN bash /tmp/install-cuda-headers.sh "$TARGETARCH"

# Build all shared SDK prerequisites before the cloud SDKs. Renovate keeps each annotated version
# current; the script reads the versions from these ARGs.
# attribution: shipped module=py/runai_model_streamer_s3,py/runai_model_streamer_azure,cpp
# renovate: datasource=github-tags depName=openssl/openssl extractVersion=^openssl-(?<version>\d+\.\d+\.\d+)$
ARG OPENSSL_VERSION=3.5.0
# attribution: shipped module=py/runai_model_streamer_s3,py/runai_model_streamer_azure,cpp
# renovate: datasource=github-releases depName=curl/curl versioning=regex:^(?<major>\d+)_(?<minor>\d+)_(?<patch>\d+)$ extractVersion=^curl-(?<version>\d+_\d+_\d+)$
ARG CURL_VERSION=8_13_0
# attribution: shipped module=py/runai_model_streamer_s3,py/runai_model_streamer_azure,cpp
# renovate: datasource=github-releases depName=madler/zlib extractVersion=^v(?<version>.*)$
ARG ZLIB_VERSION=1.3.1
# attribution: shipped module=py/runai_model_streamer_azure,cpp
# renovate: datasource=github-tags depName=GNOME/libxml2 extractVersion=^v(?<version>.*)$
ARG LIBXML2_VERSION=2.13.5
COPY .devcontainer/scripts/build-basic-libs.sh /tmp/build-basic-libs.sh
RUN OPENSSL_VERSION="$OPENSSL_VERSION" CURL_VERSION="$CURL_VERSION" ZLIB_VERSION="$ZLIB_VERSION" \
    LIBXML2_VERSION="$LIBXML2_VERSION" bash /tmp/build-basic-libs.sh "$TARGETARCH"

# Keep AWS in the final image's layer chain so CI can reuse its build layer
# through the published image cache when later dependencies change.
# Cloned with its submodules (aws-crt-cpp and the aws-c-* libraries), which its tag pins.
# attribution: shipped module=py/runai_model_streamer_s3,cpp submodules
# renovate: datasource=github-tags depName=aws/aws-sdk-cpp
ARG AWS_SDK_CPP_VERSION=1.11.924
COPY .devcontainer/scripts/build-aws-sdk.sh /tmp/build-aws-sdk.sh
RUN AWS_SDK_CPP_VERSION="$AWS_SDK_CPP_VERSION" bash /tmp/build-aws-sdk.sh "$TARGETARCH"

# Build Azure SDK for C++ (optimized with shared CMake options)
SHELL ["/bin/bash", "-c"]
# attribution: shipped module=py/runai_model_streamer_azure,cpp
# renovate: datasource=github-tags depName=Azure/azure-sdk-for-cpp extractVersion=^azure-storage-blobs_(?<version>.*)$
ARG AZURE_STORAGE_BLOBS_VERSION=12.15.0
COPY .devcontainer/scripts/build-azure-sdk.sh /tmp/build-azure-sdk.sh
RUN AZURE_STORAGE_BLOBS_VERSION="$AZURE_STORAGE_BLOBS_VERSION" bash /tmp/build-azure-sdk.sh "$TARGETARCH"

# Compile the pinned Google SDK once per image architecture. Only the installed
# archives and headers enter the final image, not sources or the Bazel cache.
FROM base AS gcs-builder
COPY .devcontainer/gcs/.bazelversion /build/gcs/
COPY .devcontainer/gcs/.bazelrc .devcontainer/gcs/MODULE.bazel .devcontainer/gcs/BUILD.bazel \
    .devcontainer/gcs/abi_check.cc .devcontainer/gcs/export.bzl .devcontainer/gcs/install.py \
    .devcontainer/gcs/test_install.py /build/gcs/
COPY .devcontainer/gcs/config/ /build/gcs/config/
COPY .devcontainer/scripts/build-gcs-sdk.sh /build/scripts/build-gcs-sdk.sh
# Cache the SDK as a Docker layer. Bazel caches contain host-specific repository
# configuration and must not be shared between native image builds.
RUN bash /build/scripts/build-gcs-sdk.sh "$TARGETARCH"

FROM base AS devcontainer
COPY --from=gcs-builder /opt/gcs-sdk /opt/gcs-sdk

# Keep I/O libraries last so their changes can reuse the cloud SDK build layers.
# attribution: shipped module=py/runai_model_streamer,cpp
# renovate: datasource=github-tags depName=axboe/liburing extractVersion=^liburing-(?<version>.*)$
ARG LIBURING_VERSION=2.14
ARG LIBURING_SHA256=5f80964108981c6ad979c735f0b4877d5f49914c2a062f8e88282f26bf61de0c
ARG LIBURING_URL=https://github.com/axboe/liburing/archive/refs/tags/liburing-${LIBURING_VERSION}.tar.gz
COPY .devcontainer/scripts/build-io-libs.sh /tmp/build-io-libs.sh
RUN LIBURING_VERSION="$LIBURING_VERSION" LIBURING_SHA256="$LIBURING_SHA256" LIBURING_URL="$LIBURING_URL" \
    bash /tmp/build-io-libs.sh "$TARGETARCH"
