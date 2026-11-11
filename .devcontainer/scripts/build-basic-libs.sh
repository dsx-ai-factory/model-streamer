#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build for the image architecture supplied by Docker's TARGETARCH.
if (( $# != 1 )); then
    echo "Usage: $0 <amd64|arm64>" >&2
    exit 2
fi
# Versions come from the Dockerfile's annotated ARGs, which Renovate keeps current.
: "${OPENSSL_VERSION:?}" "${CURL_VERSION:?}" "${ZLIB_VERSION:?}" "${LIBXML2_VERSION:?}"
CURL_DOTTED="${CURL_VERSION//_/.}"
LIBXML2_SERIES="${LIBXML2_VERSION%.*}"

case "$1" in
    amd64) arch=x86_64 ;;
    arm64) arch=aarch64 ;;
    *)
        echo "Unsupported architecture: $1" >&2
        exit 2
        ;;
esac

build_openssl() {
    curl -fsSL -o openssl-${OPENSSL_VERSION}.tar.gz https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz
    tar zxf openssl-${OPENSSL_VERSION}.tar.gz
    cd openssl-${OPENSSL_VERSION}

    ./Configure no-ssl2 no-zlib no-shared no-comp no-dynamic-engine enable-ec_nistp_64_gcc_128 --prefix=/opt/${arch}-ssl --openssldir=/opt/${arch}-ssl --libdir=lib linux-${arch}
    make -j4 > /dev/null
    make install_sw > /dev/null
    make clean > /dev/null
    make distclean > /dev/null

    cd -
    rm -rf openssl-${OPENSSL_VERSION} openssl-${OPENSSL_VERSION}.tar.gz
}

build_curl() {
    curl -fsSL -o curl-${CURL_DOTTED}.tar.gz https://github.com/curl/curl/releases/download/curl-${CURL_VERSION}/curl-${CURL_DOTTED}.tar.gz
    tar zxf curl-${CURL_DOTTED}.tar.gz
    cd curl-${CURL_DOTTED}

    CPPFLAGS=-I/opt/${arch}-ssl/include \
    LDFLAGS=-L/opt/${arch}-ssl/lib \
    CFLAGS="-fPIC" \
    ./configure --disable-shared --without-libpsl --with-openssl=/opt/${arch}-ssl --prefix=/opt/${arch}-curl
    make VERBOSE=1 -j4 > /dev/null
    make install > /dev/null
    make clean > /dev/null

    cd -
    rm -rf curl-${CURL_DOTTED} curl-${CURL_DOTTED}.tar.gz
}

build_zlib() {
    curl -fsSL -o zlib-${ZLIB_VERSION}.tar.gz https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz
    tar zxf zlib-${ZLIB_VERSION}.tar.gz
    cd zlib-${ZLIB_VERSION}

    ./configure --static --shared --prefix=/opt/${arch}-zlib
    make -j4 > /dev/null
    make install > /dev/null
    make clean > /dev/null

    cd -
    rm -rf zlib-${ZLIB_VERSION} zlib-${ZLIB_VERSION}.tar.gz
}

# -O2 is explicit because setting CFLAGS at all REPLACES autoconf's own default of "-g -O2".
# Passing only -fPIC therefore silently removed the optimisation. curl survives the same treatment
# a few blocks up only because its configure applies --enable-optimize itself; libxml2 does not.
build_libxml2() {
    curl -fsSL -o libxml2-${LIBXML2_VERSION}.tar.xz https://download.gnome.org/sources/libxml2/${LIBXML2_SERIES}/libxml2-${LIBXML2_VERSION}.tar.xz
    tar xf libxml2-${LIBXML2_VERSION}.tar.xz
    cd libxml2-${LIBXML2_VERSION}

    CFLAGS="-fPIC -O2" \
    ./configure \
    --disable-shared --enable-static \
    --without-python --without-lzma --without-zlib \
    --prefix=/opt/${arch}-xml2
    make -j4 > /dev/null
    make install > /dev/null
    make distclean > /dev/null

    cd -
    rm -rf libxml2-${LIBXML2_VERSION} libxml2-${LIBXML2_VERSION}.tar.xz
}

build_openssl
build_curl
build_zlib
build_libxml2
