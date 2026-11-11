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

# liburing, for the io_uring engine. Static only: it is linked into libstreamer.so, which exports an
# explicit allowlist via streamer.ldscript, so these symbols stay hidden.
#   --use-libc    route syscalls through libc wrappers instead of raw syscall instructions, so
#                 sanitizers and interposition can see them
#   ENABLE_SHARED=0  liburing has no ./configure --disable-shared; the switch is a make variable
#
# CHECKSUMMED, like libaio below and like googletest in cpp/rules.bzl.
#
# An earlier version of this block skipped the check, reasoning that GitHub's generated /archive/
# tarballs are rebuilt on demand and have changed bytes for an unchanged tag before. That reasoning
# does not survive contact with what this repository already does: rules.bzl pins the sha256 of a
# GitHub-generated archive for googletest and has been fine.
#
# The trade is the right way round. Without a pin, altered bytes are linked statically into
# libstreamer.so and shipped, silently. With one, a regenerated archive fails the image build loudly -
# and a loud failure that needs a one-line update beats an unverified dependency in a release
# artifact.
build_liburing() {
    curl -fsSL -o liburing-${LIBURING_VERSION:?}.tar.gz "${LIBURING_URL:?}"
    echo "${LIBURING_SHA256:?}  liburing-${LIBURING_VERSION}.tar.gz" | sha256sum -c -
    tar zxf liburing-${LIBURING_VERSION}.tar.gz
    cd liburing-liburing-${LIBURING_VERSION}

    ./configure --use-libc --prefix=/opt/${arch}-uring --libdir=/opt/${arch}-uring/lib
    make -C src -j4 ENABLE_SHARED=0 CFLAGS="-fPIC -O2" > /dev/null
    make install ENABLE_SHARED=0 > /dev/null
    make clean > /dev/null

    cd -
    rm -rf liburing-liburing-${LIBURING_VERSION} liburing-${LIBURING_VERSION}.tar.gz
}

# libaio, for the fallback engine where io_uring is unavailable. Static only, same reasoning as
# liburing. Its symbols carry .symver directives, which need no version nodes here because
# streamer.ldscript ends "local: *" and localizes them.
#   CFLAGS -I.       libaio has no ./configure, and its sources include <libaio.h> from src/
#   copy, not make install   its install target also lays down libaio.so and its sonames; we want
#                            only the archive
#   the URL     Debian's orig tarball, byte-identical to the upstream release. releases.pagure.org
#               (pagure.io was sunset in 2026) now returns 404.
#   the sha256  so the host is not load-bearing. If this host disappears, point the URL at any
#               mirror - a wrong tarball fails here instead of shipping.
build_libaio() {
    curl -fsSL -o libaio-0.3.113.tar.gz https://deb.debian.org/debian/pool/main/liba/libaio/libaio_0.3.113.orig.tar.gz
    echo "2c44d1c5fd0d43752287c9ae1eb9c023f04ef848ea8d4aafa46e9aedb678200b  libaio-0.3.113.tar.gz" | sha256sum -c -
    tar zxf libaio-0.3.113.tar.gz
    cd libaio-0.3.113

    make -C src CFLAGS="-fPIC -O2 -I." > /dev/null
    mkdir -p /opt/${arch}-aio/lib /opt/${arch}-aio/include
    cp src/libaio.a /opt/${arch}-aio/lib/
    cp src/libaio.h /opt/${arch}-aio/include/
    make -C src clean > /dev/null

    cd -
    rm -rf libaio-0.3.113 libaio-0.3.113.tar.gz
}

build_liburing
build_libaio
