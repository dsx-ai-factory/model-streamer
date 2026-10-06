#!/usr/bin/env bash
# shellcheck source=.devcontainer/native-common.sh
source /opt/build/native-common.sh

fetch https://github.com/axboe/liburing/archive/refs/tags/liburing-2.14.tar.gz
cd liburing-liburing-2.14
./configure --use-libc --prefix="${prefix}-uring" --libdir="${prefix}-uring/lib"
make -C src -j"$jobs" ENABLE_SHARED=0 CFLAGS='-fPIC -O2'
make install ENABLE_SHARED=0
cd "$work"

fetch https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz
cd libaio-0.3.113
make -C src CFLAGS='-fPIC -O2 -I.'
mkdir -p "${prefix}-aio/lib" "${prefix}-aio/include"
cp src/libaio.a "${prefix}-aio/lib/"
cp src/libaio.h "${prefix}-aio/include/"
cd /
