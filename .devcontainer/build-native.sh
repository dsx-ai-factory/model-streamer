#!/usr/bin/env bash
# shellcheck source=.devcontainer/native-common.sh
source /opt/build/native-common.sh

fetch https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz
cd openssl-3.5.9
./Configure no-shared no-zlib no-comp no-dynamic-engine -fPIC \
    --prefix="${prefix}-ssl" --openssldir="${prefix}-ssl" --libdir=lib
make -j"$jobs"
make install_sw
cd "$work"

fetch https://zlib.net/zlib-1.3.2.tar.gz
cd zlib-1.3.2
CFLAGS='-fPIC -O2' ./configure --static --prefix="${prefix}-zlib"
make -j"$jobs"
make install
cd "$work"

fetch https://curl.se/download/curl-8.22.0.tar.gz
cd curl-8.22.0
CPPFLAGS="-I${prefix}-ssl/include" LDFLAGS="-L${prefix}-ssl/lib" CFLAGS='-fPIC -O2' \
    ./configure --disable-shared --without-libpsl --without-brotli --without-zstd \
    --without-libidn2 --disable-ldap --disable-ldaps \
    --with-openssl="${prefix}-ssl" --with-zlib="${prefix}-zlib" --prefix="${prefix}-curl"
make -j"$jobs"
make install
cd "$work"

fetch https://download.gnome.org/sources/libxml2/2.15/libxml2-2.15.4.tar.xz
cd libxml2-2.15.4
CFLAGS='-fPIC -O2' ./configure --disable-shared --enable-static \
    --without-python --without-lzma --without-zlib --without-iconv \
    --prefix="${prefix}-xml2"
make -j"$jobs"
make install
cd "$work"

