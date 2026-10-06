#!/usr/bin/env bash
# shellcheck source=.devcontainer/scripts/native-common.sh
source /opt/build/native-common.sh

# aws-sdk-cpp 1.11.584
checkout aws https://github.com/aws/aws-sdk-cpp bba3cfc14d4fc148aeee7a8ff7822dd7a9a0f4d3
cmake -S aws -B aws-build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="${prefix}-aws" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_PREFIX_PATH="${prefix}-ssl;${prefix}-curl;${prefix}-zlib" \
    -DBUILD_ONLY=s3-crt -DBUILD_SHARED_LIBS=OFF -DFORCE_SHARED_CRT=OFF \
    -DENABLE_TESTING=OFF -DAUTORUN_UNIT_TESTS=OFF \
    -DZLIB_LIBRARY="${prefix}-zlib/lib/libz.a" -DZLIB_INCLUDE_DIR="${prefix}-zlib/include" \
    -DCURL_LIBRARY="${prefix}-curl/lib/libcurl.a" -DCURL_INCLUDE_DIR="${prefix}-curl/include" \
    -DOPENSSL_ROOT_DIR="${prefix}-ssl" -DOPENSSL_USE_STATIC_LIBS=TRUE
cmake --build aws-build -j"$jobs"
cmake --install aws-build

