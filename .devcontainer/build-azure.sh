#!/usr/bin/env bash
# shellcheck source=.devcontainer/native-common.sh
source /opt/build/native-common.sh

# azure-storage-blobs 12.15.0
checkout azure https://github.com/Azure/azure-sdk-for-cpp 9ada70c530db9dc08c2b15c9c9886641b7a06504
export AZURE_SDK_DISABLE_AUTO_VCPKG=1
for component in core/azure-core storage/azure-storage-common storage/azure-storage-blobs identity/azure-identity; do
    cmake -S "azure/sdk/$component" -B "azure-build/${component##*/}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_INSTALL_PREFIX="${prefix}-azure" \
        -DCMAKE_PREFIX_PATH="${prefix}-ssl;${prefix}-curl;${prefix}-xml2;${prefix}-azure" \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_TRANSPORT_CURL=ON \
        -DDISABLE_AZURE_CORE_OPENTELEMETRY=ON \
        -DCURL_LIBRARY="${prefix}-curl/lib/libcurl.a" -DCURL_INCLUDE_DIR="${prefix}-curl/include" \
        -DOPENSSL_ROOT_DIR="${prefix}-ssl" -DOPENSSL_USE_STATIC_LIBS=TRUE \
        -DLIBXML2_LIBRARY="${prefix}-xml2/lib/libxml2.a" -DLIBXML2_INCLUDE_DIR="${prefix}-xml2/include/libxml2"
    cmake --build "azure-build/${component##*/}" -j"$jobs"
    cmake --install "azure-build/${component##*/}"
done

