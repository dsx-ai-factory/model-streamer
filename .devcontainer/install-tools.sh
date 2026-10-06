#!/usr/bin/env bash
set -euo pipefail
case $(uname -m) in
    x86_64) goarch=amd64; nodearch=x64; bazelarch=x86_64; cudaarch=x86_64 ;;
    aarch64) goarch=arm64; nodearch=arm64; bazelarch=arm64; cudaarch=sbsa ;;
    *) exit 1 ;;
esac
work=$(mktemp -d)
cd "$work"
fetch() {
    local name=${1##*/}
    curl --fail --location --retry 3 --output "$name" "$1"
    grep "  ${name}$" /opt/build/downloads.sha256 | sha256sum --check -
}
# Only headers are needed; libcuda is loaded from the host driver at runtime.
fetch "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-${cudaarch}/cuda_cudart-linux-${cudaarch}-12.8.57-archive.tar.xz"
tar xf "cuda_cudart-linux-${cudaarch}-12.8.57-archive.tar.xz"
mkdir -p /usr/local/cuda
cp -a "cuda_cudart-linux-${cudaarch}-12.8.57-archive/include" /usr/local/cuda/
cp "cuda_cudart-linux-${cudaarch}-12.8.57-archive/LICENSE" /usr/local/cuda/LICENSE
fetch "https://github.com/bazelbuild/bazel/releases/download/7.6.1/bazel-7.6.1-linux-${bazelarch}"
install -m 755 "bazel-7.6.1-linux-${bazelarch}" /usr/local/bin/bazel
fetch "https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-${nodearch}.tar.xz"
tar xf "node-v24.21.0-linux-${nodearch}.tar.xz" --strip-components=1 -C /usr/local
fetch "https://github.com/seaweedfs/seaweedfs/releases/download/4.47/linux_${goarch}.tar.gz"
tar xf "linux_${goarch}.tar.gz" weed
install -m 755 weed /usr/local/bin/weed
fetch "https://github.com/fsouza/fake-gcs-server/releases/download/v1.52.2/fake-gcs-server_1.52.2_Linux_${goarch}.tar.gz"
tar xf "fake-gcs-server_1.52.2_Linux_${goarch}.tar.gz" fake-gcs-server
install -m 755 fake-gcs-server /usr/local/bin/fake-gcs-server
cd /
rm -rf "$work"
