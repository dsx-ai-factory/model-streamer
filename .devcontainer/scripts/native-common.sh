#!/usr/bin/env bash
# Shared variables are used by the scripts sourcing this file.
# shellcheck disable=SC2034
set -euo pipefail

# All archives are versioned and checked against downloads.sha256. SDK git
# checkouts use full commits; recursive submodules use the recorded gitlinks.
arch=$(uname -m)
prefix=/opt/${arch}
jobs=${BUILD_JOBS:-4}
work=$(mktemp -d)
cd "$work"

fetch() {
    local url=$1 name=${1##*/}
    curl --fail --location --retry 3 --output "$name" "$url"
    grep "  ${name}$" /opt/build/downloads.sha256 | sha256sum --check -
    tar xf "$name"
}

checkout() {
    git init "$1"
    git -C "$1" remote add origin "$2"
    git -C "$1" fetch --depth 1 origin "$3"
    git -C "$1" checkout --detach FETCH_HEAD
    git -C "$1" submodule update --init --recursive --depth 1
}

trap 'rm -rf "$work"' EXIT
