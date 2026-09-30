#!/usr/bin/env bash
set -euo pipefail

archive=$(realpath "${1:?Usage: bash sdk/verify.sh ARCHIVE}")
arch=$(uname -m)
[[ "$archive" == *"-linux-${arch}.tar.gz" ]]
cd "$(dirname "$archive")"
sha256sum -c "$(basename "$archive").sha256"

# Deliberately outside the checkout, with no Bazel output or Python package on
# the search path. Every run uses a different prefix to catch baked-in paths.
work=$(mktemp -d)
prefix="$work/relocated-sdk"
mkdir -p "$prefix"
tar -xzf "$archive" -C "$prefix" --strip-components=1
unset LD_LIBRARY_PATH
export PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"
unset PKG_CONFIG_PATH
test "$(pkg-config --modversion runai-streamer)" = "$(cat "$prefix/VERSION")"

printf 'sdk consumer test' > "$work/input.txt"
source="$prefix/examples/hello_streamer/hello_streamer.c"
for language in c c++; do
    if [[ "$language" == c ]]; then compiler=cc; standard=c99; else compiler=c++; standard=c++17; fi
    # pkg-config output intentionally expands to separate compiler arguments.
    # shellcheck disable=SC2046
    "$compiler" -x "$language" -std="$standard" -Wall -Wextra -Werror "$source" \
        $(pkg-config --cflags --libs runai-streamer) \
        -Wl,-rpath,"$prefix/lib" -o "$work/hello-$language"
    "$work/hello-$language" "$work/input.txt" 'sdk consumer test'
done
cmake -S "$prefix/examples/hello_streamer" -B "$work/build" -DCMAKE_PREFIX_PATH="$prefix"
cmake --build "$work/build"
"$work/build/hello_streamer" "$work/input.txt" 'sdk consumer test'

for library in "$prefix"/lib/*.so; do
    dependencies=$(ldd "$library")
    printf '%s\n' "$dependencies"
    if [[ "$dependencies" == *'not found'* ]]; then exit 1; fi
done
readelf -dW "$prefix/lib/libstreamer.so" | grep -E '\((RUNPATH|RPATH)\).*\[\$ORIGIN\]'
echo "C/C++ SDK consumer checks passed for $arch"
