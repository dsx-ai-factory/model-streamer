# Model Streamer C/C++ SDK

The C/C++ SDK is published alongside the Python SDK wheels on GitHub Releases, with the
same version and source tag. It exposes the C API from `streamer/streamer.h`,
which can also be used from C++17. No Python package or Bazel installation is
needed to consume it.

## Download and build an example

Choose a published SDK version and run on Linux x86_64 or aarch64:

```bash
VERSION=0.17.0
ARCH=$(uname -m)
PACKAGE=runai-model-streamer-sdk-${VERSION}-linux-${ARCH}.tar.gz
RELEASE=https://github.com/dsx-ai-factory/model-streamer/releases/download/v${VERSION}
curl -fsSLO "${RELEASE}/${PACKAGE}"
curl -fsSLO "${RELEASE}/${PACKAGE}.sha256"
sha256sum -c "${PACKAGE}.sha256"

PREFIX="$HOME/.local/runai-model-streamer-${VERSION}"
mkdir -p "$PREFIX"
tar -xzf "$PACKAGE" -C "$PREFIX" --strip-components=1
cmake -S "$PREFIX/examples/hello_streamer" -B build -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build build
printf 'hello streamer' > input.txt
./build/hello_streamer input.txt 'hello streamer'
```

Use the release's actual tag in the URL if it does not have a `v` prefix. The
archive filename and its `VERSION` file always omit that prefix.

CMake consumers use `find_package(runai-streamer CONFIG REQUIRED)` and link
`runai::streamer`. A version file supports exact version requests. For pkg-config:

```bash
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cc -std=c99 -Wall -Wextra -Werror \
  "$PREFIX/examples/hello_streamer/hello_streamer.c" \
  $(pkg-config --cflags --libs runai-streamer) \
  -Wl,-rpath,"$PREFIX/lib" -o hello_streamer
./hello_streamer input.txt 'hello streamer'
```

An application needs a runtime search path to the SDK's `lib` directory. CMake
sets a build-tree runtime path for the example; the pkg-config command above
sets one explicitly. For a system install, unpack into `/usr/local` and run
`sudo ldconfig`, ensuring `/usr/local/lib` is in the loader configuration. Plain
`ldconfig` does not automatically discover arbitrary user prefixes.

## Contents and runtime requirements

- `include/streamer/`: the public API, submission IDs, response codes, and device types.
- `lib/libstreamer.so`: filesystem streaming and the object-storage plugin loader.
- `lib/libstreamers3.so`, `lib/libstreamergcs.so`, `lib/libstreamerazure.so`: the cloud backends.
- `lib/pkgconfig/` and `lib/cmake/`: relocatable build-system discovery files.
- `examples/hello_streamer/`, `VERSION`, and `LICENSE`.

Keep all four libraries together: the core locates plugins through `$ORIGIN`.
Plugins are loaded on demand. Their credentials and endpoint environment
variables are documented in the repository's
[environment variable reference](https://github.com/dsx-ai-factory/model-streamer/blob/master/docs/src/env-vars.md).
TLS connections require a system CA certificate bundle.

The public header documents configuration and response handling. Calls that
submit requests and receive responses must not run in parallel on the same
streamer. Response buffers must remain valid until their reads finish.

CI validates the archives on Ubuntu 20.04 userspace on both architectures. This
does not establish a glibc 2.17/manylinux2014 compatibility guarantee; testing an
older userspace and freezing a minimum glibc baseline remain follow-up work.
The ARM build currently requires ARMv8 CRC and crypto extensions.

## Building and release validation

`PACKAGE_VERSION=0.17.0 make build` builds the wheels and stages each architecture's
SDK before Bazel switches to the next architecture. Archives and GNU-compatible
checksum files are written to `sdk/dist/`. Packaging rejects missing libraries,
the wrong ELF architecture, and a core library without `$ORIGIN` in its runpath.

The top-level Makefile provides the same test entry points for developers and CI:

```bash
make test-python                         # Python SDK unit tests
PACKAGE_VERSION=0.17.0 make test-cpp     # Build and test the C/C++ SDK
make test-cpp CPP_SDK_ARCHIVE=/path/to/runai-model-streamer-sdk-0.17.0-linux-x86_64.tar.gz
PACKAGE_VERSION=0.17.0 make test         # Full suite, including both SDKs
```

Run C/C++ SDK tests on Linux with Python 3, gcc/g++, CMake, make, pkg-config, binutils,
and the runtime dependencies installed. Building the SDK additionally requires
the repository's Bazel toolchain; `CPP_SDK_ARCHIVE` skips that build and tests the
supplied archive without installing any streamer Python packages.

`make test-cpp` runs packaging unit tests and consumer checks: checksum
verification, C99 and C++17 compilation via pkg-config, the shipped CMake example,
file-byte verification, and runtime dependency checks. Consumers build against
an unpacked temporary prefix, outside the repository.

PR CI calls `make test` once with the PR version and tests the C/C++ SDK in the
devcontainer.
Release CI builds each architecture on a separate runner, then calls
`make test-cpp CPP_SDK_ARCHIVE=...` in clean Ubuntu 20.04 containers on native x86_64
and aarch64 runners. GitHub asset publishing and the four PyPI package uploads
run in parallel after both consumers pass, using the tested artifacts without
rebuilding them.

Object-storage end-to-end C/C++ SDK tests and an exported-symbol compatibility baseline
are follow-up work tracked by issue #179; this publishing change does not close
all of that issue's acceptance criteria.
