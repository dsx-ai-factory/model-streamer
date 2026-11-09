# Native library sources and rebuilding

The standard build statically links libaio and liburing into `libstreamer.so`.
liburing provides io_uring reads; libaio provides asynchronous direct-I/O reads.
The S3, GCS and Azure plugins do not directly link either library. Installing a
replacement system `libaio.so` or `liburing.so` does not change the code in the
core library: rebuild and replace `libstreamer.so` to use a modified dependency.

libaio 0.3.113 is LGPL-2.1-or-later. liburing 2.14 uses its MIT licensing option.
The [component manifest](../third_party_notices/components.json) records versions,
licenses and linkage. The binary deliverables contain shared libraries, public
SDK headers and notices; they do not include dependency source archives or a
separate set of relinkable object files. The replacement mechanism described here
is a full rebuild from source.

## Obtain matching sources

Start with the Git tag or commit used to produce your installed binary, rather
than the moving `master` branch. Release tags have no `v` prefix. The repository's
GitHub release page provides its tagged source as a tar or zip archive; cloning
and checking out that tag gives the same project source and build scripts:

```sh
git clone https://github.com/dsx-ai-factory/model-streamer.git
cd model-streamer
STREAMER_REF=X.Y.Z  # Replace with your release tag or exact commit.
git checkout "$STREAMER_REF"
```

Use the dependency versions, download URLs, checksums and build flags in that
checkout's [I/O dependency build script](../.devcontainer/scripts/build-io-libs.sh).
For older releases the recipe may be inline in `.devcontainer/Dockerfile`.
Download and retain the complete dependency archives, not just their license
texts from `third_party_notices/licenses/`. The commands below match this
revision's pins; use the matching recipe when rebuilding another revision.

## Rebuild with modified libaio

Open the checked-out project in its development container as described in the
[development instructions](../README.md#development). Build natively on Linux
x86_64 or aarch64; use the architecture of the binary you are replacing. The
container supplies the compiler, Bazel, headers and other dependencies. Its
Dockerfile and scripts are included in the project source so the environment
can also be built from source. The following commands run **inside that container**.
They change its private `/opt` dependency installation; do not share that
installation with a production application.

First download and verify the original source. `mktemp` gives each rebuild its
own directory, and the original archive remains available after modification.
Run the snippets in the same Bash session. Error exits are enabled below: stop
and resolve any failure before continuing with another block.

```sh
set -euo pipefail
STREAMER_SOURCE_DIR=$PWD
AIO_REBUILD_DIR=$(mktemp -d)
ARCH=$(uname -m)
cd "$AIO_REBUILD_DIR"
curl -fsSL --connect-timeout 15 --max-time 60 -o libaio-0.3.113.tar.gz \
  https://deb.debian.org/debian/pool/main/liba/libaio/libaio_0.3.113.orig.tar.gz || exit 1
echo "2c44d1c5fd0d43752287c9ae1eb9c023f04ef848ea8d4aafa46e9aedb678200b  libaio-0.3.113.tar.gz" | sha256sum -c - || exit 1
tar -xzf libaio-0.3.113.tar.gz
```

Make your changes under `$AIO_REBUILD_DIR/libaio-0.3.113`. Keep a copy of the patch
and preserve the upstream license and copyright notices. Then compile the static
archive with position-independent code and replace the container's build input:

```sh
make -C "$AIO_REBUILD_DIR/libaio-0.3.113/src" -B \
  ENABLE_SHARED=0 CFLAGS="-fPIC -O2 -I."
cp "$AIO_REBUILD_DIR/libaio-0.3.113/src/libaio.a" "/opt/$ARCH-aio/lib/libaio.a"
cp "$AIO_REBUILD_DIR/libaio-0.3.113/src/libaio.h" "/opt/$ARCH-aio/include/libaio.h"
STREAMER_BAZEL_OUTPUT=$(mktemp -d)
cd "$STREAMER_SOURCE_DIR/cpp"
bazel --output_base="$STREAMER_BAZEL_OUTPUT" build streamer:libstreamer.so \
  --config="$ARCH" --noremote_accept_cached --disk_cache=
```

The archives are passed through linker flags, so Bazel does not track their
contents as inputs. Replacing an archive alone can leave the old core or test
binary cached. **Create a new `STREAMER_BAZEL_OUTPUT` after every dependency
change**, as above, and use it for the subsequent builds and tests. These commands
also disable remote cache reads and the disk cache to prevent reuse of stale
link results from another output directory.

The resulting library is `cpp/bazel-bin/streamer/libstreamer.so`. Retain a linker
map to check that the archive supplied code, and inspect ELF dependencies to
confirm it was not linked dynamically:

```sh
cd "$STREAMER_SOURCE_DIR/cpp"
bazel --output_base="$STREAMER_BAZEL_OUTPUT" build streamer:libstreamer.so \
  --config="$ARCH" --noremote_accept_cached --disk_cache= \
  --linkopt="-Wl,-Map=$AIO_REBUILD_DIR/libstreamer.map"
grep 'libaio.a' "$AIO_REBUILD_DIR/libstreamer.map"
readelf -d bazel-bin/streamer/libstreamer.so | grep NEEDED
```

Run your filesystem workload with `RUNAI_STREAMER_FS_STRATEGY=libaio_direct` to
exercise libaio without falling back to another strategy. The host and filesystem
must support direct I/O. Test the modified library before using it in production.
For the project's native engine tests, run:

```sh
cd "$STREAMER_SOURCE_DIR/cpp"
bazel --output_base="$STREAMER_BAZEL_OUTPUT" test \
  //posix_io/libaio_probe:libaio_probe_test \
  //posix_io/libaio_engine:libaio_engine_test --config="$ARCH" --test_output=errors \
  --noremote_accept_cached --disk_cache=
```

## Use the rebuilt library

For a Python installation of the same Streamer revision, set `STREAMER_LIBRARY`
to the absolute path of the rebuilt `libstreamer.so` before starting Python.
The override is read when the package is imported. If you use cloud backends,
keep their matching plugin libraries beside the rebuilt core library so its
`$ORIGIN` plugin lookup can find them.

Alternatively, build and install a replacement core wheel. First determine the
installed version in your **application's Python environment**:

```sh
python3 -c 'from importlib.metadata import version; print(version("runai-model-streamer"))'
```

In the build container, set `PACKAGE_VERSION` to that exact version and check
that you obtained its matching source tag or commit. The core wheel pins its
cloud extras to this version, so do not replace it with `0.0.0`.

```sh
cd "$STREAMER_SOURCE_DIR"
PACKAGE_VERSION=X.Y.Z  # Replace with the installed version reported above.
PACKAGE_VERSION="$PACKAGE_VERSION" make -C py/runai_model_streamer build
```

Back in the application's Python environment, from the project root, install the
wheel for that version and architecture. Force-reinstall it because the version
has not changed. `--no-deps` preserves the application's existing dependencies:

```sh
PACKAGE_VERSION=X.Y.Z  # Use the same installed version.
python3 -m pip install --force-reinstall --no-deps \
  py/runai_model_streamer/dist/runai_model_streamer-"$PACKAGE_VERSION"-py3-none-*_"$(uname -m)".whl
```

For a native SDK installation, copy the rebuilt library into a separate SDK
prefix containing the matching headers and plugins, then point your application's
library search path at that prefix. Preserve the original installation for rollback.

## Rebuild liburing

To change liburing too, obtain the complete pinned archive from the I/O build
script and verify its checksum before editing it. For this revision:

```sh
set -euo pipefail
cd "$AIO_REBUILD_DIR"
curl -fsSL --connect-timeout 15 --max-time 60 -o liburing-2.14.tar.gz \
  https://codeload.github.com/axboe/liburing/tar.gz/refs/tags/liburing-2.14 || exit 1
echo "5f80964108981c6ad979c735f0b4877d5f49914c2a062f8e88282f26bf61de0c  liburing-2.14.tar.gz" | sha256sum -c - || exit 1
tar -xzf liburing-2.14.tar.gz
cd liburing-liburing-2.14
# Make your source changes before configuring and compiling.
./configure --use-libc --prefix="/opt/$ARCH-uring" --libdir="/opt/$ARCH-uring/lib"
make -C src -j4 ENABLE_SHARED=0 CFLAGS="-fPIC -O2"
make install ENABLE_SHARED=0
STREAMER_BAZEL_OUTPUT=$(mktemp -d)
cd "$STREAMER_SOURCE_DIR/cpp"
bazel --output_base="$STREAMER_BAZEL_OUTPUT" build streamer:libstreamer.so \
  --config="$ARCH" --noremote_accept_cached --disk_cache=
```

After changing liburing, repeat [Use the rebuilt library](#use-the-rebuilt-library):
rebuild and force-reinstall the wheel, or replace the native library again.
Rebuilding the Bazel library does not update an already installed wheel or SDK.

Use `RUNAI_STREAMER_FS_STRATEGY=io_uring_direct` or `io_uring_buffered` when testing
that path. The host kernel and container security configuration must permit
io_uring; see the development container's `runArgs`.

## Release source delivery

Before distributing a binary release, maintainers must provide recipients with:

- The exact Streamer source tag or commit, including build scripts and this guide.
- The complete corresponding libaio source archive and any patches used in the
  binary, with its license and notices. A license-text download alone is not source.
- The dependency versions, checksums and build instructions needed to rebuild
  the core with a modified library.

Publish those source materials alongside the downloadable binaries and link them
from the release notes. Check that recipients can download them without employee
credentials. Retain the source archives used for the release so the delivery does
not depend solely on upstream URLs remaining available. Include any other source
materials required by the licenses of statically linked components listed in the
component manifest.

The current bundle automation assembles wheels and SDK binaries with notices;
it does not upload a corresponding-source bundle. Preparing and publishing the
source materials above is therefore a separate release step. Confirm their
availability and the recipient rebuild procedure before marking that step complete.
