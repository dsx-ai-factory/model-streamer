# Native library sources and rebuilding

The standard build statically links libaio and liburing into `libstreamer.so`.
liburing provides io_uring reads; libaio provides asynchronous direct-I/O reads.
The S3, GCS and Azure plugins do not directly link either library. Installing a
replacement system `libaio.so` or `liburing.so` does not change the code in the
core library: rebuild and replace `libstreamer.so` to use a modified dependency.

libaio 0.3.113 is LGPL-2.1-or-later. liburing 2.14 uses its MIT licensing option.
Each deliverable's `NOTICE` (`py/runai_model_streamer*/NOTICE`, `cpp/NOTICE`) records versions
and licenses; the `# attribution:` comments in `cpp/MODULE.bazel` (and, for the libraries
the development container builds, `.devcontainer/Dockerfile`) say which library each
component is linked into. The binary deliverables contain shared libraries, public
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

Bazel builds both libraries from source as modules of the
[Bazel Central Registry](https://registry.bazel.build) (BCR). That checkout's
[`cpp/MODULE.bazel`](../cpp/MODULE.bazel) pins their versions (`bazel_dep`) and names
any patch applied on top of the registry's (`single_version_override`). For each
module version, the registry's `source.json`
(`https://bcr.bazel.build/modules/<name>/<version>/source.json`) gives the upstream
archive URL and its checksum, and its `overlay/` directory the `BUILD.bazel` that
compiles it. [`cpp/bazel_downloader.cfg`](../cpp/bazel_downloader.cfg) lists any
archive Bazel downloads from another mirror than the registry names.
Older releases built these libraries in the development container instead; for
them, use the versions, URLs, checksums and flags in that checkout's
`.devcontainer/scripts/build-io-libs.sh` and `.devcontainer/Dockerfile`.
Download and retain the complete dependency archives, not just their license
texts from the `NOTICE` files. The commands below match this
revision's pins; use the matching recipe when rebuilding another revision.

## Rebuild with modified libaio

Open the checked-out project in its development container as described in the
[development instructions](../README.md#development). Build natively on Linux
x86_64 or aarch64; use the architecture of the binary you are replacing. The
container supplies the compiler, Bazel, headers and other dependencies. Its
Dockerfile and scripts are included in the project source so the environment
can also be built from source. The following commands run **inside that container**.

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
```

Then let Bazel fetch the module and copy its source tree out of Bazel's cache.
That tree is the archive above with the registry's `BUILD.bazel` and `MODULE.bazel`
added, plus any patch `cpp/MODULE.bazel` applies: the exact source the standard
build compiles. (`libaio~` is the directory name Bazel 7 uses; Bazel 8 names it
`libaio+`.)

```sh
cd "$STREAMER_SOURCE_DIR/cpp"
bazel fetch @libaio//:aio
cp -R "$(bazel info output_base)/external/libaio~" "$AIO_REBUILD_DIR/libaio"
```

Make your changes under `$AIO_REBUILD_DIR/libaio`. Keep a copy of the patch
and preserve the upstream license and copyright notices. Then build with Bazel
using your copy in place of the registry's. `--override_module` has the effect of a
`local_path_override` in `MODULE.bazel`, and the module is still compiled into
`libstreamer.so` statically, with position-independent code and `-O2`:

```sh
cd "$STREAMER_SOURCE_DIR/cpp"
bazel build streamer:libstreamer.so --config="$ARCH" \
  --override_module=libaio="$AIO_REBUILD_DIR/libaio"
```

Bazel tracks the copied sources as build inputs, so a later change to them is
rebuilt without clearing any cache. Pass the same `--override_module` to every
later build and test command, or Bazel builds the registry's source again.

The resulting library is `cpp/bazel-bin/streamer/libstreamer.so`. Retain a linker
map to check that the archive supplied code, and inspect ELF dependencies to
confirm it was not linked dynamically. Run this link action locally so Bazel's
sandbox cannot hide the temporary directory receiving the map:

```sh
cd "$STREAMER_SOURCE_DIR/cpp"
bazel build streamer:libstreamer.so --config="$ARCH" \
  --override_module=libaio="$AIO_REBUILD_DIR/libaio" \
  --strategy=CppLink=local --linkopt="-Wl,-Map=$AIO_REBUILD_DIR/libstreamer.map"
grep 'libaio.a' "$AIO_REBUILD_DIR/libstreamer.map"
readelf -d bazel-bin/streamer/libstreamer.so | grep NEEDED
```

Run your filesystem workload with `RUNAI_STREAMER_FS_STRATEGY=libaio_direct` to
exercise libaio without falling back to another strategy. The host and filesystem
must support direct I/O. Test the modified library before using it in production.
For the project's native engine tests, run:

```sh
cd "$STREAMER_SOURCE_DIR/cpp"
bazel test \
  //posix_io/libaio_probe:libaio_probe_test \
  //posix_io/libaio_engine:libaio_engine_test --config="$ARCH" --test_output=errors \
  --override_module=libaio="$AIO_REBUILD_DIR/libaio"
```

To keep the change in your own checkout instead of passing the flag, add
`local_path_override(module_name = "libaio", path = "<your copy>")` to
`cpp/MODULE.bazel`. To build from a modified archive you host, use
`archive_override` with its URL, `integrity` and `strip_prefix`; that archive must
then contain the `BUILD.bazel` and `MODULE.bazel` above, which the registry adds
only to its own download.

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
that you obtained its matching source tag or commit. The wheel packages
`cpp/bazel-bin/streamer/libstreamer.so` as it was last built, so make the
`--override_module` build above the last Bazel build before this step. The core wheel pins its
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

To change liburing too, obtain the complete pinned archive the registry names
and verify its checksum before editing. For this revision:

```sh
set -euo pipefail
cd "$AIO_REBUILD_DIR"
curl -fsSL --connect-timeout 15 --max-time 60 -o liburing-2.14.tar.gz \
  https://github.com/axboe/liburing/archive/refs/tags/liburing-2.14.tar.gz || exit 1
echo "5f80964108981c6ad979c735f0b4877d5f49914c2a062f8e88282f26bf61de0c  liburing-2.14.tar.gz" | sha256sum -c - || exit 1
```

Copy the source tree Bazel builds, as for libaio. It already carries the
registry's `BUILD.bazel`, which runs liburing's `configure --use-libc` during the
build, and the patch from `cpp/third_party` that `cpp/MODULE.bazel` applies.
Make your source changes under `$AIO_REBUILD_DIR/liburing`, then rebuild:

```sh
cd "$STREAMER_SOURCE_DIR/cpp"
bazel fetch @liburing//:uring
cp -R "$(bazel info output_base)/external/liburing~" "$AIO_REBUILD_DIR/liburing"
# Make your source changes before building.
bazel build streamer:libstreamer.so --config="$ARCH" \
  --override_module=liburing="$AIO_REBUILD_DIR/liburing"
```

To rebuild with both libraries modified, pass both `--override_module` flags.
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
  the core with a modified library: `cpp/MODULE.bazel`, the registry's
  `source.json` and `BUILD.bazel` overlay for each module version, and the patches
  under `cpp/third_party`.

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
