# Development and release dependencies

The two PyPA `manylinux_2_28` base images are pinned by immutable digest in the
Dockerfile. They provide AlmaLinux 8.10, GCC 14.2.1, Python 3.12.15 and the native
build utilities. AlmaLinux 8 receives security support through June 2029; Python
3.12 receives upstream security fixes through October 2028. The default build
interpreter is 3.12; CI also exercises 3.13 and 3.14 from the same pinned images.

A Debian-based official Python image would raise the native libraries' glibc
requirements. The manylinux base builds against glibc 2.28, preserving support for all
previously supported glibc 2.30+ systems. The wheels use `manylinux_2_28` tags
so they can also be installed and tested inside the build image itself.
Native runners build both x86_64 and aarch64; `make build` builds the host
architecture. `.devcontainer/scripts/check-wheel-compatibility.py` checks every ELF in
each wheel for the correct architecture and GLIBC symbol versions at most 2.28.
Changing a filename alone does not make a binary compatible with an older libc.

## Layout

- `dependencies/`: Python and RPM pins, download checksums, and Azurite's npm lockfiles.
- `scripts/`: image setup, native dependency builds, and CI validation scripts.
- `cmake/`: existing CMake toolchain definitions.

`Dockerfile` and `devcontainer.json` remain at the root. Build scripts and pins
are copied into `/opt/build` inside the image.

Docker, the package Makefiles, and Python-version CI tests all install
`dependencies/requirements.lock` directly. Edit `dependencies/requirements.in`
and regenerate that lock to update the shared Linux build/test environment.
It includes CPU PyTorch; consumer dependencies remain in each package's `setup.py`.

## Pins

| Dependency | Pin mechanism |
| --- | --- |
| OS, Python interpreters, GCC, CMake and base utilities | Architecture-specific image digests |
| Additional Perl and ThreadSanitizer packages | Full RPM versions, including the install transaction's dependencies, in `dependencies/rpm-packages.lock` |
| Python build/test packages, cloud clients and CPU PyTorch | Direct versions in `dependencies/requirements.in`; transitive versions and hashes in `dependencies/requirements.lock` |
| Node.js 24.21.0, Bazel 7.6.1, SeaweedFS 4.47, fake-gcs-server 1.52.2 | Versioned downloads and `dependencies/downloads.sha256` |
| CUDA 12.8 driver headers | NVIDIA cudart 12.8.57 archive and published SHA-256; no CUDA runtime is copied |
| Azurite 3.37.0 | `dependencies/azurite/package.json` and complete `package-lock.json`; installed using `npm ci` |
| OpenSSL 3.5.9 LTS, curl 8.22.0, zlib 1.3.2, libxml2 2.15.4, liburing 2.14, libaio 0.3.113 | Versioned downloads and `dependencies/downloads.sha256` |
| AWS C++ SDK 1.11.584 and Azure Storage Blobs C++ SDK 12.15.0 | Full git commit IDs and recursive submodule gitlinks in `scripts/build-aws.sh` and `scripts/build-azure.sh` |
| Google Cloud C++ SDK 2.37.0 and Bazel dependencies | Existing versioned, checksummed archives in `cpp/third_party/gcp_repo.bzl`, `cpp/toolchain/deps.bzl`, and `cpp/rules.bzl`; GCP's transitive pins come from its pinned release |
| GitHub Actions | Release commit SHAs in workflow files |

The Ubuntu 20.04 hand-written ThreadSanitizer preinit workaround is replaced by
the matching GCC toolset's sanitizer development package. It supplies the actual
compiler runtime and preinit object. Sanitizer execution still depends on the
host kernel and the existing ASLR settings in `cpp/.bazelrc`.

OpenSSL stays on its supported 3.5 LTS line. libxml2 moves from the old 2.13 line
to 2.15. The native cloud SDK versions and libaio have no lifecycle guarantee
established here; their exact pins are retained for API compatibility and should
be reviewed against upstream advisories when refreshed. Azurite's locked tree
still contains upstream-deprecated packages (including uuid and dottie); these
are emulator-only dependencies, and replacing them requires an upstream Azurite
update rather than arbitrary transitive overrides.

## Updating

1. Choose supported versions and resolve image digests for both architectures.
   Update RPM pins against those images together; do not add floating OS installs.
2. Change direct Python versions in `dependencies/requirements.in`, then regenerate the lock:

   ```sh
   uv pip compile .devcontainer/dependencies/requirements.in --python-version 3.12 \
     --python-platform x86_64-manylinux_2_28 --torch-backend cpu \
     --generate-hashes --no-annotate --emit-index-url \
     -o .devcontainer/dependencies/requirements.lock
   ```

   Keep `--extra-index-url https://download.pytorch.org/whl/cpu` in the generated
   lock for pip. The resolver selects CPU PyTorch explicitly; some uv versions do
   not emit that index when `--torch-backend` is used. Validate the lock on both
   architectures and all tested Python versions with `--require-hashes` and
   `--only-binary=:all:`. Do not change published package requirements to match
   development pins.
3. Update `dependencies/azurite/package.json`, then run `npm install --package-lock-only
   --ignore-scripts` in that directory. Commit both files.
4. Update native tool/source versions and their checksums together. Use upstream
   release checksums when available and preserve CUDA's license with its headers.
5. Build both containers and run `make test`, the Python version checks, and the
   wheel compatibility check. CI must pass before releasing the resulting wheels.

The image sets `PIP_CONSTRAINT` to a version-only copy of the Python lock so
subsequent project-wheel installs cannot silently upgrade the environment.
Installing the environment itself uses the hash-checked lock. No constraints
are embedded into the published packages.

Sources: [PyPA manylinux](https://github.com/pypa/manylinux),
[AlmaLinux lifecycle](https://wiki.almalinux.org/release-notes/8.10.html),
[Python lifecycle](https://devguide.python.org/versions/),
[NVIDIA redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json).
