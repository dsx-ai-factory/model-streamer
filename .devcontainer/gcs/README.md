# Prebuilt Google Cloud C++ SDK

The devcontainer builds the pinned `storage` and `storage_grpc` SDK targets,
including their transitive dependencies, and installs static archives and headers
under `/opt/gcs-sdk/<architecture>`. Normal builds compile our GCS adapter and
tests against that installation. REST and gRPC remain enabled.

Docker's `TARGETARCH` selects the installed SDK (`amd64` or `arm64`). Both CI
and local images build only their native architecture using the native toolchain.
The Bazel repository exposes only installed architectures and reports an
unsupported target clearly.

The build entry point is `.devcontainer/scripts/build-gcs-sdk.sh`. It downloads
and verifies Bazel, runs the installer tests, builds the native SDK, and exports
the installation and metadata. The Dockerfile copies its inputs and invokes it.

The SDK has its own `.bazelversion`, `.bazelrc` and standalone Bazel module here
(`MODULE.bazel`, Bzlmod). Bazel detects the native compiler; the application keeps its
existing custom toolchain. `MODULE.bazel` pins google-cloud-cpp; Bazel resolves
everything it depends on (grpc, protobuf, abseil, BoringSSL, ...) from the Bazel
Central Registry. `config/` is a local module both builds share (the ABI compatibility
header). The image does not copy or watch any `cpp/` files.

`export.bzl` preserves the transitive `CcInfo` headers, include paths, defines,
static libraries, always-link requirements and linker flags. `install.py`
dereferences generated and virtual headers into a standalone Bazel package. The
final image contains no SDK builder sources or Bazel cache. Application builds do
not compile protobuf or the SDK's other dependencies.

Rebuild the devcontainer after changing SDK sources, settings or pins under
`.devcontainer/gcs/`, or the `build-gcs-sdk.sh` script. The consumer checks these
SDK-owned inputs (including `MODULE.bazel`, so a google-cloud-cpp bump is a mismatch)
and the installed compiler version. Changing this directory's `.bazelversion`
also requires updating the matching Linux AMD64 and ARM64 Bazel binary checksums
in `build-gcs-sdk.sh`. The application's `cpp/.bazelversion` can change independently.

Normal application changes to warnings, optimization or debug symbols do not
invalidate the SDK. The shared `compatibility.h`, compiled by both workspaces,
requires GCC 9, C++17 or newer, libstdc++'s C++11 ABI and no `_GLIBCXX_DEBUG`.
An incompatible application configuration fails during compilation. Intentional
compiler/ABI migrations must update the SDK contract and rebuild the image.
These checks cover the supported contract, not every possible compiler flag;
other ABI-changing toolchain edits still require compatibility review. Application
sanitizer flags do not instrument the prebuilt SDK itself. Build from the full
repository checkout so both workspaces can access the shared configuration.

The first PR adding this SDK builds new candidate images. After merging, allow
**Build and Push DevContainer** to publish the matching image before relying on
it from application-only PRs or release builds. Existing images without the SDK
fail with a rebuild instruction; there is no silent source-build fallback.

Run the installer tests with:

```sh
python3 -B -m unittest discover -s .devcontainer/gcs -p 'test_*.py'
```

Validate the native images with GCS C++ builds/tests and the existing Python
integration suite. Inspect `bazel aquery 'mnemonic("CppCompile", deps(//gcs:libstreamergcs.so))' --config=<architecture>`
to confirm that no Google SDK dependency compilation remains. Measure fresh
application build time separately from SDK image construction and image transfer.
