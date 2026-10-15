# Prebuilt Google Cloud C++ SDK

The Dockerfile builds `storage` and `storage_grpc` for x86_64 and aarch64, then
installs the static SDK under `/opt/gcs-sdk`. Application Bazel builds load that
directory as `@google_cloud_cpp`; they do not download or compile the Google SDK.
Both REST and gRPC support remain enabled.

Application builds still fetch protobuf's Bazel rule definitions because
`rules_cc` imports them. They do not compile protobuf or its dependencies.

The builder uses the application's toolchains, `.bazelrc`, and pinned dependency
definitions. `cpp/third_party/gcp_repo.bzl` is the Google SDK version source and
includes the existing protobuf override. The workspace functions run only during
image construction. Bazel is checksum-verified in the builder stage; when updating
`cpp/.bazelversion`, update its Dockerfile checksums.

`export.bzl` obtains the transitive static archives, headers, include directories,
defines, and linker flags from Bazel's `CcInfo`. `install.py` copies them into
standalone architecture packages, preserving generated headers, virtual include
paths, and always-link semantics. This avoids maintaining a manual list of gRPC,
protobuf, Abseil, and other transitive libraries. The final image contains these
packages, not the builder's sources, generators, or Bazel cache.

Rebuild the devcontainer when SDK versions, toolchains, or compiler settings
change. Existing images without `/opt/gcs-sdk` cannot build this checkout; there
is deliberately no fallback that silently recompiles the SDK during PR CI. The
repository rule also rejects an installed SDK with different Google SDK or
protobuf source checksums. If CI uses a published image directly, publish the
matching image before running that checkout. Both PR image selection and the image publisher watch all
the source inputs copied by this stage.

From the repository root, test the installer with:

```sh
python3 -m unittest discover -s .devcontainer/gcs -p 'test_*.py'
```

Validate the image by building `//gcs:libstreamergcs.so` with both architecture
configs, running the native C++ tests and GCS integration tests, and checking that
the application action graph contains no third-party SDK compilation actions.
