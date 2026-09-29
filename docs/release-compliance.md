# Release compliance checklist

Use this checklist when preparing source and binary distributions. Repository
checks provide evidence for review; they do not grant release approval.

## Repository evidence

| Requirement | Evidence |
| --- | --- |
| Project license | [Apache 2.0 license](../LICENSE) |
| NVIDIA source attribution | Headers in [C++ sources](../cpp/streamer/streamer.cc), [Python sources](../py/runai_model_streamer/runai_model_streamer/__init__.py), build files, and examples |
| Incorporated third-party source | Original glibc notice and modification notice in [portability.c](../cpp/cc/portability/portability.c) |
| Third-party licenses | [THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt) and its [source manifest](../licensing/third_party_manifest.json) |
| External contributions | [Sign-off instructions and full DCO](../CONTRIBUTING.md#signing-off-your-work) |

Use links at the reviewed commit when providing evidence to OSRB. The four
Python package directories contain symlinks to the root license and notices;
their setup metadata includes both files in wheels and source distributions.

## Before distributing a release

- Follow the [NVIDIA IP review process](https://nv/ip_review_process) for
  ongoing changes and third-party contributions.
- Record the closest VP's release approval in the OSRB bug comments.
  A PR, a signed-off commit, or this checklist is not that approval.
- Reconcile the native dependency inventory with the actual Linux build for
  both supported architectures and each released backend. Include embedded
  dependencies, generated code, optional features, and compiler runtime
  libraries. The checked-in inventory follows source dependency manifests;
  it is not an SBOM extracted from the finished binaries.
- Verify the LGPL distribution arrangement with OSRB for libaio and
  the incorporated glibc function. The default build statically links
  libaio, and the glibc-derived function is compiled into the
  native libraries. Distribute the required corresponding source and, where
  required, relinkable object files and build instructions with the binaries.
  License texts and upstream download URLs alone do not establish that these
  obligations have been fulfilled. The system-library build option does not
  remove the incorporated glibc code.
- If redistributing a Python environment, resolve and inventory its exact
  package versions and all transitive dependencies. Model Streamer wheels
  declare Python dependencies for separate installation; the reference
  Python notices in this repository are not a dependency lockfile.
- Review a distributed development container separately, including its OS
  packages, MinIO, fake-gcs-server, Azurite, Node.js, and other build/test tools.
  The wheel notice inventory is not a complete container inventory.
- Run the repository license check and inspect every built wheel:

  ```bash
  python3 tools/check_licenses.py
  python3 tools/check_licenses.py --wheels py/*/dist/*.whl
  ```

## Updating notices

The dependency versions originate in `.devcontainer/Dockerfile`,
`cpp/toolchain/deps.bzl`, `cpp/third_party/gcp_repo.bzl`, and their upstream
manifests. AWS CRT revisions follow the SDK's pinned submodules. Bazel uses
the first declared repository: for example, this project selects Protobuf
27.0 before Google Cloud C++ declares its fallback version.

For each dependency change, review upstream license and NOTICE files, update
`licensing/third_party_manifest.json` with the source revision and SHA-256
of each original file, then run:

```bash
python3 tools/update_third_party_notices.py
```

This command downloads only the recorded files, verifies their hashes, and
regenerates the aggregate notice. For source files used as attribution evidence,
it includes their opening comments. Review the generated diff, including
upstream NOTICE references to additional bundled licenses. A successful
generation checks provenance, not completeness of the release inventory.
