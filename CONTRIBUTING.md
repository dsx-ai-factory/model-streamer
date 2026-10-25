# Contributing to Model Streamer

Thank you for your interest in contributing to Model Streamer! This document provides guidelines and instructions to help you get started with contributing to our project.

## Getting Started
### New Contributors
We're excited to help you make your first contribution! Whether you're looking to file issues, develop features, fix bugs, or improve documentation, we're here to support you through the process.

Browse issues labeled `good first issue` or `help wanted` on GitHub for an easy introduction.

### Developers
The main building blocks of Model Streamer are documented in the [docs](docs/README.md) folder. Here are the key components:
- `cpp/streamer` - Core C++ streaming engine
- `cpp/s3`, `cpp/gcs`, `cpp/azure` - Object storage backend clients
- `cpp/cc` - C API exposed to Python via ctypes
- `py/runai_model_streamer` - Python SDK
- `py/runai_model_streamer_s3`, `py/runai_model_streamer_gcs`, `py/runai_model_streamer_azure` - Storage-specific Python packages

We recommend reviewing the [README](README.md) to understand the system architecture and build process before making significant contributions.

## How to Contribute
### Signing Off Your Work

We require that all contributors "sign-off" on their commits. This certifies
that the contribution is your original work, or you have rights to submit it
under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be
accepted. Read the full [Developer Certificate of Origin 1.1](DCO) before
signing off. The upstream text is at [developercertificate.org](https://developercertificate.org/).

Use the `--signoff` (or `-s`) option when committing your changes:

```bash
git commit -s -m "Add cool feature."
```

This appends a line using your configured Git identity:

```text
Signed-off-by: Your Name <your@email.com>
```

Only add your own sign-off when you can make the DCO certification. Preserve
the sign-offs and attribution of other contributors.

### Full Text of the DCO

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### NVIDIA IP Review

NVIDIA contributors and maintainers must follow the
[NVIDIA IP review process](https://nv/ip_review_process) for ongoing changes
and third-party contributions before accepting or distributing them.

### Source File Headers

Every source file needs this two-line NVIDIA SPDX header, written in the comment style of the file type:

```text
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Documentation, configuration and data files (for example `.md`, `.yaml`, `.json`) do not need one.
A pull request check fails when a file is missing the header. To add it, run
`python3 .github/scripts/spdx_headers/spdx_headers.py --fix` from the repository root.

To add the header automatically on every commit, install [pre-commit](https://pre-commit.com) and run
`pre-commit install` once.

### Reporting Issues
Open an issue with a clear description, steps to reproduce, and relevant environment details.

### Improving Documentation
Help us keep the docs clear and useful by fixing typos, updating outdated information, or adding examples.

### Contributing Changes
- Open or Find an Issue – Non-trivial pull requests should be associated with a GitHub issue. Search existing issues first; if none fits, open one before you start work. See [Linking Pull Requests to Issues](#linking-pull-requests-to-issues).
- Fork and Clone – Begin by forking the repository and cloning it to your local machine.
- Create a Branch – Use a descriptive branch name, such as `feature/add-cool-feature` or `bugfix/fix-issue123`.
- Make Changes – Keep your commits small, focused, and well-documented. For build and test instructions, refer to the [Development section](README.md#development) of the README.
- Submit a PR – For non-trivial changes, link the pull request to its issue (for example `Fixes #123` or `Relates to #123` in the description).
- Coverage - Please look at the coverage change details and create unit tests, integration tests or end-to-end tests to cover new functionality or changes.

### Linking Pull Requests to Issues

We ask that non-trivial pull requests reference an issue. The issue is the single place where
the problem, the scope, and the design discussion live. A non-trivial pull request without
an issue is hard to track, especially when:

- Changes are unified or split across several pull requests.
- The scope of a change shifts during review.
- Maintainers need to see why a change was made, or what else is related to it.

Use `Fixes #<issue>` or `Closes #<issue>` when the pull request completes the
issue, and `Relates to #<issue>` when it is only a part of it. If review changes
the scope, update the issue, or open follow-up issues for the work that moves out.

Trivial changes, such as typo fixes or small documentation updates, don't need
an issue. Maintainers may ask for one if a change turns out to be larger than it
first appeared.

### PR Title Guidelines

We recommend following the [Conventional Commits](https://www.conventionalcommits.org/) title specification. The format is:

```
<type>[optional scope]: <description>
```

#### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, whitespace)
- **refactor**: Code changes that neither fix a bug nor add a feature
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI/CD configuration
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

#### Scopes (Optional)

Common scopes for Model Streamer:
- `streamer`
- `s3`
- `gcs`
- `azure`
- `py`
- `cpp`
- `ci`
- `docs`

#### Breaking Changes

Breaking changes MUST be indicated by adding `!` after the type/scope: `feat(s3)!: remove deprecated field`

#### Examples

```
feat(s3): add retry with bounded deadline for transient chunk failures
fix(azure): resolve race condition in credential refresh
docs: update installation guide
refactor(streamer): simplify ring buffer allocation
feat(gcs)!: remove deprecated field from client config
```

#### Tips

- Use the imperative mood: "add feature" not "added feature"
- Don't end with a period

### Pull Request Checklist

Before introducing major changes, we strongly recommend opening a PR that outlines your proposed design.
Each pull request should meet the following requirements:
- All tests pass – Run the full test suite locally with: `make test`
- Test coverage – Add or update tests for any affected code.
- Documentation – Update relevant documentation to reflect your changes.
- PR description – Clearly describe what changed and why.
- Linked issue – For non-trivial changes, the description references the associated issue (`Fixes #123`).
- DCO – Every commit has the contributor's `Signed-off-by` line.

## Pull Request CI Environment

### Linux wheel compatibility

All four packages target `manylinux_2_28` on x86_64 and aarch64. The Ubuntu
build image is newer than that runtime baseline: the semaphore resolves
`sem_clockwait` at runtime and uses `sem_timedwait` when unavailable, while
the portable linker rules select older compatible libm symbols for SDK calls.
Do not lower wheel tags without checking the actual ELF requirements.

The devcontainer prebuilds the Google Cloud C++ SDK, including REST/gRPC and their
transitive dependencies, alongside AWS and Azure. CI and local images install only
their native SDK architecture.
Ordinary PRs compile our GCS implementation against these installed libraries.
SDK versions and SDK build settings live under `.devcontainer/gcs/`; changes there
trigger image preparation and publication. Application settings under `cpp/` do
not trigger an image rebuild. Rebuild local images after changing SDK inputs;
stale SDKs and unsupported compiler/ABI settings produce an explicit error. See
[the GCS SDK build notes](.devcontainer/gcs/README.md) for validation and rollout.

The setup job checks whether `.devcontainer/**` or the PR/devcontainer workflows
changed. If they did, two preparation jobs build
AMD64 and ARM64 images in parallel using the reusable `devcontainer.yml` workflow.
Each builds only its matching architecture's dependencies, on `ubuntu-24.04` or
`ubuntu-24.04-arm`, using the matching published architecture tag as a cache.
Missing caches allow a clean build. Each image installs only its native GCC/G++
and binutils; dependencies use native compilation without CMake cross toolchains.

With `publish: false`, each preparation job exports its image as a one-day Actions
artifact named `ci-devcontainer-<run>-<attempt>-<amd64|arm64>`. Image tags also include
the run, attempt, and architecture. Each component job downloads only its matching
image, and each integration job downloads that image and its matching wheels.
PRs never push images to GHCR. These artifacts provide a per-run image handoff,
not a cache restored across PR runs.

When image-related files are unchanged, preparation is skipped. Setup resolves
`ghcr.io/dsx-ai-factory/model-streamer/devcontainer:latest` (or `release-X.Y` for a
release-branch PR) to one immutable multi-architecture index digest. Setup
requires both Linux AMD64 and ARM64 entries. Docker selects the matching entry
on each native runner; all consumers in the run use the same published image pair.
A missing or incomplete index fails setup with publication guidance.

Each component job starts one container and uses `docker exec` for separate C++
build, Python package build, C++ test, and Python test steps. Integration uses
`devcontainers/ci` with an image-only configuration. Both paths preserve the
seccomp setting required by io_uring tests and reuse the image's installed Features.

The **Build and Push DevContainer** workflow builds only the two native images.
On master it publishes `latest-amd64` and `latest-arm64`; release branches publish
`release-X.Y-amd64` and `release-X.Y-arm64`. After both succeed, the reusable
`publish-devcontainer.yml` workflow creates the generic `latest` or `release-X.Y`
manifest from the immutable image digests produced by that run. It verifies both
Linux platforms and returns the index digest and immutable image reference.
Native image builds and pushes remain in `devcontainer.yml`. The generic tag
references those same two images; it does not require another image build. If either image job fails, the generic tag stays
unchanged, even if the other architecture-specific tag was already updated.

The native build workflow `devcontainer.yml` requires an `architecture` input
(`amd64` or `arm64`) to select the native runner and image tag. Docker's
`TARGETARCH` selects the libraries and services installed in both CI and local
source builds. Bazel registers only
the native toolchain, and `make build` (including `make install` and `make test`)
builds only the current architecture. Build the other architecture on its native
runner. Dependency versions, installation prefixes, and release artifact names
are unchanged.

### Parallel PR checks

A setup job computes `PACKAGE_VERSION` once. Separate `build_x86` and `build_arm`
jobs in `on-pr.yaml` each call `build.yml` for four components: core (`streamer`),
S3, GCS, and Azure. They pass `x86_64` and `aarch64`, respectively, for eight builds
in total. The reusable workflow handles one architecture/component pair,
accepting the package version, selected image tag, and optional image artifact.
It contains no matrix, so other workflows can call a single build or define their
own matrix. Each invocation runs
`make ci-build-cpp` and `make ci-build-python` inside the selected devcontainer, using the same
native build command and packaging target as the existing full build. Both
commands take `COMPONENT` and `ARCH`; `make ci-build` runs them in order locally.
Each native job then runs
`make ci-test-cpp COMPONENT=... ARCH=<architecture>` before uploading its wheel. The test
command uses the same architecture and backend defines as the preceding build,
allowing Bazel to reuse compatible compiled outputs on that runner.

S3, GCS, and Azure run their respective C++ test trees; core runs everything
outside those three trees, including common, POSIX I/O, and utility tests.
The core job on each architecture also runs `make ci-test-python`, installing only
its core wheel and Python test dependencies before running unit and distributed
tests. Python feedback therefore does not wait for the cloud backend builds.

After both component build matrices succeed, two integration jobs download their
matching wheels and run `make ci-test-integration` on native AMD64 and ARM64
runners: integration suites followed by the filesystem strategy sweep. Each uses
`make ci-install` to install all four wheels via the existing Python package
Makefiles. Python distributed tests load the installed real library; the mock
unit tests retain their existing override. Each integration job also reuses the
installed core library for the Python filesystem strategy
tests through `test-unit-real-installed`, avoiding another full native build.
The same tests still run for all four strategies. Each integration job still builds
the two C++ strategy test targets and the Azure testing variant. `make test` remains
the full sequential local entry point and builds from source as before.

The build jobs upload eight intermediate `ci-packages-<component>-<arch>`
artifacts, retained for seven days for use by later CI jobs. The **Assemble packages**
job runs alongside integration tests after both architectures finish building.
It calls the reusable `assemble-bundle.yml` workflow to create one final download,
`release-bundle-pr-<run>-<attempt>`, with the same layout as the release bundle:
eight wheels under `wheels/`, two complete C++ SDK
archives under `cpp/`, and `SHA256SUMS`. SDK archives contain the libraries,
public headers, license, and third-party notices. The bundler checks completeness,
ELF architectures, matching headers/licenses, and checksums before upload.
PR versions use `0.0.0+dev.<commit>`; release publication still requires `X.Y.Z`.
Download the bundle for Python wheels and complete C++ SDKs. Bundle assembly
runs independently of integration tests; require the final **Test, Build & Push**
check to pass before using a PR build.
Intermediate wheel and C++ artifacts are retained for seven days in both PR and
release builds to support retries. Final PR bundles are retained for seven days;
tested `release-bundle-<version>-<run>-<attempt>` bundles are retained for 30 days
for approval and recovery.
Release CI resolves the release branch's generic multi-architecture tag once,
then uses that index digest for both native build and integration matrices. It
calls the same assembly workflow after both integration jobs succeed,
then passes its bundle artifact ID to the approval-gated publication job.
The shared workflow downloads components, tests the packaging code, assembles
and verifies the bundle, and uploads it. Callers select the version, artifact
name, final-bundle retention, and development-version opt-in.
Each build job also uploads an intermediate `ci-cpp-<component>-<arch>`
artifact containing its shared library under `lib/` and the repository license.
All C++ artifacts include public C/C++ headers under
`include/streamer/`. Backend artifacts contain their plugin library; use them
alongside the core library for the same architecture. Libraries are copied out
of the container before it stops so the downloads contain real files rather
than Bazel symlinks. The `Test, Build & Push` required check succeeds
only when all eight builds (including their C++ and Python tests), integration
tests on both architectures, package assembly, and any triggered image builds
succeed. Each architecture's component jobs can start as soon as its own image is
ready, without waiting for the other image builder.

To verify an extracted PR bundle locally, substitute its development version:

```sh
python3 .github/scripts/release_bundle.py verify \
  --version 0.0.0+dev.abc1234 --allow-development-version --bundle release-bundle
```

## Getting Help
Need support or have a question? We're here to help:
- Report issues or ask questions by [opening an issue on GitHub](https://github.com/dsx-ai-factory/model-streamer/issues).

## License
By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for your interest and happy coding!
