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
### Reporting Issues
Open an issue with a clear description, steps to reproduce, and relevant environment details.

### Improving Documentation
Help us keep the docs clear and useful by fixing typos, updating outdated information, or adding examples.

### Contributing Changes
- Fork and Clone – Begin by forking the repository and cloning it to your local machine.
- Create a Branch – Use a descriptive branch name, such as `feature/add-cool-feature` or `bugfix/fix-issue123`.
- Make Changes – Keep your commits small, focused, and well-documented. For build and test instructions, refer to the [Development section](README.md#development) of the README.
- Submit a PR – Open a pull request and reference any relevant issues or discussions.
- Coverage - Please look at the coverage change details and create unit tests, integration tests or end-to-end tests to cover new functionality or changes.

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

## Pull Request CI Environment

The setup job checks whether `.devcontainer/**` changed. If it did, a single
preparation job builds the checked-out devcontainer using the reusable
`devcontainer.yml` workflow and the published image as a build cache. With
`publish: false`, it saves the image as a temporary Actions artifact, retained for
one day. The artifact and image tags include the run ID and attempt to isolate
retries. This is a per-run image handoff, not a cache restored across PR runs.

All eight build jobs and the integration job download and load that same image.
Each build job starts one Docker container and uses `docker exec` for separate
C++ build, Python package build, C++ test, and Python test steps. The container
shares build outputs and installed dependencies across steps and is stopped even
if a build or test fails. The integration job runs through `devcontainers/ci`
with an image-only configuration. Neither path rebuilds the Dockerfile or
reinstalls Features. When `.devcontainer/**` is
unchanged, the preparation job is skipped and jobs use the published
`ghcr.io/dsx-ai-factory/model-streamer/devcontainer:latest` image directly. Both
paths preserve the seccomp setting required by the io_uring tests.

The **Build and Push DevContainer** workflow calls the same reusable workflow
with `publish: true` on master. It pushes the image to GHCR instead of uploading
an image artifact. PRs never push images to GHCR. The final required check fails
if preparation of a changed image fails; unchanged-image PRs still run all jobs.

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
Each x86_64 job then runs
`make ci-test-cpp COMPONENT=... ARCH=x86_64` before uploading its wheel. The test
command uses the same architecture and backend defines as the preceding build,
allowing Bazel to reuse compatible compiled outputs on that runner.

S3, GCS, and Azure run their respective C++ test trees; core runs everything
outside those three trees, including common, POSIX I/O, and utility tests.
The core x86_64 job also runs `make ci-test-python`, installing only its core wheel
and Python test dependencies before running unit and distributed tests. Python
feedback therefore does not wait for the cloud backend builds. ARM64 jobs build
packages only.

After all four x86_64 builds succeed, the integration job downloads their wheels
and runs `make ci-test-integration`: integration suites followed by the filesystem
strategy sweep, without waiting for ARM64 builds. It uses `make ci-install` to install all four wheels via the
existing Python package Makefiles. Python distributed tests load the installed
real library; the mock unit tests retain their existing override. The integration
job also reuses the installed core library for the Python filesystem strategy
tests through `test-unit-real-installed`, avoiding another full native build.
The same tests still run for all four strategies. The integration job still builds
the two C++ strategy test targets and the Azure testing variant. `make test` remains
the full sequential local entry point and builds from source as before.

Wheels are available in eight `packages-<component>-<arch>` artifacts, uploaded
by the individual build jobs. Each job also uploads a `cpp-<component>-<arch>`
artifact containing its shared library under `lib/` and the repository license.
All C++ artifacts include public C/C++ headers under
`include/streamer/`. Backend artifacts contain their plugin library; use them
alongside the core library for the same architecture. Libraries are copied out
of the container before it stops so the downloads contain real files rather
than Bazel symlinks. The `Test, Build & Push` required check succeeds
only when all eight builds (including their C++ and Python tests) and integration
tests succeed. Changed devcontainers are built once before the matrix starts;
peak matrix concurrency remains eight runners.

## Getting Help
Need support or have a question? We're here to help:
- Report issues or ask questions by [opening an issue on GitHub](https://github.com/dsx-ai-factory/model-streamer/issues).

## License
By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for your interest and happy coding!
