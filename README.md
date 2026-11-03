# Model Streamer

> [!IMPORTANT]
> **Repository move and rename notice:** On September 13, 2026, Run:ai Model Streamer was
> renamed to Model Streamer and the repository moved from the run-ai GitHub organization to
> [`dsx-ai-factory/model-streamer`](https://github.com/dsx-ai-factory/model-streamer).
> The `runai-model-streamer` PyPI packages and Python import paths are unchanged.
> Existing repository URLs and standard Git operations continue to work through
> GitHub redirects. If you maintain automation or integrations that reference
> `run-ai/runai-model-streamer`, such as GitHub Actions, webhooks, or pinned
> repository URLs, update them to `dsx-ai-factory/model-streamer`.


## Overview
Model Streamer is a Python SDK designed to facilitate the streaming of tensors from tensors files to GPU memory with concurrency and streaming. It provides an API for loading SafeTensors files and building AI models, allowing loading models seamlessly.

For documentation click [here](docs/README.md)

For benchmarks click [here](docs/src/benchmarks.md)

## Usage
Install the package
```
pip install runai-model-streamer
```

And stream tensors on available GPU device
```
from runai_model_streamer import SafetensorsStreamer
import torch

file_path = "model.safetensors"

device = "cuda"
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"

with SafetensorsStreamer() as streamer:
    streamer.stream_file(file_path)
    for name, tensor in streamer.get_tensors():
        gpu_tensor = tensor.to(f'{device}:0')
```

## Development

Our repository is built using devcontainer ([Further reading](https://containers.dev/))

The following commands should run inside the dev container

> [!NOTE]
> You can use devcontainer-cli tool ([Further reading](https://github.com/devcontainers/cli)) by installing it, and run every command with the following prefix `devcontainer exec --workspace-folder . [COMMAND]`

**Build**
```
make build
```

> [!NOTE]
> We build `libstreamers3.so` and statically link it to libssl, libcrypto, and libcurl. if you would like to use your system libraries by dynamic link to them, run `USE_SYSTEM_LIBS=1 make build`

> [!NOTE]
> On successful build, the `.whl` file would be at `py/runai_model_streamer/dist/<PACKAGE_FILE>` and `py/runai_model_streamer_s3/dist/<PACKAGE_FILE>`


**Run tests**
```
make test
```

**Install locally**
```
pip3 install py/runai_model_streamer py/runai_model_streamer_s3
```

> [!IMPORTANT]
> In order to the CPP to run, you need to install libcurl4 and libssl1.1_1

### Release-branch CI

Branches named `release/<major>.<minor>` have independent CI environments. For
example, PRs targeting `release/0.17` use
`ghcr.io/dsx-ai-factory/model-streamer/devcontainer:release-0.17`, while PRs
targeting `master` continue to use `latest`. Other PR targets retain the existing
`latest` behavior. CI selects the environment from the PR's target branch, not
the contributor's branch name.

Create a release branch from a commit containing these workflows. Creating the
branch automatically starts **Build and Push DevContainer**, even if no
devcontainer files changed. Wait for that run to finish before running ordinary
PRs against the branch. The image is rebuilt when changes to `.devcontainer/`,
the reusable devcontainer workflow, or its publishing workflow land on that
branch. Each branch updates only its own image tag.

PRs changing those files build a temporary image and use it for component builds
and integration tests. They do not publish the temporary image. Builds use the
target branch's image as a cache when available and can build from source when
the cache does not yet exist.

If initialization fails, or an existing release branch has just received these
workflow changes, run **Build and Push DevContainer** manually with the release
branch selected in GitHub Actions, or use:

```sh
gh workflow run build-devcontainer.yml --repo dsx-ai-factory/model-streamer --ref release/0.17
```

Wait for successful publication, then rerun the PR workflow. An ordinary release
PR fails if its published image is unavailable; it never falls back to `master`.
Manual publication is supported only for `master` and `release/<major>.<minor>`.
Tags and unrelated branch creation do not publish images. Branch image tags
track the latest successfully published environment on their branch.

### Publishing a release

Release tags use `X.Y.Z` without a `v` prefix or prerelease suffix. For example,
`0.17.2` must identify a commit contained in `release/0.17`. Git tags do not
record a source branch, so CI checks ancestry against the matching release
branch. The tagged commit must already contain the release workflow and scripts.

Before the first release, configure **Settings → Environments → pypi** with
required reviewers. Manage the approver list in those settings and disable
administrator bypass to require approval for every publication. The publishing
job references `environment: pypi`, so GitHub waits for approval before starting
it. CI does not inspect reviewers or check their repository permissions.
The publishing job uses the existing `PYPI_API_TOKEN` secret; an environment
secret of the same name can be used to scope it to approved jobs.

Wait for the branch's devcontainer publication to succeed, then manually create
and push the tag from the release branch, for example:

```sh
git switch release/0.17
git pull --ff-only
git tag -a 0.17.2 -m "Release 0.17.2"
git push origin refs/tags/0.17.2
```

The tag push triggers **RunAI Model Streamer Library - Release**. It validates
the tag, selects `devcontainer:release-0.17`, and resolves that image to a digest
shared by all jobs. CI builds all four packages for x86_64 and aarch64 and runs
the existing x86_64 C++, Python, integration, and filesystem-strategy tests.
There is no fallback to master's devcontainer. Ensure the branch image matches
the tagged source's build requirements, especially when tagging an older commit.

After every build and test succeeds, CI calls the reusable `assemble-bundle.yml`
workflow, also used for PR bundles. It collects one wheel for each package and
architecture at the tagged version, plus the C++ component artifacts from the
same run. It packages one C++ SDK per architecture, checking the ELF architecture
and matching public headers and licenses across all components. The eight wheels,
two SDK archives, and combined `SHA256SUMS` are saved in a workflow artifact
retained for 30 days. Review the version, commit, image digest, and checksums in
the job summaries, then approve the **pypi** deployment in the Actions run.
Rejection prevents publication.

After approval, CI checks that the saved bundle contains all ten assets and that
every checksum matches. It publishes those exact wheels to the existing PyPI
projects and attaches the same eight wheels, both C++ SDK archives, and
`SHA256SUMS` to the GitHub release. Publication does not rebuild anything.
The existing publishing actions handle the uploads:
`pypa/gh-action-pypi-publish` for PyPI and `softprops/action-gh-release` for GitHub.
A GitHub release is created if needed. Existing release notes are preserved.
Automatically created maintenance releases do not change GitHub's **Latest**
release selection.

For C/C++ applications, download the native SDK with headers and shared libraries.
The libraries link their third-party dependencies, including `libxml2`, statically,
so the SDK needs only the system C and C++ runtime libraries.
For the Python API, install `runai-model-streamer` with pip or use its `.whl`
release assets; the Python package bundles its native library.

The SDK archives are named
`runai-model-streamer-cpp-sdk-<version>-linux-x86_64.tar.gz` and
`runai-model-streamer-cpp-sdk-<version>-linux-aarch64.tar.gz`. Each extracts into a
directory with the same name (without `.tar.gz`) containing:

```text
include/streamer/    Public C API headers, usable from C and C++
lib/libstreamer.so
lib/libstreamers3.so
lib/libstreamergcs.so
lib/libstreamerazure.so
LICENSE
THIRD_PARTY_NOTICES.txt    Licenses of the third-party code linked into the libraries
```

The archives contain the original build outputs without rebuilding or modifying
the library and header bytes. They use normalized ordering, permissions,
ownership, and timestamps. Wheels remain under `wheels/` in the saved workflow
bundle; SDK archives are under `cpp/`. Only the wheels are sent to PyPI.

GitHub release assets and `SHA256SUMS` use plain filenames. Download all assets
and the checksum file into one directory and run `sha256sum --check SHA256SUMS`.
To verify only the SDK for your architecture, download it and `SHA256SUMS` into
one directory and run `sha256sum --check --ignore-missing SHA256SUMS`.
To verify an extracted workflow bundle before manual recovery, use:

```sh
python3 .github/scripts/release_bundle.py verify --version 0.17.2 --bundle release-bundle
```

PyPI uploads fail on duplicate filenames. If nothing reached PyPI, use
**Re-run failed jobs** and approve again. If some wheels were uploaded, download
the original saved bundle, verify that the existing files match its checksums,
and manually upload only the missing wheels. If PyPI completed but GitHub
publication failed, publish the original wheels, both C++ SDK archives, and
`SHA256SUMS` to GitHub separately, checking any existing assets before skipping
them. GitHub assets are not overwritten, and remote checksums are not checked
automatically.

Do not rebuild for recovery: rebuilt wheels can have different bytes. If an
existing PyPI file differs, or the saved bundle has expired, publish a new patch
version. The tag and its release-branch membership are checked again after
approval; do not move published release tags.

Creating or editing a GitHub release does not trigger a second publication run;
the version tag push is the single release trigger. A locally created tag must
be pushed to GitHub before any workflow starts.

Validate release automation locally with:

```sh
python3 -B -m unittest discover -s .github/scripts -p 'test_release.py' -v
```

## Licensing

Model Streamer is licensed under the [Apache License 2.0](LICENSE).

- `py/runai_model_streamer*/LICENSE` are symbolic links to the root [LICENSE](LICENSE), so
  that each Python wheel carries the license text. They are the same license, not a separate one.
- The native libraries statically link third-party code. Their notices and the full upstream
  license texts live in [`third_party_notices/`](third_party_notices/README.md):
  - `third_party_notices/licenses/<component>/` holds the verbatim license and NOTICE files
    of each third-party component (Apache-2.0, MIT, BSD, zlib, curl, LGPL-2.1 and others).
  - `libstreamer.txt`, `libstreamers3.txt`, `libstreamergcs.txt` and `libstreamerazure.txt`
    are the consolidated notices for each wheel. `sdk.txt` is the notice for the C/C++ SDK archive.

