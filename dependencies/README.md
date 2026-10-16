# Dependency list

`dependencies.json` lists everything this repository depends on to **build, test and run**, with
the exact version and where to download it (URL, hash, and the commit for git repositories).
The tooling keeps and stores no dependency, but collecting does download: Bazel fetches the
external repositories into a temporary cache, and an archive that no file pins is downloaded once
to compute its hash.

## What is in the list

Each entry has a `name`, a `version`, an `ecosystem`, the `categories` that say why we need it, the
files or lines that declare it (`declared_by`), and the `files` to download (one per platform where
the file depends on the architecture).

| Category | Meaning |
|---|---|
| `ship` | Code linked into a native library we distribute. These are the components of `third_party_notices/components.json`. |
| `run` | Installed by the user's pip next to our wheels (`install_requires`). Not distributed by us. |
| `build` | Needed to build the deliverables. |
| `test` | Needed only to run the tests. |
| `base-image` | A base image of the devcontainer, listed by digest and registry link only. Its packages are not listed. |
| `ci` | A GitHub action the workflows use. |

An entry can have several categories, for example a library that is built from source (`build`)
and linked into a native library (`ship`).

| Ecosystem | Where the entry comes from | Download |
|---|---|---|
| `docker-image` | the `FROM` lines of `.devcontainer/Dockerfile` | the digest of the image |
| `devcontainer-feature` | `.devcontainer/devcontainer.json` | the digest of the OCI artifact |
| `apt` | packages the devcontainer has on top of its base image | the `.deb` and its SHA-256 from the package index |
| `pip` | the `run`, `test` and `build` requirement groups, resolved by pip | the wheel and its SHA-256 |
| `npm` | `npm install -g` lines (azurite) and their dependencies | the tarball and its SHA-512 |
| `url` | archives the Dockerfile downloads, the Bazel release, bazelisk, buildifier | the file and its SHA-256 |
| `git` | repositories the Dockerfile clones, with every submodule | the repository and the commit |
| `bazel` | the repositories `bazel fetch //...` leaves in the output base | the archive and its SHA-256, or the git remote and commit |
| `github-action` | the `uses:` lines of the workflows | the repository and the commit |

## Where the inputs come from

The collector reads these files, so a new dependency is picked up when it is added there:

| File | Gives |
|---|---|
| `.devcontainer/Dockerfile` | base images, `apt-get install`, `pip install`, `npm install`, `curl` downloads (with the hash the file pins), `git clone` |
| `.devcontainer/devcontainer.json` | devcontainer features and the packages of the apt-packages feature |
| `py/*/setup.py` | `install_requires` of the four wheels (`run`) |
| `py/*/pyproject.toml` | build requirements (`build`) |
| `py/runai_model_streamer/requirements.dev` and the `pip download` line of `py/runai_model_streamer/Makefile` | test requirements and the torch index (`test`) |
| `cpp/.bazelversion` | the Bazel release |
| `cpp/WORKSPACE` and what it loads | the Bazel repositories, as Bazel resolves them |
| `.github/workflows/*` | GitHub actions |

`categories.json` holds the judgment calls the files cannot tell: which packages are only for tests,
and which dependencies are linked into a native library (`ship`, mapped to the ids of
`third_party_notices/components.json`). Edit it when a dependency is added.

## Reproducing the list

Run this again whenever a dependency, a Dockerfile line, a workflow action or a requirement file
changes, and before a release. The build environment is the devcontainer image. CI builds both
architectures in the one x86_64 image (aarch64 is cross-compiled with the cross toolchain in it),
so one run of the image covers the build and the tests for both.

1. **Prerequisites:** Docker, network access, Python 3.8 or newer, and `git` on the host. On an ARM
   host, Docker runs the x86_64 image under emulation, which is slower but gives the same result.
2. **Use the image CI uses.** Pull the current one:
   `docker pull ghcr.io/dsx-ai-factory/model-streamer/devcontainer:latest` (for a release branch,
   the `release-<major>.<minor>` tag the release workflow uses). The list records the image digest.
3. **Collect:**
   ```
   python3 dependencies/collect.py collect --image ghcr.io/dsx-ai-factory/model-streamer/devcontainer:latest
   ```
   It writes `dependencies/dependencies.json` and takes tens of minutes; the Bazel fetch, which
   downloads the external repositories for both architectures into a temporary cache inside the
   container, is the longest part and needs several GB of disk. `--no-hash` skips the downloads
   that only compute a hash. `--out` writes elsewhere.
4. **Validate and test:**
   ```
   python3 dependencies/collect.py check
   python3 dependencies/test_collect.py
   ```
   `check` fails on an unknown category, an entry with nothing to download, or a component of
   `third_party_notices/components.json` that `categories.json` does not map to a `ship` dependency.
5. **Review the diff of `dependencies.json`.** New entries mean a new dependency; if one is linked
   into a native library, add it to `ship` in `categories.json`, and to
   `third_party_notices/components.json`. A package that is only for tests belongs in `categories.json`
   under its ecosystem. `collect` stops if a `ship` key matches nothing.

### Base images

The devcontainer image does not record the registry digest of the base it was built on: its config
holds only the base's name and version labels, the creation time and the layer content digests. So
the digest of a base image that the Dockerfile does not pin is asked of the registry, in this order:

1. the registry's anonymous API (Docker Hub, ghcr.io);
2. `docker buildx imagetools inspect`, which uses this machine's Docker credentials and works for any
   registry (this is what runs when Docker Hub answers HTTP 429 to anonymous requests);
3. the image the local Docker has already pulled. Pull the tag first or run `docker login` if you
   rely on this, so the local copy is current.

The entry says which of these gave the digest. A tag such as `ubuntu:20.04` can move, so the digest is
what the tag pointed to on the day the list was made. To record exactly what the image was built on,
pin the digest in the Dockerfile.

**Changing the base image.** The `FROM` lines are read from the Dockerfile, so another image or registry
is listed without a change here, and the last `FROM` is taken as the base of the devcontainer image.
What else to check after the change:
- `collect` lists the packages of the base with `dpkg-query` and attributes the added ones with
  `/var/log/apt/history.log`. A base that is not Debian or Ubuntu stops with a message; extend
  `container_sources.py` for its package manager.
- The `ship` entries in `categories.json` name apt packages that depend on the distribution release
  (`libstdc++-9-dev`, `libgcc-9-dev`, `libc6-dev` for Ubuntu 20.04, GCC 9). `collect` stops if a key
  no longer matches a package, so update them to the new names, and update the toolchain entries
  (`gcc-runtime`, `glibc-nonshared`) of `third_party_notices/components.json`.
- The Python version of the image (3.8 now) decides what pip resolves; see the limits below.

`collect` does this:

1. On the host, from the files above: base image digests, devcontainer features, the Bazel release,
   the Dockerfile downloads and git clones (and the submodules of each, from the git tree without
   cloning the contents), the GitHub actions (each reference resolved to a commit).
2. In the image (`resolve`, started by `collect`):
   - **apt:** the packages installed on top of the base image (the base image is read by running it),
     with the `.deb` URL and SHA-256 from the package index. `/var/log/apt/history.log` in the image says
     which apt command installed each package. The packages that command names, when the Dockerfile
     or a devcontainer feature names them, give the categories and the declaring line. Commands that a
     script runs, such as the one the nodesource setup script runs, are mapped in `categories.json`
     (`apt_commands`).
   - **pip:** `pip install --dry-run --report` per requirement group, wheels only, for the image's
     Python (3.8) and architecture. The `run` group is also resolved for aarch64, because users
     install our aarch64 wheel. Test requirements use the PyTorch CPU index, as `make install_dev_dep` does.
   - **npm:** a lock file written without installing (`--package-lock-only`).
   - **bazel:** `bazel fetch //...` for both architectures, then every repository in the output base
     with the URL and hash of its repository rule.
   - **tools:** the bazelisk and buildifier versions the devcontainer feature installed.
3. The two sets are merged: one entry per ecosystem, name and version, categories united.

## Limits

- The list is a snapshot of what the registries and package indexes serve on the day it is made
  (`generated`). A mutable reference (`ubuntu:20.04`, an unpinned `pip install boto3`) resolves to
  what it points to now. The Dockerfile does not pin those, so a rebuilt image can differ.
- An apt package whose exact version has left the index has no download URL; the entry says so.
- Python is resolved for 3.8, the devcontainer's Python. Our own wheels are `py3-none`, so users on
  other Python versions resolve other dependency versions. The dependency wheels in the list are
  either universal (`py3-none-any`) or specific to Python 3.8 and an architecture (`cp38`,
  `manylinux`).
- Archives that no file pins (OpenSSL, curl, zlib, libxml2, fake-gcs-server) get a SHA-256 computed
  from the download; the entry says `computed from the download`.
- Git submodule entries take the highest release tag that points to the pinned commit.
