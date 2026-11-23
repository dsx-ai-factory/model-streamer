# Dependency list

The scripts in this directory create `dependencies.json`, a list of everything this repository
depends on to **build, test and run**, with the exact version and where to download it (URL, hash,
and the commit for git repositories). The list itself is not stored in the repository: it is a
snapshot of what registries serve on the day it is made, so generate it when you need it (see
[Reproducing the list](#reproducing-the-list)). The tooling keeps and stores no dependency, but collecting does download: Bazel fetches the
external repositories into a temporary cache, and an archive that no file pins is downloaded once
to compute its hash.

## What is in the list

Each entry has a `name`, a `version`, an `ecosystem`, the `categories` that say why we need it, the
files or lines that declare it (`declared_by`), the `files` to download (one per platform where
the file depends on the architecture), and its [`license`](#licenses).

| Category | Meaning |
|---|---|
| `ship` | Code linked into a native library we distribute: what the `NOTICE` files list (`# attribution:` comments in `.devcontainer/Dockerfile`). |
| `run` | Installed by the user's pip next to our wheels (`install_requires`). Not distributed by us. |
| `build` | Needed to build the deliverables. |
| `test` | Needed only to run the tests. |
| `base-image` | A base image of the devcontainer, listed by digest and registry link only. Its packages are not listed. |
| `ci` | A GitHub action the workflows use. |

An entry can have several categories, for example a library that is built from source (`build`)
and linked into a native library (`ship`).

| Ecosystem | Where the entry comes from | Download |
|---|---|---|
| `docker-image` | the `FROM` lines of `.devcontainer/Dockerfile` (a `FROM` that names an earlier stage pulls nothing), and the `image:` lines of the test compose files (`test`) | the digest of the image |
| `devcontainer-feature` | `.devcontainer/devcontainer.json` | the digest of the OCI artifact |
| `apt` | packages the devcontainer has on top of its base image | the `.deb` and its SHA-256 from the package index |
| `pip` | the `run`, `test` and `build` requirement groups, resolved by pip | the wheel and its SHA-256 |
| `npm` | `npm install -g` lines (azurite) and their dependencies | the tarball and its SHA-512 |
| `url` | archives the Dockerfile and its scripts download, the Bazel release, bazelisk, buildifier | the file and its SHA-256 |
| `git` | repositories the Dockerfile and its scripts clone, with every submodule | the repository and the commit |
| `bazel` | the repositories `bazel fetch //...` leaves in the output base | the archive and its SHA-256, or the git remote and commit |
| `github-action` | the `uses:` lines of the workflows | the repository and the commit |

## Where the inputs come from

The collector reads these files, so a new dependency is picked up when it is added there:

| File | Gives |
|---|---|
| `.devcontainer/Dockerfile` | base images, `apt-get install`, `pip install`, `npm install`, `curl` downloads (with the hash the file pins), `git clone` |
| `.devcontainer/scripts/*.sh` | the same commands, in the shell scripts the Dockerfile runs. A script is not run: the architecture argument (`amd64`, `arm64`), `NAME=value` lines and `case amd64)` branches are expanded to give one URL and hash per architecture, and the hash is the `echo "<hash>  <file>" \| sha256sum -c -` line for the file. A `NAME=$(cat FILE)` line is expanded from the file the Dockerfile `COPY`s to that path in the image (for example the SDK builder's `.bazelversion`). A URL with a variable that cannot be expanded stops the collector with the script and line. |
| `.devcontainer/devcontainer.json` | devcontainer features and the packages of the apt-packages feature |
| `tests/**/compose*.yaml` | the images of the test environments (`image: name:tag`, or the default of `${VARIABLE:-name:tag}`) (`test`) |
| `py/*/setup.py` | `install_requires` of the four wheels (`run`) |
| `py/*/pyproject.toml` | build requirements (`build`) |
| `py/runai_model_streamer/requirements.dev` and the `pip download` line of `py/runai_model_streamer/Makefile` | test requirements and the torch index (`test`) |
| `cpp/.bazelversion` | the Bazel release |
| `cpp/MODULE.bazel` and `.devcontainer/gcs/MODULE.bazel` | the Bazel modules, as Bzlmod resolves them |
| `.github/workflows/*` | GitHub actions |

`categories.json` holds the judgment calls the files cannot tell: which packages are only for tests,
which dependencies are linked into a native library (`ship`, mapped to component ids), and the
licenses nothing else names (`licenses`). Edit it when a dependency is added.

## Licenses

Every entry has a `license`, a `license_source` that says where the license came from, and, for a
nonstandard license, a `license_url` with its text. The collector takes the license from the first
source that covers the entry:

1. The row of `cpp/NOTICE` for the component, for an entry with the `ship` category. This is the license the shipped
   notices state. The row is found by the last part of its package name (`github.com/axboe/liburing` is `liburing`).
   An entry that maps to two components lists both licenses.
2. `categories.json`, section `licenses`: `"<ecosystem>:<name>": "<SPDX>"` or `{"license": "<SPDX>", "url": "<license text>"}`.
   Use it for what the package cannot say: a Docker image, a GitHub action, a downloaded archive, a git repository.
3. What the package says about itself:

| Ecosystem | Source |
|---|---|
| `pip` | the wheel metadata in pip's resolver report: `License-Expression`, then a short `License`, then a `License` that is the text of a common license, then the trove classifiers |
| `apt` | `/usr/share/doc/<package>/copyright` in the image: its `License:` fields, or the `/usr/share/common-licenses` files it points to, or the text of a common license |
| `npm` | the `license` of the lock file |
| `bazel`, `go` | the top-level license files of the fetched repository, when every one of them is the text of one common license |

What the package says about itself is kept in `detected_license`, and `license`, `license_source` and `license_url` are rebuilt
from the sources on every run, `relicense` included: removing or changing an override in `categories.json` and running
`relicense` leaves nothing of the old value.

An entry no source covers has the license `UNKNOWN`. The collector never guesses. `collect` logs how many
dependencies have each license and lists the `UNKNOWN` ones, to add to `licenses` in `categories.json` once you have read their license;
`check --strict-licenses` fails while any is left. A license from `cpp/NOTICE` or `categories.json` that is not a common
one (`curl`, a CUDA header notice) must have a `license_url`, and `check` fails without it.
A `ship` entry whose component has no row in `cpp/NOTICE` is reported as a warning: the component ships but is not
declared for the notice pipeline.

`collect` also writes `dependencies.md` next to `dependencies.json`: one row per dependency with its name, license, category and
ecosystem, with the link for a nonstandard license. It is what to paste into a review. To print it again, as a markdown or CSV
table: `python3 .github/scripts/dependencies/collect.py table --format md|csv [--out FILE]`.

### Finding what the collector does not read

Reading files cannot tell that a file is missing from the reading. `collect.py completeness` needs no
network, Docker or list, and fails when:

- a tracked file contains a command that downloads or installs something (`curl` or `wget` with a URL,
  `git clone`, `pip`, `npm` or `apt-get install`, `FROM`, `http_archive`, `uses:`) and the collector does not
  read the file. The files the collector reads are listed in `completeness.py`; Bazel's own `.bzl` files
  and files that are not an input (this directory, Markdown) are listed under `coverage` in `categories.json`;
- the Dockerfile or one of its scripts has more `curl` URLs, `git clone`, `npm`, `pip` or `apt-get install`
  commands than the collector turned into dependencies, which means a command is written in a shape the
  readers do not know, or it uses `wget`, which the readers do not support;
- a file the collector reads for something other than commands has a command in it: a Makefile is read for
  `pip install` and `pip download`, a workflow for `uses:`, so a `curl` in a Makefile or an `apt-get install`
  in a workflow step is not in the list.

`check` runs it too. When it fails, teach the collector about the new file or shape, or list the file
under `coverage` with a reason: a command that is not an input of the build (a CI tool, say) goes under
`allowed_commands` with the kinds it may hold and why.

## Reproducing the list

Run this again whenever a dependency, a Dockerfile line, a workflow action or a requirement file
changes, and before a release. The build environment is the devcontainer image. CI builds the image
natively for each architecture; the collector runs the x86_64 one (see [Limits](#limits) for what
that leaves out of the aarch64 build environment).

1. **Prerequisites:** Docker, network access, Python 3.8 or newer, and `git` on the host. On an ARM
   host, Docker runs the x86_64 image under emulation, which is slower but gives the same result.
2. **Use the image CI uses.** Pull the current one:
   `docker pull ghcr.io/dsx-ai-factory/model-streamer/devcontainer:latest` (for a release branch,
   the `release-<major>.<minor>` tag the release workflow uses). The list records the image digest.
3. **Collect:**
   ```
   python3 .github/scripts/dependencies/collect.py collect --image ghcr.io/dsx-ai-factory/model-streamer/devcontainer:latest
   ```
   It writes `.github/scripts/dependencies/dependencies.json` and, next to it, the license table
   `dependencies.md` (see [Licenses](#licenses)), and takes tens of minutes; the Bazel fetch, which
   downloads the external repositories for both architectures into a temporary cache inside the
   container, is the longest part and needs several GB of disk. `--no-hash` skips the downloads
   that only compute a hash. `--out` writes elsewhere (the table goes next to it, with the extension `.md`).
   Neither file is committed.
4. **Validate and test:**
   ```
   python3 .github/scripts/dependencies/collect.py check
   python3 .github/scripts/dependencies/test_collect.py
   ```
   `check` fails on an unknown category, an entry with nothing to download or without a license, a nonstandard
   license without a link, or a file or command the collector does not read (see above). It also lists the
   dependencies whose license is `UNKNOWN`; pass `--strict-licenses` to fail on them.
5. **Review the list.** Compare it with the previous one you generated. New entries mean a new dependency; if one is linked
   into a native library, add it to `ship` in `categories.json`, and mark it
   `# attribution: shipped` where it is pinned so the `NOTICE` files list it. A package that is only for tests belongs in `categories.json`
   under its ecosystem. `collect` stops if a `ship` key matches nothing.
6. **Complete the licenses.** Read the license of each dependency that `collect` lists as `UNKNOWN`, add it to `licenses` in
   `categories.json` (with a `url` if it is not a common license), then run
   `python3 .github/scripts/dependencies/collect.py relicense`. It applies the licenses again to the existing list and rewrites
   `dependencies.md`, with no new collection. Repeat until `collect.py check --strict-licenses` passes.
7. **Take the table.** `dependencies.md` has one row per dependency with its name, license and category: it is what a review asks for.

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
  no longer matches a package, so update them to the new names, and update the toolchain
  `# attribution: shipped component=` lines (`gcc-runtime`, `glibc-nonshared`) in `.devcontainer/Dockerfile`.
- The Python version of the image (3.11 now) decides what pip resolves; see the limits below.

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
     Python (3.11) and architecture. The `run` group is also resolved for aarch64, because users
     install our aarch64 wheel. Test requirements use the PyTorch CPU index, as `make install_dev_dep` does.
   - **npm:** a lock file written without installing (`--package-lock-only`).
   - **bazel:** `bazel fetch //...` in the application (`cpp/`) and the SDK builder
     (`.devcontainer/gcs/`), then every fetched module of the resolved graphs (`bazel mod graph`)
     with the URL and hash `bazel mod show_repo` records for it. Module resolution does not depend
     on the architecture, so the native configuration covers the set.
   - **tools:** the bazelisk and buildifier versions the devcontainer feature installed.
3. The two sets are merged: one entry per ecosystem, name and version, categories united.

## Limits

- The list is a snapshot of what the registries and package indexes serve on the day it is made
  (`generated`). A mutable reference (`ubuntu:20.04`, an unpinned `pip install boto3`) resolves to
  what it points to now. The Dockerfile does not pin those, so a rebuilt image can differ.
- An apt package whose exact version has left the index has no download URL; the entry says so.
- The devcontainer image is built natively for each architecture, and the collector runs the x86_64 one. The apt packages
  and Bazel repositories are those of the x86_64 build environment; the arm64 image installs the same packages for its own
  architecture (same names and versions, different `.deb` files), which the list does not repeat. Archives that a script
  downloads for both architectures, and the `run` requirements, are listed for both.
- Python is resolved for 3.11, the devcontainer's Python. Our own wheels are `py3-none`, so users on
  other Python versions resolve other dependency versions. The dependency wheels in the list are
  either universal (`py3-none-any`) or compatible with Python 3.11 on an architecture (`cp311`
  or `abi3`, `manylinux`).
- Archives that no file pins (OpenSSL, curl, zlib, libxml2, fake-gcs-server) get a SHA-256 computed
  from the download; the entry says `computed from the download`.
- Git submodule entries take the highest release tag that points to the pinned commit.
- The real-GPU tests (`tests/gpu`) install Python and apt packages inside their container when they run, from
  requirement specs the run passes in, and a nightly run takes the newest torch. Those packages are not in the list; the
  images of the test environments are (from `tests/gpu/compose.yaml`), and the packages the tests share with the
  `test` group are listed there. The scripts are named under `resolved_at_test_time` in `categories.json`.
- A license is what the package's own metadata or license file says, matched by its text. It is not a legal review: a license
  the matcher does not recognize is `UNKNOWN`, and a package with a nonstandard license must be read.
