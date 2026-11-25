# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the corresponding-source archive that accompanies a release.

    release_sources.py create --version X.Y.Z [--output source-archive] [--cache DIR]

The archive holds the sources listed in .github/release-sources.json, one subdirectory per
component, and a build-materials/ subdirectory with the files needed to rebuild the native
libraries with a modified copy of one of them. Each download is checked against the sha256
recorded in the manifest.

The manifest names the statically linked copyleft (LGPL/GPL) components whose licence requires
their source to be available. Each one must have a row in cpp/NOTICE, the notice the SDK archives
ship, and the version in the manifest must be the version of that row (a Debian revision suffix
after a dash is allowed: 2.31-0ubuntu9.18 matches 2.31). The script stops otherwise, so the
manifest cannot drift from what the notices say is shipped.

The manifest also lists, under "registry", the files that the Bazel Central Registry holds for the
modules the build takes from it (source.json and the overlay build files), because the libraries
cannot be rebuilt without them and the registry is not ours. They go to
build-materials/bazel-registry/<module>/<version>/. Each module version must be the one that
cpp/MODULE.bazel depends on.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request

from release_bundle import regular_files, require, sources_name, write_sdk

# What a recipient needs, besides the sources, to rebuild with a modified library. The list is
# relative to the repository root; a directory is included recursively.
BUILD_MATERIALS = (
    ".devcontainer",
    ".github/release-sources.json",
    "attribution.json",
    "cpp/.bazelrc",
    "cpp/.bazelversion",
    "cpp/MODULE.bazel",
    "cpp/bazel_downloader.cfg",
    "cpp/NOTICE",
    "cpp/rules.bzl",
    "cpp/third_party",
    "cpp/toolchain",
    "docs/native-library-rebuilding.md",
)
LISTING = "SOURCES.txt"
ATTEMPTS = 3
# Ubuntu's mirrors answer 403 to Python's default User-Agent but serve a named client.
USER_AGENT = "runai-model-streamer-release-sources"


def _load_licenses():
    """Load dependencies/licenses.py by path. The release gate and the license table must read a
    cpp/NOTICE row the same way, so they share its NOTICE_ROW and component_id. Loading by path keeps
    the other modules of that folder (util.py, inputs.py) off the import path."""
    path = Path(__file__).resolve().parent / "dependencies" / "licenses.py"
    spec = importlib.util.spec_from_file_location("dependency_licenses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_licenses = _load_licenses()
NOTICE_ROW = _licenses.NOTICE_ROW
component_id = _licenses.component_id


def notice_rows(path):
    """Map component id to the list of (version, licence) rows of the notice file."""
    require(path.is_file(), f"Missing notice file: {path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = NOTICE_ROW.match(line)
        if match:
            rows.setdefault(component_id(match.group(1)), []).append((match.group(2), match.group(3)))
    require(rows, f"No component rows in {path}")
    return rows


def version_matches(version, row_version):
    """The manifest version equals the notice version, extends it with a dash revision (2.31-0ubuntu9.18
    for 2.31), or is the upstream version of a Bazel Central Registry module whose notice row adds the
    registry revision (0.3.113 for 0.3.113.bcr.1)."""
    return (version == row_version or version.startswith(row_version + "-")
            or row_version.startswith(version + ".bcr."))


def select_sources(entries, rows):
    """Check the manifest against the notice rows and return (entry, licence) pairs in manifest order."""
    ids = [entry["id"] for entry in entries]
    require(len(set(ids)) == len(ids), "Duplicate component id in the source manifest")
    missing = [entry["id"] for entry in entries if entry["id"] not in rows]
    require(not missing, "Components in the source manifest without a row in cpp/NOTICE: " + ", ".join(missing))
    selected = []
    for entry in entries:
        found = [row for row in rows[entry["id"]] if version_matches(entry["version"], row[0])]
        require(found, f"{entry['id']}: the manifest has version {entry['version']} but cpp/NOTICE lists "
                       + ", ".join(version for version, _ in rows[entry["id"]]))
        selected.append((entry, found[0][1]))
    return selected


def file_name(url):
    name = Path(urllib.parse.urlparse(url).path).name
    require(name and name not in (".", ".."), f"Cannot derive a file name from {url}")
    return name


def download(url, destination, digest):
    """Download url to destination and check its sha256. Retry transient network errors only."""
    for attempt in range(1, ATTEMPTS + 1):
        sha = hashlib.sha256()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                        timeout=120) as response, destination.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    sha.update(chunk)
                    stream.write(chunk)
        except OSError as error:
            if attempt == ATTEMPTS:
                raise ValueError(f"Cannot download {url}: {error}") from error
            time.sleep(2 ** attempt)
            continue
        require(sha.hexdigest() == digest,
                f"Checksum mismatch for {url}: expected {digest}, got {sha.hexdigest()}")
        return


def sha256_of(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def fetch(url, destination, digest, cache, key=None):
    """Put the verified file in destination: from the cache directory when it holds a good copy,
    otherwise from url, and then store it in the cache. A damaged cached copy is downloaded again.
    key is the path of the copy in the cache (default: the folder and name of destination)."""
    cached = cache / (key or f"{destination.parent.name}/{destination.name}") if cache else None
    if cached and cached.is_file() and sha256_of(cached) == digest:
        shutil.copyfile(cached, destination)
        return
    download(url, destination, digest)
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, cached)


def check_registry(entries, module_file):
    """Check that each registry entry is the module version that cpp/MODULE.bazel depends on, so the
    files in the archive are the ones the build uses. Return the entries."""
    require(module_file.is_file(), f"Missing module file: {module_file}")
    text = module_file.read_text(encoding="utf-8")
    seen = set()
    for entry in entries:
        module, version = entry["module"], entry["version"]
        require(re.fullmatch(r"[0-9A-Za-z._+-]+", module) and re.fullmatch(r"[0-9A-Za-z._+-]+", version),
                f"Invalid registry module: {module} {version}")
        require((module, version) not in seen, f"Duplicate registry module: {module} {version}")
        seen.add((module, version))
        pin = rf'bazel_dep\(\s*name\s*=\s*"{re.escape(module)}"\s*,\s*version\s*=\s*"{re.escape(version)}"'
        require(re.search(pin, text), f"{module}: the registry entry has version {version} but "
                                      f"{module_file.name} does not depend on that version")
        require(entry["files"], f"{module}: the registry entry lists no file")
        for item in entry["files"]:
            parts = Path(item["path"]).parts
            require(parts and ".." not in parts and not Path(item["path"]).is_absolute(),
                    f"Invalid registry file path: {item['path']}")
    return entries


def registry_name(entry, item):
    """The path of a registry file in the archive."""
    return f"build-materials/bazel-registry/{entry['module']}/{entry['version']}/{item['path']}"


def build_materials(root):
    """Map archive names (build-materials/...) to the regular files under the repository root."""
    files = {}
    for entry in BUILD_MATERIALS:
        path = root / entry
        require(path.exists() and not path.is_symlink(), f"Missing build material: {entry}")
        if path.is_dir():
            found = {f"{entry}/{name}": file for name, file in regular_files(path).items()}
        else:
            require(path.is_file(), f"Not a regular file: {entry}")
            found = {entry: path}
        files.update({f"build-materials/{name}": file for name, file in found.items()
                      if "__pycache__" not in Path(name).parts})
    return files


def listing(version, selected, registry=()):
    lines = [f"Corresponding sources for runai-model-streamer {version}", "",
             "Each subdirectory holds the unmodified upstream source of a statically linked",
             "component whose licence requires it to be available, with the sha256 of each file.",
             "No patches are applied to any of them.", ""]
    for component, licence in selected:
        lines += [f"{component['name']} {component['version']} - {licence}",
                  f"  directory: {component['dir']}/"]
        lines += [f"  {entry['sha256']}  {file_name(entry['url'])}" for entry in component["files"]]
        lines.append("")
    lines += ["build-materials/ holds the container recipe, the pinned dependency builds and the",
              "rebuild guide (build-materials/docs/native-library-rebuilding.md): everything needed",
              "to rebuild the native libraries with a modified copy of a component above.", ""]
    if registry:
        lines += ["build-materials/bazel-registry/ holds the files that the Bazel Central Registry",
                  "gives each module the build uses: its source.json and the overlay files (the Bazel",
                  "build files that the registry adds to the upstream archive), with their sha256:", ""]
        for entry in registry:
            lines.append(f"{entry['module']} {entry['version']}")
            lines += [f"  {item['sha256']}  {item['path']}" for item in entry["files"]]
            lines.append("")
    lines += [
              "The GCC runtime libraries (libstdc++, libgcc) are linked statically under the GCC",
              "Runtime Library Exception, which permits distributing the result under other terms.",
              "Their sources are not included.", ""]
    return "\n".join(lines)


def create(version, manifest, notices, root, output, cache=None):
    require(re.fullmatch(r"[0-9A-Za-z._+-]+", version), f"Invalid version: {version}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    selected = select_sources(data["components"], notice_rows(notices))
    require(selected, "The source manifest lists no component")
    registry = check_registry(data.get("registry", []), root / "cpp" / "MODULE.bazel")
    archive = output / f"{sources_name(version)}.tar.gz"
    require(not archive.exists(), f"Already exists: {archive}")
    with tempfile.TemporaryDirectory(prefix="release-sources-") as work:
        work = Path(work)
        files = {}
        for component, _ in selected:
            require(re.fullmatch(r"[0-9A-Za-z._+-]+", component["dir"]), f"Invalid directory: {component['dir']}")
            (work / component["dir"]).mkdir()
            for entry in component["files"]:
                name = f"{component['dir']}/{file_name(entry['url'])}"
                require(name not in files, f"Duplicate source file: {name}")
                files[name] = work / name
                fetch(entry["url"], files[name], entry["sha256"], cache)
        files.update(build_materials(root))
        for entry in registry:
            for item in entry["files"]:
                name = registry_name(entry, item)
                require(name not in files, f"Duplicate registry file: {name}")
                files[name] = work / name
                files[name].parent.mkdir(parents=True, exist_ok=True)
                fetch(item["url"], files[name], item["sha256"], cache, key=name.removeprefix("build-materials/"))
        files[LISTING] = work / LISTING
        files[LISTING].write_text(listing(version, selected, registry), encoding="utf-8")
        output.mkdir(parents=True, exist_ok=True)
        write_sdk(archive, sources_name(version), files)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("create",))
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=Path("source-archive"))
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root holding the build materials")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="source manifest (default: .github/release-sources.json under --root)")
    parser.add_argument("--notices", type=Path, default=None,
                        help="notice file whose rows the manifest is checked against (default: cpp/NOTICE under --root)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="directory that keeps verified downloads between runs; a file whose sha256 "
                             "matches the manifest is used without downloading")
    args = parser.parse_args()
    try:
        archive = create(args.version, args.manifest or args.root / ".github" / "release-sources.json",
                         args.notices or args.root / "cpp" / "NOTICE", args.root, args.output, args.cache)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Release sources error: {error}")
    print(archive)


if __name__ == "__main__":
    main()
