# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the collectors. Python 3.8 compatible: part of the collection runs in the
devcontainer, whose Python is 3.8."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CATEGORIES = ("ship", "run", "build", "test", "base-image", "ci")


def log(message):
    print(message, file=sys.stderr, flush=True)


def entry(name, version, ecosystem, categories, files, declared_by, note=None):
    """One dependency. `files` is a list of {platform, url, sha256, sha256_from, filename?}; platform
    is "any" for a file that does not depend on the architecture."""
    for category in categories:
        assert category in CATEGORIES, category
    out = {"name": name, "version": version, "ecosystem": ecosystem, "categories": sorted(set(categories)),
           "declared_by": sorted(set(declared_by)), "files": files}
    if note:
        out["note"] = note
    return out


def file_record(url, platform="any", filename=None, digest=None, **extra):
    """One downloadable file. `digest` is (algorithm, hex, where it came from), for example
    ("sha256", "ab12...", "pinned in Dockerfile"). Extra keys, such as commit or reference, are kept."""
    record = {"platform": platform, "url": url}
    if filename:
        record["filename"] = filename
    if digest:
        algorithm, value, source = digest
        record[algorithm] = value
        record[algorithm + "_from"] = source
    record.update({key: value for key, value in extra.items() if value})
    return record


def http_sha256(url):
    """SHA-256 of what a URL serves. Downloads and discards the body; only the hash is kept."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "dependency-list"})
    with urllib.request.urlopen(request, timeout=300) as response:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run(command, cwd=None, check=True, timeout=None):
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError("%s failed (%d): %s" % (" ".join(command), result.returncode, result.stderr.strip()))
    return result.stdout


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False)
        f.write("\n")


def archive_name_version(filename, url):
    """(name, version) from an archive file name such as openssl-3.5.0.tar.gz or
    fake-gcs-server_1.52.2_Linux_amd64.tar.gz. A file name with no version (linux_amd64.tar.gz) takes
    the repository and tag of a GitHub release URL instead."""
    match = re.match(r"^(?P<name>.+?)[-_]v?(?P<version>\d[\w.]*?)(?:[-_].*)?\.(?:tar\.gz|tar\.xz|tgz|zip)$", filename or "")
    if match:
        return match.group("name"), match.group("version")
    release = re.match(r"https://github\.com/[^/]+/([^/]+)/releases/download/v?([^/]+)/", url)
    if release:
        return release.group(1), release.group(2)
    return os.path.basename(url), "unversioned"


def oci_digest(registry, repository, tag, accept=None):
    """Digest the registry serves for repository:tag, using an anonymous pull token, or None."""
    accept = accept or ("application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,"
                        "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json")
    hosts = {"docker.io": ("registry-1.docker.io", "https://auth.docker.io/token?service=registry.docker.io&scope=repository:%s:pull"),
             "ghcr.io": ("ghcr.io", "https://ghcr.io/token?scope=repository:%s:pull")}
    if registry not in hosts:
        return None
    host, token_url = hosts[registry]
    try:
        with urllib.request.urlopen(token_url % repository, timeout=60) as response:
            token = json.load(response)["token"]
        request = urllib.request.Request("https://%s/v2/%s/manifests/%s" % (host, repository, tag),
                                         headers={"Authorization": "Bearer " + token, "Accept": accept}, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.headers.get("Docker-Content-Digest")
    except (urllib.error.URLError, KeyError):
        return None


def image_parts(image):
    """(registry, repository) of an image name: ubuntu -> docker.io, library/ubuntu;
    nvidia/cuda -> docker.io, nvidia/cuda; ghcr.io/org/name -> ghcr.io, org/name."""
    first = image.split("/")[0]
    if "/" in image and ("." in first or ":" in first or first == "localhost"):
        return first, image[len(first) + 1:]
    return "docker.io", image if "/" in image else "library/" + image


def imagetools_digest(reference):
    """Digest of the image a tag points to, asked of the registry through this machine's Docker (its
    credentials apply, so it works for any registry), or None."""
    try:
        out = run(["docker", "buildx", "imagetools", "inspect", reference, "--format", "{{.Manifest.Digest}}"],
                  check=False, timeout=120).strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out if re.fullmatch(r"sha256:[0-9a-f]{64}", out) else None


def local_image_digest(reference):
    """Digest of an image already pulled by the local Docker, or None. Used when the registry
    refuses an anonymous request (Docker Hub rate-limits them)."""
    try:
        out = run(["docker", "image", "inspect", reference, "--format", "{{range .RepoDigests}}{{println .}}{{end}}"],
                  check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    digests = [line.split("@")[1] for line in out.splitlines() if "@" in line]
    return digests[0] if digests else None


def ls_remote(repo_url, ref):
    """Commit a tag or branch points to (an annotated tag is peeled), or None."""
    try:
        out = run(["git", "ls-remote", repo_url, ref, ref + "^{}"], check=False, timeout=90)
    except subprocess.TimeoutExpired:
        return None
    commits = {}
    for line in out.splitlines():
        sha, _, name = line.partition("\t")
        commits[name] = sha
    for name in ("refs/tags/%s^{}" % ref, "refs/tags/%s" % ref, "refs/heads/%s" % ref):
        if name in commits:
            return commits[name]
    return None


def submodules(repo_url, commit, depth=0, seen=None):
    """Every submodule of a repository at a commit, recursively, as (path, url, commit). Only the
    tree and .gitmodules are fetched, not the file contents."""
    seen = set() if seen is None else seen
    found = []
    with tempfile.TemporaryDirectory() as work:
        run(["git", "init", "-q", work])
        run(["git", "remote", "add", "origin", repo_url], cwd=work)
        run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], cwd=work)
        tree = run(["git", "ls-tree", "-r", commit], cwd=work)
        links = [line.split() for line in tree.splitlines() if line.startswith("160000")]
        if not links:
            return found
        config = run(["git", "show", "%s:.gitmodules" % commit], cwd=work)
    urls = {}
    for block in re.split(r"\[submodule ", config)[1:]:
        path = re.search(r"path\s*=\s*(\S+)", block)
        url = re.search(r"url\s*=\s*(\S+)", block)
        if path and url:
            urls[path.group(1)] = url.group(1)
    for _, _, sha, path in links:
        url = urls.get(path)
        if not url or (url, sha) in seen:
            continue
        seen.add((url, sha))
        found.append((path, url, sha))
        found.extend(submodules(url, sha, depth + 1, seen))
    return found
