# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies that can be listed on any machine, with network access but without the
devcontainer: base images, the archives and git repositories the Dockerfile fetches, devcontainer
features, the Bazel binary and the GitHub actions the workflows use."""

import re
import urllib.error
import urllib.request

import dockerfile
import inputs
from util import (archive_name_version, entry, file_record, http_sha256, image_parts, imagetools_digest, local_image_digest,
                  log, ls_remote, oci_digest, run, submodules)

def _page(registry, repository):
    if registry == "docker.io":
        official = repository.startswith("library/")
        return "https://hub.docker.com/" + ("_/" + repository[len("library/"):] if official else "r/" + repository)
    return "https://%s/%s" % (registry, repository)


def base_images(repo):
    """Each FROM in the Dockerfile, whatever the image or registry: the digest the Dockerfile pins, or
    the one a registry serves today for a tag. The last FROM is the base of the devcontainer image;
    the others are build stages whose files are copied in."""
    found = []
    images = dockerfile.base_images(inputs.read(repo, inputs.DOCKERFILE))
    for index, (line, image, tag, digest, stage) in enumerate(images):
        registry, repository = image_parts(image)
        tag = tag or "latest"
        source = "pinned in Dockerfile"
        if not digest:
            digest, source = oci_digest(registry, repository, tag), "registry, for the mutable tag"
        if not digest:  # a registry without anonymous access, or Docker Hub rate-limiting (HTTP 429)
            digest, source = imagetools_digest("%s:%s" % (image, tag)), "the registry, with this machine's Docker credentials, for the mutable tag"
        if not digest:
            digest, source = local_image_digest("%s:%s" % (image, tag)), "the image pulled by the local Docker, for the mutable tag"
        note = ("The Dockerfile pins this digest." if source.startswith("pinned") else
                "The Dockerfile uses the mutable tag %s:%s; the digest is what it pointed to when the list was made." % (image, tag))
        note += (" Base of the devcontainer image." if index == len(images) - 1 else
                 " Build stage %s: files are copied from it into the image." % (stage or index))
        found.append(entry(image, tag, "docker-image", ["base-image"],
                           [file_record(_page(registry, repository), digest=("sha256", digest.split(":")[1], source) if digest else None,
                                        reference="%s/%s:%s@%s" % (registry, repository, tag, digest) if digest else None)],
                           ["%s:%d" % (inputs.DOCKERFILE, line)], note))
    return found


def _download_name(record):
    url = next(iter(record["urls"].values()))
    if "nodesource.com" in url:
        return "nodesource-setup", re.search(r"setup_(\S+)", url).group(1)
    return archive_name_version(record["filename"], url)


def downloads(repo, compute_hashes):
    """Archives the Dockerfile downloads. A hash the Dockerfile pins is used as it is; otherwise it
    is computed from the file the URL serves now."""
    found = []
    for record in dockerfile.downloads(inputs.read(repo, inputs.DOCKERFILE)):
        name, version = _download_name(record)
        categories = inputs.categories_for("download", name)
        files = []
        for platform, url in record["urls"].items():
            pinned = record["sha256"].get(platform) or record["sha256"].get("any")
            if pinned:
                digest = ("sha256", pinned, "pinned in Dockerfile")
            elif compute_hashes:
                log("hashing " + url)
                digest = ("sha256", http_sha256(url), "computed from the download")
            else:
                digest = None
            files.append(file_record(url, platform=platform, digest=digest))
        found.append(entry(name, version, "url", categories, files, ["%s:%d" % (inputs.DOCKERFILE, record["line"])]))
    return found


def _tag_for(url, commit):
    """A tag that points to the commit, preferring release-looking names; the short commit if none."""
    commits = {}
    for line in run(["git", "ls-remote", "--tags", url], check=False).splitlines():
        sha, _, ref = line.partition("\t")
        name = ref[len("refs/tags/"):]
        if name.endswith("^{}"):
            commits[name[:-3]] = sha  # an annotated tag: the commit it points to wins over the tag object
        else:
            commits.setdefault(name, sha)
    tags = [name for name, sha in commits.items() if sha == commit]
    releases = [tag for tag in tags if re.match(r"v?\d+(\.\d+)*$", tag)]
    if releases:  # several release tags can point to one commit; take the highest
        return max(releases, key=lambda tag: [int(part) for part in re.findall(r"\d+", tag)])
    return sorted(tags)[0] if tags else commit[:12]


def _repo_name(url):
    return re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1])


def git_repositories(repo):
    """The git repositories the Dockerfile clones, with the commit and every submodule."""
    found = []
    for clone in dockerfile.git_clones(inputs.read(repo, inputs.DOCKERFILE)):
        commit = ls_remote(clone["url"], clone["ref"])
        if not commit:
            raise RuntimeError("cannot resolve %s at %s" % (clone["url"], clone["ref"]))
        categories = ["build"]
        declared = ["%s:%d" % (inputs.DOCKERFILE, clone["line"])]
        found.append(entry(_repo_name(clone["url"]), clone["ref"], "git", categories,
                           [file_record(clone["url"], commit=commit)], declared,
                           "Cloned with its submodules; they are listed as separate entries." if clone["recursive"] else None))
        if clone["recursive"]:
            log("submodules of %s" % clone["url"])
            for path, url, sha in submodules(clone["url"], commit):
                found.append(entry(_repo_name(url), _tag_for(url, sha), "git", categories,
                                   [file_record(url, commit=sha)], declared,
                                   "Submodule %s of %s %s." % (path, _repo_name(clone["url"]), clone["ref"])))
    return found


def devcontainer_features(repo):
    """Devcontainer features: OCI artifacts the devcontainer build pulls by tag."""
    found = []
    for reference in inputs.devcontainer_features(repo):
        path, _, tag = reference.partition(":")
        registry, _, repository = path.partition("/")
        digest = oci_digest(registry, repository, tag, "application/vnd.oci.image.manifest.v1+json")
        found.append(entry(path, tag, "devcontainer-feature", ["build"],
                           [file_record("https://" + path, digest=("sha256", digest.split(":")[1], "registry") if digest else None,
                                        reference="%s:%s@%s" % (path, tag, digest) if digest else None)],
                           [inputs.DEVCONTAINER], "Pulled by tag when the devcontainer is built."))
    return found


def bazel_binary(repo):
    """The Bazel release bazelisk downloads, per .bazelversion."""
    version = inputs.bazel_version(repo)
    files = []
    for arch, suffix in (("x86_64", "x86_64"), ("aarch64", "arm64")):
        url = "https://releases.bazel.build/%s/release/bazel-%s-linux-%s" % (version, version, suffix)
        try:
            checksum = http_text(url + ".sha256").split()[0]
            digest = ("sha256", checksum, "published next to the release")
        except urllib.error.URLError:
            digest = None
        files.append(file_record(url, platform=arch, digest=digest))
    return [entry("bazel", version, "url", ["build"], files, ["cpp/.bazelversion"])]


def http_text(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode()


def github_actions(repo):
    """The actions the workflows use, with the commit each reference points to."""
    found = []
    for (action, ref), workflows in sorted(inputs.workflow_actions(repo).items()):
        owner_repo = "/".join(action.split("/")[:2])
        url = "https://github.com/" + owner_repo
        commit = ref if re.fullmatch(r"[0-9a-f]{40}", ref) else ls_remote(url, ref)
        found.append(entry(action, ref, "github-action", ["ci"], [file_record(url, commit=commit)], workflows,
                           None if commit else "Could not resolve this reference to a commit."))
    return found


def collect(repo, compute_hashes=True):
    found = []
    for step in (base_images, devcontainer_features, bazel_binary, github_actions, git_repositories):
        log("static: " + step.__name__)
        found += step(repo)
    log("static: downloads")
    found += downloads(repo, compute_hashes)
    return found
