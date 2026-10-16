# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List everything this repository depends on to build, test and run, with where to download it.

    python3 dependencies/collect.py collect --image IMAGE    write dependencies.json (needs Docker and network)
    python3 dependencies/collect.py check                    validate dependencies.json
    python3 dependencies/collect.py resolve ...              (runs inside the devcontainer, started by collect)

No dependency is kept or stored. Collecting does download: Bazel fetches the external repositories
into a temporary cache, and an archive that no file pins is downloaded once to compute its hash.
See README.md.
"""

import argparse
import datetime
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dockerfile  # noqa: E402
import inputs  # noqa: E402
import static_sources  # noqa: E402
from util import CATEGORIES, REPO, local_image_digest, log, read_json, run, write_json  # noqa: E402

OUTPUT = os.path.join(HERE, "dependencies.json")
COMPONENTS = os.path.join(REPO, "third_party_notices", "components.json")


def resolve(args):
    """Inside the devcontainer: the dependencies that depend on the image."""
    import container_sources
    with open(args.base_packages) as f:
        base = dict(line.split("\t") for line in f.read().splitlines() if line.strip())
    found = []
    found += container_sources.apt_dependencies(args.repo, base)
    found += container_sources.pip_dependencies(args.repo)
    found += container_sources.npm_dependencies(args.repo)
    found += container_sources.devcontainer_tools(args.repo, not args.no_hash)
    found += container_sources.bazel_dependencies(args.repo, args.workspace)
    write_json(args.out, found)
    log("resolved %d dependencies" % len(found))


def in_container(image, base_image, no_hash):
    """Run `resolve` in the devcontainer image and return its entries. CI builds both architectures
    in the one x86_64 image (aarch64 is cross-compiled), so one run covers the build environment."""
    with tempfile.TemporaryDirectory() as work:
        try:
            names = run(["docker", "run", "--rm", "--platform", "linux/amd64", base_image, "dpkg-query", "-W", "-f", "${Package}\t${Version}\n"])
        except RuntimeError as error:
            sys.exit("cannot list the packages of the base image %s: %s\nThe apt part of the list supports Debian and Ubuntu "
                     "bases. For another package manager, extend container_sources.py." % (base_image, error))
        with open(os.path.join(work, "base.txt"), "w") as f:
            f.write(names)
        script = ("set -e; cp -r /src /work; cd /work; python3 dependencies/collect.py resolve --repo /src "
                  "--workspace /work --base-packages /out/base.txt --out /out/resolved.json" + (" --no-hash" if no_hash else ""))
        log("container: " + image)
        subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "-v", "%s:/src:ro" % REPO,
                        "-v", "%s:/out" % work, image, "bash", "-c", script], check=True)
        return read_json(os.path.join(work, "resolved.json"))


def merge(entries):
    """One entry per (ecosystem, name, version): categories and declaring files are united, and the
    files of different platforms are listed together."""
    merged = {}
    for item in entries:
        key = (item["ecosystem"], item["name"], item["version"])
        if key not in merged:
            merged[key] = dict(item, categories=list(item["categories"]), declared_by=list(item["declared_by"]),
                               files=list(item["files"]))
            continue
        target = merged[key]
        target["categories"] = sorted(set(target["categories"]) | set(item["categories"]))
        target["declared_by"] = sorted(set(target["declared_by"]) | set(item["declared_by"]))
        seen = {(f["platform"], f["url"]) for f in target["files"]}
        target["files"] += [f for f in item["files"] if (f["platform"], f["url"]) not in seen]
        if item.get("note") and item["note"] not in (target.get("note") or ""):
            target["note"] = ((target.get("note") or "") + " " + item["note"]).strip()
    return sorted(merged.values(), key=lambda e: (e["ecosystem"], e["name"].lower(), e["version"]))


def tag_ship(entries):
    """Add the `ship` category to the dependencies categories.json maps to a linked component.
    Returns the map keys that matched nothing."""
    ship = {key: ids for key, ids in inputs.config()["ship"].items() if not key.startswith("_")}
    unmatched = set(ship)
    for item in entries:
        key = "%s:%s" % (item["ecosystem"], item["name"])
        if key in ship:
            unmatched.discard(key)
            item["categories"] = sorted(set(item["categories"]) | {"ship"})
    return sorted(unmatched)


def base_commit():
    """The master commit the list is based on: where this branch left master, or HEAD when no master
    ref exists (a shallow clone). Resolved before the slow steps so that a failure costs nothing."""
    for ref in ("origin/master", "master"):
        result = subprocess.run(["git", "merge-base", "HEAD", ref], cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            return result.stdout.strip()
    log("no master ref found; recording HEAD as the base commit")
    return run(["git", "rev-parse", "HEAD"], cwd=REPO).strip()


def collect(args):
    base_commit_id = base_commit()
    entries = static_sources.collect(REPO, compute_hashes=not args.no_hash)
    _, image, tag, _, _ = dockerfile.base_images(inputs.read(REPO, inputs.DOCKERFILE))[-1]  # the last FROM is the base
    base = next(e for e in entries if e["ecosystem"] == "docker-image" and (e["name"], e["version"]) == (image, tag or "latest"))
    entries += in_container(args.image, base["files"][0].get("reference", "%s:%s" % (image, base["version"])), args.no_hash)
    merged = merge(entries)
    unmatched = tag_ship(merged)
    if unmatched:
        sys.exit("categories.json `ship` names dependencies that were not found: " + ", ".join(unmatched))
    result = {
        "generated": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "base_commit": base_commit_id,
        "devcontainer_image": {"image": args.image, "digest": local_image_digest(args.image)},
        "categories": {
            "ship": "code linked into a native library we distribute",
            "run": "installed by the user's pip next to our wheels (install_requires)",
            "build": "needed to build the deliverables",
            "test": "needed only to run the tests",
            "base-image": "a base image of the devcontainer, listed by digest only",
            "ci": "a GitHub action the workflows use",
        },
        "dependencies": merged,
    }
    write_json(args.out, result)
    log("wrote %d dependencies to %s" % (len(merged), args.out))


def check(args):
    """Fail if the list is malformed or out of step with the third-party notices."""
    data = read_json(args.out)
    problems = []
    for item in data["dependencies"]:
        label = "%s %s %s" % (item["ecosystem"], item["name"], item["version"])
        if not item["categories"] or any(c not in CATEGORIES for c in item["categories"]):
            problems.append(label + ": bad categories %s" % item["categories"])
        if not item["files"]:
            problems.append(label + ": nothing to download")
    shipped = {"%s:%s" % (item["ecosystem"], item["name"]) for item in data["dependencies"] if "ship" in item["categories"]}
    for key in inputs.config()["ship"]:
        if not key.startswith("_") and key not in shipped:
            problems.append("categories.json `ship` key %s has no entry with the ship category in the list" % key)
    if os.path.exists(COMPONENTS):
        mapped = {cid for key, ids in inputs.config()["ship"].items() if not key.startswith("_") for cid in ids}
        for component in read_json(COMPONENTS)["components"]:
            if component["id"] not in mapped:
                problems.append("component %s of third_party_notices/components.json is not mapped in categories.json `ship`" % component["id"])
    if problems:
        sys.exit("\n".join(problems))
    print("%d dependencies, all well formed" % len(data["dependencies"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("collect", help="build dependencies.json (needs Docker and network)")
    one.add_argument("--image", required=True, help="the devcontainer image, for example ghcr.io/<org>/model-streamer/devcontainer:latest")
    one.add_argument("--no-hash", action="store_true", help="do not download files whose hash nothing pins")
    one.add_argument("--out", default=OUTPUT)
    one.set_defaults(run=collect)
    inside = commands.add_parser("resolve", help="inside the devcontainer")
    inside.add_argument("--repo", required=True)
    inside.add_argument("--workspace", required=True, help="writable copy of the repository for Bazel")
    inside.add_argument("--base-packages", required=True, help="file with the name and version of each package of the base image, tab separated")
    inside.add_argument("--no-hash", action="store_true")
    inside.add_argument("--out", required=True)
    inside.set_defaults(run=resolve)
    verify = commands.add_parser("check", help="validate dependencies.json")
    verify.add_argument("--out", default=OUTPUT)
    verify.set_defaults(run=check)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
