# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List everything this repository depends on to build, test and run, with where to download it.

    python3 .github/scripts/dependencies/collect.py collect --image IMAGE    write dependencies.json and its license table (needs Docker and network)
    python3 .github/scripts/dependencies/collect.py check                    validate dependencies.json
    python3 .github/scripts/dependencies/collect.py relicense                apply categories.json licenses to dependencies.json again
    python3 .github/scripts/dependencies/collect.py table [--format md|csv]  print the name, license and category of every dependency
    python3 .github/scripts/dependencies/collect.py completeness             offline: does the collector read every file that downloads something
    python3 .github/scripts/dependencies/collect.py resolve ...              (runs inside the devcontainer, started by collect)

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

import completeness as completeness_checks  # noqa: E402
import dockerfile  # noqa: E402
import inputs  # noqa: E402
import licenses  # noqa: E402
import static_sources  # noqa: E402
from util import CATEGORIES, REPO, local_image_digest, log, read_json, run, write_json  # noqa: E402

OUTPUT = os.path.join(HERE, "dependencies.json")


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
    """Run `resolve` in the devcontainer image and return its entries. CI builds the image natively
    for each architecture; this runs the x86_64 one, so the apt and Bazel entries are those of the
    x86_64 build environment (see "Limits" in README.md)."""
    with tempfile.TemporaryDirectory() as work:
        try:
            names = run(["docker", "run", "--rm", "--platform", "linux/amd64", base_image, "dpkg-query", "-W", "-f", "${Package}\t${Version}\n"])
        except RuntimeError as error:
            sys.exit("cannot list the packages of the base image %s: %s\nThe apt part of the list supports Debian and Ubuntu "
                     "bases. For another package manager, extend container_sources.py." % (base_image, error))
        with open(os.path.join(work, "base.txt"), "w") as f:
            f.write(names)
        script = ("set -e; cp -r /src /work; cd /work; python3 .github/scripts/dependencies/collect.py resolve --repo /src "
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
        if item.get("license") and not target.get("license"):
            target["license"], target["license_source"] = item["license"], item["license_source"]
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
    table_path(args.out)  # refuse a name the table would overwrite, before the slow steps
    base_commit_id = base_commit()
    entries = static_sources.collect(REPO, compute_hashes=not args.no_hash)
    _, image, tag, _, _ = dockerfile.external_images(inputs.read(REPO, inputs.DOCKERFILE))[1]  # the image the last stage builds on
    base = next(e for e in entries if e["ecosystem"] == "docker-image" and (e["name"], e["version"]) == (image, tag or "latest"))
    entries += in_container(args.image, base["files"][0].get("reference", "%s:%s" % (image, base["version"])), args.no_hash)
    merged = merge(entries)
    unmatched = tag_ship(merged)
    if unmatched:
        sys.exit("categories.json `ship` names dependencies that were not found: " + ", ".join(unmatched))
    components = licenses.notice_components(REPO)
    licenses.assign(merged, components, inputs.config())
    for problem in licenses.ship_without_notice(merged, components, inputs.config()):
        log("warning: " + problem)
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
    write_table(result, args.out)
    report_licenses(result)


def table_path(out):
    """The path of the license table, next to the list: dependencies.json -> dependencies.md. A list
    that is already named `.md` would be overwritten by its own table, so it is refused."""
    path = os.path.splitext(out)[0] + ".md"
    if path == out:
        sys.exit("--out %s must not end in .md: the license table is written to that name" % out)
    return path


def write_table(data, out):
    """The license table, next to the list: dependencies.json -> dependencies.md."""
    path = table_path(out)
    with open(path, "w") as f:
        f.write(licenses.table(data))
    log("wrote the license table to %s" % path)


def relicense(args):
    """Apply the rows of cpp/NOTICE and the `licenses` of categories.json to a list again,
    so that a license added to categories.json does not need a new collection. Rewrites the list and
    its table."""
    table_path(args.out)
    if not os.path.exists(args.out):
        sys.exit("%s does not exist: run `collect` first, or pass --out" % args.out)
    data = read_json(args.out)
    components = licenses.notice_components(REPO)
    licenses.assign(data["dependencies"], components, inputs.config())
    for problem in licenses.ship_without_notice(data["dependencies"], components, inputs.config()):
        log("warning: " + problem)
    write_json(args.out, data)
    write_table(data, args.out)
    report_licenses(data)


def report_licenses(data):
    """Say how the licenses of a list are spread, and which entries no source names."""
    log("licenses: " + ", ".join("%s: %d" % (name, count) for name, count in
                                 sorted(licenses.summary(data).items(), key=lambda pair: (-pair[1], pair[0]))))
    missing = licenses.unknown(data)
    if missing:
        log("%d dependencies have no known license (add them to `licenses` in categories.json):\n  %s" % (len(missing), "\n  ".join(missing)))


def completeness(args):
    """Fail if a file or a command that downloads something is not read by the collector. Needs no
    network and no list."""
    problems = completeness_checks.problems(REPO)
    if problems:
        sys.exit("\n".join(problems))
    print("every file and command that downloads something is read by the collector")


def check(args):
    """Fail if the list is malformed, out of step with categories.json, or the collector misses a
    file or a command."""
    if not os.path.exists(args.out):
        sys.exit("%s does not exist: run `collect` first, or pass --out" % args.out)
    data = read_json(args.out)
    problems = completeness_checks.problems(REPO)
    for item in data["dependencies"]:
        label = "%s %s %s" % (item["ecosystem"], item["name"], item["version"])
        if not item["categories"] or any(c not in CATEGORIES for c in item["categories"]):
            problems.append(label + ": bad categories %s" % item["categories"])
        if not item["files"]:
            problems.append(label + ": nothing to download")
    problems += licenses.problems(data)
    shipped = {"%s:%s" % (item["ecosystem"], item["name"]) for item in data["dependencies"] if "ship" in item["categories"]}
    for key in inputs.config()["ship"]:
        if not key.startswith("_") and key not in shipped:
            problems.append("categories.json `ship` key %s has no entry with the ship category in the list" % key)
    if problems:
        sys.exit("\n".join(problems))
    print("%d dependencies, all well formed" % len(data["dependencies"]))
    report_licenses(data)
    if args.strict_licenses and licenses.unknown(data):
        sys.exit("%d dependencies have no known license" % len(licenses.unknown(data)))


def table(args):
    """Print the name, license and category of every dependency of a list."""
    if not os.path.exists(args.out):
        sys.exit("%s does not exist: run `collect` first, or pass --out" % args.out)
    sys.stdout.write(licenses.table(read_json(args.out), args.format))


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
    verify.add_argument("--strict-licenses", action="store_true", help="also fail when a dependency has no known license")
    verify.set_defaults(run=check)
    again = commands.add_parser("relicense", help="apply categories.json licenses to dependencies.json again, no new collection")
    again.add_argument("--out", default=OUTPUT)
    again.set_defaults(run=relicense)
    listing = commands.add_parser("table", help="print the name, license and category of every dependency")
    listing.add_argument("--out", default=OUTPUT, help="the dependencies.json to read")
    listing.add_argument("--format", choices=("md", "csv"), default="md")
    listing.set_defaults(run=table)
    commands.add_parser("completeness", help="check that the collector reads every file and command that downloads something "
                                              "(offline, needs no list)").set_defaults(run=completeness)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
