# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies that only the devcontainer can resolve, because the answer depends on the image's
operating system, Python and architecture: apt packages, pip and npm packages, and the repositories
Bazel fetches. Run inside the image by `collect.py resolve`. Python 3.8 compatible."""

import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

import inputs
from util import entry, file_record, http_sha256, log, run


def architecture():
    return platform.machine()


APT_HISTORY = "/var/log/apt/history.log"


def installed_apt():
    """{package: (version, architecture, source package, source version)} for the installed packages."""
    out = run(["dpkg-query", "-W", "-f",
               "${Package}\t${Version}\t${Architecture}\t${source:Package}\t${source:Version}\n"])
    return {line.split("\t")[0]: tuple(line.split("\t")[1:]) for line in out.splitlines()}


def parse_apt_history(text):
    """{package: (command line, installed automatically)} for the first apt transaction that
    installed or upgraded each package, from /var/log/apt/history.log."""
    first = {}
    for block in text.split("\n\n"):
        command = re.search(r"^Commandline: (.*)$", block, re.M)
        if not command:
            continue
        for action in ("Install", "Upgrade"):
            line = re.search(r"^%s: (.*)$" % action, block, re.M)
            if line:
                for name, details in re.findall(r"(\S+?):\w+ \(([^)]*)\)", line.group(1)):
                    first.setdefault(name, (command.group(1), "automatic" in details))
    return first


def attribute_apt(added, history, roots, command_categories):
    """{package: (categories, declared_by, installed automatically)} for the packages added on top of
    the base image. The apt history says which command installed a package. The packages that command
    names, when we name them in the Dockerfile or a devcontainer feature, give the categories and
    the declaring line; `command_categories` (categories.json) covers commands we do not write,
    such as the one the nodesource setup script runs."""
    result = {}
    for name in added:
        if name not in history:
            result[name] = (["build"], ["(installed in the image, not in the apt history)"], False)
            continue
        command, automatic = history[name]
        override = next((cats for pattern, cats in command_categories.items() if pattern in command), None)
        named = [root for root in roots if root in command.split()] if override is None else []
        categories = override or sorted({c for root in named for c in roots[root]["categories"]}) or ["build"]
        declared = sorted({line for root in named for line in roots[root]["declared_by"]} | {"apt history: " + command})
        result[name] = (categories, declared, automatic)
    return result


def apt_dependencies(repo, base_packages):
    """The apt packages the image has that its base image does not, or has in another version (an
    install pulled in an upgrade), with the .deb of each. `base_packages` is {name: version}."""
    arch = architecture()
    added = {name: info for name, info in installed_apt().items() if base_packages.get(name) != info[0]}
    with open(APT_HISTORY) as f:
        history = parse_apt_history(f.read())
    commands = {pattern: cats for pattern, cats in inputs.config()["apt_commands"].items() if not pattern.startswith("_")}
    attribution = attribute_apt(added, history, inputs.apt_roots(repo), commands)
    pins = ["%s=%s" % (name, added[name][0]) for name in sorted(added)]
    urls, hashes = _apt_files(pins)
    found = []
    for name in sorted(added):
        version, package_arch, source, source_version = added[name]
        cats, declared, automatic = attribution[name]
        note = "Installed automatically, as a dependency." if automatic else None
        if name in base_packages:
            note = (note + " " if note else "") + "Upgraded from %s in the base image." % base_packages[name]
        if name not in urls:
            note = (note + " " if note else "") + "No download URL: the version is no longer in the package index."
        files = [file_record(urls[name], platform=arch, filename=os.path.basename(urls[name]),
                             digest=("sha256", hashes[name], "apt package index") if name in hashes else None)] if name in urls else []
        found.append(entry(name, version, "apt", cats, files, declared,
                           (note + " " if note else "") + "Source package %s %s." % (source, source_version)))
    return found


def _apt_files(pins):
    """{package: .deb URL} and {package: sha256} for exact package=version pins, from the index."""
    run(["apt-get", "update", "-qq"], check=False)
    urls, hashes = {}, {}
    for start in range(0, len(pins), 40):
        batch = pins[start:start + 40]
        result = subprocess.run(["apt-get", "download", "--print-uris"] + batch, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        lines = result.stdout.splitlines() if result.returncode == 0 else []
        if result.returncode != 0:
            for pin in batch:  # some version may have left the index; resolve one at a time
                single = subprocess.run(["apt-get", "download", "--print-uris", pin], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, universal_newlines=True)
                if single.returncode == 0:
                    lines += single.stdout.splitlines()
        for line in lines:
            match = re.match(r"'([^']+)'\s+(\S+)", line)
            if match:
                name = os.path.basename(match.group(1)).split("_")[0]
                urls[name] = match.group(1)
        shown = subprocess.run(["apt-cache", "show"] + batch, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True).stdout
        for stanza in shown.split("\n\n"):
            package = re.search(r"^Package: (\S+)", stanza, re.M)
            checksum = re.search(r"^SHA256: (\S+)", stanza, re.M)
            if package and checksum:
                hashes[package.group(1)] = checksum.group(1)
    return urls, hashes


def pip_dependencies(repo):
    """The pip packages each use resolves to, as wheels. The resolver runs on this image, so the
    result is for this Python (3.8) and architecture. The `run` group is also resolved for aarch64,
    because users install our aarch64 wheel with their own pip."""
    arch = architecture()
    found = {}
    for group, details in inputs.pip_groups(repo).items():
        if not details["specs"]:
            continue
        targets = [(arch, [])]
        if group == "run":
            targets += [(other, aarch64_pip_tags()) for other in ("aarch64",) if other != arch]
        for target, flags in targets:
            log("pip: resolving the %s group for %s (%d requirements)" % (group, target, len(details["specs"])))
            for item in _pip_report(details, flags):
                name = item["metadata"]["name"].lower().replace("_", "-")
                url = item["download_info"]["url"]
                sha256 = item["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
                record = found.setdefault((name, item["metadata"]["version"]),
                                          {"categories": set(), "declared_by": set(), "files": {}})
                record["categories"].add(group)
                if item.get("requested"):
                    record["declared_by"].update(details["declared_by"].get(inputs.spec_name(name), []))
                wheel = os.path.basename(url)
                record["files"][url] = file_record(url, platform="any" if wheel.endswith("-any.whl") else target,
                                                   filename=wheel,
                                                   digest=("sha256", sha256, "pip resolver report") if sha256 else None)
    return [entry(name, version, "pip", sorted(r["categories"]), list(r["files"].values()),
                  sorted(r["declared_by"]) or ["(dependency of a declared package)"])
            for (name, version), r in sorted(found.items())]


def aarch64_pip_tags(version_info=None):
    """pip options that make it resolve for aarch64 on this machine. The Python version is the one of
    the interpreter that runs, so the x86_64 and aarch64 sets always use the same Python."""
    major, minor = (version_info or sys.version_info)[:2]
    return ["--python-version", "%d.%d" % (major, minor), "--implementation", "cp", "--abi", "cp%d%d" % (major, minor),
            "--abi", "abi3", "--abi", "none",
            "--platform", "manylinux_2_28_aarch64", "--platform", "manylinux_2_17_aarch64",
            "--platform", "manylinux2014_aarch64", "--platform", "linux_aarch64"]


def _pip_report(details, flags):
    """The `install` list of pip's resolver report for a set of requirements, wheels only."""
    with tempfile.TemporaryDirectory() as work:
        report = os.path.join(work, "report.json")
        command = ["python3", "-m", "pip", "install", "--dry-run", "--ignore-installed", "--only-binary", ":all:",
                   "--quiet", "--report", report, "--target", os.path.join(work, "target")] + flags
        for index in details["extra_index_urls"]:
            command += ["--extra-index-url", index]
        run(command + details["specs"])
        with open(report) as f:
            return json.load(f)["install"]


def npm_dependencies(repo):
    """The npm packages we install and everything they pull in, from a lock file npm writes
    without installing anything."""
    found = []
    for root, info in inputs.npm_roots(repo).items():
        log("npm: resolving " + root)
        with tempfile.TemporaryDirectory() as work:
            run(["npm", "install", "--package-lock-only", "--ignore-scripts", "--no-audit", "--no-fund", root], cwd=work)
            with open(os.path.join(work, "package-lock.json")) as f:
                packages = json.load(f)["packages"]
        for path, details in sorted(packages.items()):
            if not path or "resolved" not in details:
                continue
            name = path.split("node_modules/")[-1]
            integrity = details.get("integrity", "")
            digest = None
            if integrity.startswith("sha512-"):
                digest = ("sha512", base64.b64decode(integrity[7:]).hex(), "npm lock file integrity")
            found.append(entry(name, details["version"], "npm", info["categories"],
                               [file_record(details["resolved"], digest=digest)],
                               info["declared_by"] if name == root else ["(dependency of %s)" % root],
                               "Optional dependency." if details.get("optional") else None))
    return found


# --- Bazel -----------------------------------------------------------------------------------------

REPOSITORY_RULES = ("http_archive", "http_file", "http_jar", "git_repository", "new_git_repository", "go_repository")


def _bazel(workspace, *args):
    return run(["bazel"] + list(args) + ["--noenable_bzlmod"], cwd=workspace)


def bazel_dependencies(repo, workspace):
    """The repositories Bazel fetches to build and test everything, with the URL and hash of each.

    `workspace` is a writable copy of the repository: Bazel writes into it. The set is what
    `bazel fetch //...` leaves in the output base, so it is what the build and the tests need."""
    cpp = os.path.join(workspace, "cpp")
    for config in ("x86_64", "aarch64"):  # some dependencies are selected per target architecture
        log("bazel: fetching every repository for %s (this takes a while)" % config)
        _bazel(cpp, "fetch", "--config=" + config, "//...:all")
    output_base = run(["bazel", "info", "output_base", "--noenable_bzlmod"], cwd=cpp).strip()
    fetched = {name for name in os.listdir(os.path.join(output_base, "external")) if not name.endswith(".marker")}
    definitions = _bazel(cpp, "query", "--output=build", "//external:*")
    found = []
    for match in re.finditer(r"^(\w+)\(\n((?:  .*\n)*?)\)$", definitions, re.M):
        rule, attrs = match.group(1), _attributes(match.group(2))
        name = attrs.get("name")
        if rule not in REPOSITORY_RULES or name not in fetched:
            continue
        note = "Bazel repository %s." % name
        declared = ["cpp/WORKSPACE"]
        if rule == "go_repository":
            module, version = attrs.get("importpath", name), attrs.get("version") or attrs.get("commit", "unversioned")
            found.append(entry(module, version, "go", ["build"],
                               [file_record("https://proxy.golang.org/%s/@v/%s.zip" % (module, version),
                                            digest=("h1", attrs["sum"], "go.sum hash in the repository rule") if attrs.get("sum") else None)],
                               declared, note))
        elif rule in ("git_repository", "new_git_repository"):
            found.append(entry(name, attrs.get("tag") or attrs.get("commit", "unversioned")[:12], "bazel", ["build"],
                               [file_record(attrs.get("remote", ""), commit=attrs.get("commit"))], declared, note))
        else:
            urls = attrs.get("urls") or ([attrs["url"]] if attrs.get("url") else [])
            if urls:
                primary = next((url for url in urls if "mirror" not in url), urls[0])  # upstream first, mirrors after
                found.append(entry(name, _version_from(primary, attrs.get("strip_prefix", "")), "bazel", ["build"],
                                   [file_record(primary, mirrors=[url for url in urls if url != primary],
                                                digest=("sha256", attrs["sha256"], "WORKSPACE rule") if attrs.get("sha256") else None)],
                                   declared, note))
    return found


def _attributes(block):
    """name = value pairs of a repository rule as printed by `bazel query --output=build`."""
    attrs = {}
    for key, value in re.findall(r"^\s+(\w+) = (\[[^\]]*\]|\"[^\"]*\"|\S+),?$", block, re.M):
        if value.startswith("["):
            attrs[key] = re.findall(r"\"([^\"]*)\"", value)
        elif value.startswith("\""):
            attrs[key] = value.strip("\"")
    return attrs


def _version_from(url, strip_prefix):
    for text in (strip_prefix, os.path.basename(url)):
        match = re.search(r"v?(\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)+[\w.]*?)(?:\.(?:tar\.gz|tar\.xz|zip|tgz)|$|/)", text)
        if match:
            return match.group(1)
    match = re.search(r"/([0-9a-f]{40})\.", url)
    return match.group(1)[:12] if match else "unversioned"


def devcontainer_tools(repo, compute_hashes=True):
    """bazelisk and buildifier, which the bazel devcontainer feature installs. The release files
    carry no checksum next to them, so the hash is computed from the download."""
    arch = architecture()
    suffix = "arm64" if arch == "aarch64" else "amd64"
    found = []
    for tool, repository, pattern in (("bazelisk", "bazelbuild/bazelisk", r"v(\d+\.\d+\.\d+)"),
                                      ("buildifier", "bazelbuild/buildtools", r"v?(\d+\.\d+\.\d+)")):
        path = shutil.which(tool)
        if not path:
            continue
        output = subprocess.run([path, "--version" if tool == "buildifier" else "version"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True).stdout
        match = re.search(pattern, output)
        if not match:
            continue
        version = match.group(1)
        url = "https://github.com/%s/releases/download/v%s/%s-linux-%s" % (repository, version, tool, suffix)
        digest = ("sha256", http_sha256(url), "computed from the download") if compute_hashes else None
        found.append(entry(tool, version, "url", ["build"], [file_record(url, platform=arch, digest=digest)],
                           [inputs.DEVCONTAINER],
                           "Installed by the bazel devcontainer feature; version read from the installed binary."))
    return found
