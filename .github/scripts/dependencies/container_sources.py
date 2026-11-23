# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import licenses
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
                           (note + " " if note else "") + "Source package %s %s." % (source, source_version),
                           license=apt_license(name)))
    return found


def apt_license(package):
    """(license, source) from the copyright file the package installs, or None."""
    try:
        with open("/usr/share/doc/%s/copyright" % package, errors="replace") as f:
            return licenses.debian_copyright(f.read())
    except OSError:
        return None


def _apt_files(pins):
    """{package: .deb URL} and {package: sha256} for exact package=version pins, from the index.

    `apt-get update` can fail in part (a mirror that does not answer) and still lets the next commands
    run on a partial index, which leaves packages without a URL for no lasting reason. So a pin with no URL
    is looked up again after another update, up to three times, before it is left without one: a version
    that really left the index stays without."""
    urls, hashes = {}, {}
    pending = list(pins)
    for attempt in range(3):
        run(["apt-get", "update", "-qq"], check=False)
        found_urls, found_hashes = _apt_uris(pending)
        urls.update(found_urls)
        hashes.update(found_hashes)
        pending = [pin for pin in pending if pin.split("=")[0] not in urls]
        if not pending:
            break
        log("apt: %d packages have no download URL after update %d" % (len(pending), attempt + 1))
    return urls, hashes


def _apt_uris(pins):
    """{package: .deb URL} and {package: sha256} for exact package=version pins, from the current index."""
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
    result is for this Python (the image's, 3.11) and architecture. The `run` group is also resolved for aarch64,
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
                                          {"categories": set(), "declared_by": set(), "files": {}, "license": None})
                record["license"] = record["license"] or licenses.pip_license(item["metadata"])
                record["categories"].add(group)
                if item.get("requested"):
                    record["declared_by"].update(details["declared_by"].get(inputs.spec_name(name), []))
                wheel = os.path.basename(url)
                record["files"][url] = file_record(url, platform="any" if wheel.endswith("-any.whl") else target,
                                                   filename=wheel,
                                                   digest=("sha256", sha256, "pip resolver report") if sha256 else None)
    return [entry(name, version, "pip", sorted(r["categories"]), list(r["files"].values()),
                  sorted(r["declared_by"]) or ["(dependency of a declared package)"], license=r["license"])
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
            declared_license = licenses.npm_license(details)
            found.append(entry(name, details["version"], "npm", info["categories"],
                               [file_record(details["resolved"], digest=digest)],
                               info["declared_by"] if name == root else ["(dependency of %s)" % root],
                               "Optional dependency." if details.get("optional") else None,
                               license=(declared_license, "npm lock file (license)") if declared_license else None))
    return found


# --- Bazel -----------------------------------------------------------------------------------------

RULE = re.compile(r"^(\w+)\(\n((?:  .*\n)*?)\)$", re.M)  # a rule as `bazel mod show_repo` prints it
REPOSITORY_RULES = ("http_archive", "http_file", "http_jar", "git_repository", "new_git_repository", "go_repository")
MODULE_SECTION = re.compile(r"^## ([\w.+-]+)@[^:\n]*:$", re.M)  # one per module in `bazel mod show_repo`
MODULE_DECLARATION = re.compile(r"\b(?:bazel_dep|single_version_override|archive_override|git_override)\("
                                r"[^)\n]*?\b(?:name|module_name) = \"([\w.+-]+)\"")


def _bazel(workspace, *args):
    return run(["bazel"] + list(args), cwd=workspace)


def bazel_dependencies(repo, workspace):
    """The repositories Bazel downloads to build and test everything, with the URL and hash of each.

    `workspace` is a writable copy of the repository: Bazel writes into it. Both Bazel modules are
    covered: the application (cpp/) and the devcontainer's SDK builder (.devcontainer/gcs/), whose
    graph is the code of the prebuilt GCS SDK. Dependencies are Bzlmod modules (MODULE.bazel): the
    set is the module repositories `bazel fetch //...` leaves in the output base, so it is what the
    builds and the tests need, and `bazel mod show_repo` says where each one was downloaded from."""
    cpp = os.path.join(workspace, "cpp")
    # Module resolution does not depend on the architecture, so the native configuration fetches
    # the whole set. The other architecture's configuration cannot even load: the toolchain
    # repository declares only the native platform (cross-compilation is not supported).
    log("bazel: fetching every application repository (this takes a while)")
    _bazel(cpp, "fetch", "--config=" + architecture(), "//...:all")
    found = _module_dependencies(cpp, "cpp/MODULE.bazel")
    sdk = os.path.join(workspace, ".devcontainer", "gcs")
    log("bazel: fetching every repository of the GCS SDK builder (this takes a while)")
    _bazel(sdk, "fetch", "//...:all")
    found += _module_dependencies(sdk, ".devcontainer/gcs/MODULE.bazel")
    return found  # collect.py merges the modules both builds share


def _module_dependencies(directory, module_file):
    """The fetched module repositories of the Bazel module at `directory`, named by `module_file`."""
    output_base = _bazel(directory, "info", "output_base").strip()
    fetched = {name for name in os.listdir(os.path.join(output_base, "external")) if not name.endswith(".marker")}
    modules = _module_graph(directory)
    definitions = _bazel(directory, "mod", "show_repo", *sorted(modules))
    declarations = _module_declarations(os.path.join(directory, "MODULE.bazel"))
    found = []
    for module, text in _module_sections(definitions):
        match = RULE.search(text)
        if not match or module not in modules:
            continue
        rule, attrs = match.group(1), _attributes(match.group(2))
        if rule not in REPOSITORY_RULES or attrs.get("name") not in fetched:
            continue
        version, parent = modules[module]
        if module in declarations:
            declared = ["%s:%d" % (module_file, declarations[module])]
        elif parent:
            declared = ["MODULE.bazel of Bazel module %s (Bazel Central Registry)" % parent]
        else:
            declared = [module_file]
        note = "Bazel module %s." % module
        found_license = _repository_license(os.path.join(output_base, "external", attrs["name"]))
        if rule == "go_repository":
            found.append(entry(module, version, "go", ["build"],
                               [file_record("https://proxy.golang.org/%s/@v/%s.zip" % (attrs.get("importpath", module), version),
                                            digest=("h1", attrs["sum"], "go.sum hash in the repository rule") if attrs.get("sum") else None)],
                               declared, note, license=found_license))
        elif rule in ("git_repository", "new_git_repository"):
            found.append(entry(module, version or attrs.get("tag") or attrs.get("commit", "unversioned")[:12], "bazel", ["build"],
                               [file_record(attrs.get("remote", ""), commit=attrs.get("commit"))], declared, note,
                               license=found_license))
        else:
            urls = attrs.get("urls") or ([attrs["url"]] if attrs.get("url") else [])
            if urls:
                primary = next((url for url in urls if "mirror" not in url), urls[0])  # upstream first, mirrors after
                found.append(entry(module, version or _version_from(primary, attrs.get("strip_prefix", "")), "bazel", ["build"],
                                   [file_record(primary, mirrors=[url for url in urls if url != primary],
                                                digest=_module_digest(attrs))],
                                   declared, note, license=found_license))
    return found


def _repository_license(directory):
    """(license, source) from the top-level license files of a repository Bazel fetched, or None."""
    texts = {}
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for name in names:
        path = os.path.join(directory, name)
        if licenses.LICENSE_FILE.match(name) and os.path.isfile(path):
            with open(path, errors="replace") as f:
                texts[name] = f.read(200000)
    found = licenses.repository_license(texts)
    return (found, "license file of the fetched repository, matched by its text") if found else None


def _module_graph(directory):
    """{module: (version, first requiring module or None for the root)} for the selected modules of
    the resolved graph (`bazel mod graph`). The root and modules a local_path_override supplies have
    no version and are not dependencies to report, but their children remember them as the parent."""
    graph = json.loads(_bazel(directory, "mod", "graph", "--output", "json"))
    modules = {}

    def walk(node, parent):
        for dependency in node.get("dependencies", []) + node.get("indirectDependencies", []):
            name, version = dependency["name"], dependency.get("version", "")
            if version and name not in modules:
                modules[name] = (version, parent)
            # A module's subtree appears expanded at one of its occurrences, not necessarily the
            # first; the other occurrences carry no dependencies, so walking every one terminates.
            walk(dependency, name if version else parent)

    walk(graph, None)
    return modules


def _module_declarations(module_file):
    """{module: line number} for the modules the file itself asks for (bazel_dep and overrides)."""
    declarations = {}
    with open(module_file) as f:
        for number, line in enumerate(f, 1):
            match = MODULE_DECLARATION.search(line)
            if match:
                declarations.setdefault(match.group(1), number)
    return declarations


def _module_sections(definitions):
    """(module name, text) per `## <module>@<version>:` section of `bazel mod show_repo`."""
    matches = list(MODULE_SECTION.finditer(definitions))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(definitions)
        yield match.group(1), definitions[match.start():end]


def _module_digest(attrs):
    """The archive hash of a module repository rule: `sha256`, or the SRI `integrity` the Bazel
    Central Registry serves."""
    if attrs.get("sha256"):
        return ("sha256", attrs["sha256"], "Bazel module registry")
    integrity = attrs.get("integrity", "")
    if integrity.startswith("sha256-"):
        return ("sha256", base64.b64decode(integrity[len("sha256-"):]).hex(), "Bazel module registry (integrity)")
    return None


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
