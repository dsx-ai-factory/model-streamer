# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the build and the tests ask for, read from the repository files. Nothing here uses the
network. Python 3.8 compatible."""

import ast
import glob
import json
import os
import re

import dockerfile
from util import HERE, read_json

DOCKERFILE = ".devcontainer/Dockerfile"
DEVCONTAINER = ".devcontainer/devcontainer.json"


def config():
    return read_json(os.path.join(HERE, "categories.json"))


def categories_for(kind, name):
    section = config()[kind]
    return list(section.get(name, section["default"]))


def read(repo, path):
    """Text of a file; `path` is relative to the repository or absolute."""
    with open(os.path.join(repo, path)) as f:
        return f.read()


def apt_roots(repo):
    """{package: {"categories": [...], "declared_by": [...]}} for the apt packages we name: the
    Dockerfile `apt-get install` lines and the apt-packages devcontainer feature."""
    roots = {}
    for line, packages in dockerfile.apt_packages(read(repo, DOCKERFILE)):
        for package in packages:
            roots.setdefault(package, {"categories": categories_for("apt", package), "declared_by": []})
            roots[package]["declared_by"].append("%s:%d" % (DOCKERFILE, line))
    for name, options in devcontainer_features(repo).items():
        if name.startswith("ghcr.io/rocker-org/devcontainer-features/apt-packages"):
            for package in options.get("packages", "").split(","):
                if package:
                    roots.setdefault(package, {"categories": ["build"], "declared_by": []})
                    roots[package]["declared_by"].append(DEVCONTAINER)
    return roots


def devcontainer_features(repo):
    """{feature reference: options} from devcontainer.json (which allows // comments)."""
    text = "\n".join(l for l in read(repo, DEVCONTAINER).splitlines() if not l.strip().startswith("//"))
    return json.loads(text).get("features", {})


def setup_requirements(path):
    """The install_requires list of a setup.py, read without running it."""
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "setup":
            for keyword in node.keywords:
                if keyword.arg == "install_requires":
                    return list(ast.literal_eval(keyword.value))
    return []


def spec_name(spec):
    """Normalized package name of a requirement such as `Azure_Identity>=1.2`: azure-identity."""
    return re.sub(r"[-_.]+", "-", re.split(r"[<>=!~\[; ]", spec, 1)[0]).lower()


def pip_groups(repo):
    """The pip requirements by use: {"run": g, "test": g, "build": g}, g = {specs, declared_by,
    extra_index_urls}; declared_by maps each normalized package name to the files that ask for it.
    `run` is what the wheels declare in install_requires, which the user's pip
    installs; `test` and `build` are what the devcontainer installs."""
    groups = {name: {"specs": [], "declared_by": {}, "extra_index_urls": []} for name in ("run", "test", "build")}

    def add(group, spec, declared_by):
        if spec not in groups[group]["specs"]:
            groups[group]["specs"].append(spec)
        files = groups[group]["declared_by"].setdefault(spec_name(spec), [])
        if declared_by not in files:
            files.append(declared_by)

    for setup in sorted(glob.glob(os.path.join(repo, "py", "*", "setup.py"))):
        for spec in setup_requirements(setup):
            add("run", spec, os.path.relpath(setup, repo))
    for line, requirements in dockerfile.pip_packages(read(repo, DOCKERFILE)):
        for spec in requirements:
            name = re.split(r"[<>=!~\[; ]", spec, 1)[0]
            for category in categories_for("pip", name):
                add(category, spec, "%s:%d" % (DOCKERFILE, line))
    for requirements in sorted(glob.glob(os.path.join(repo, "py", "*", "requirements*"))):
        for line in read(repo, requirements).splitlines():
            spec = line.split("#")[0].strip()
            if spec:
                add("test", spec, os.path.relpath(requirements, repo))
    for makefile in sorted(glob.glob(os.path.join(repo, "py", "*", "Makefile"))):
        for name, index in re.findall(r"pip download (\S+) --index-url (\S+)", read(repo, makefile)):
            add("test", name, os.path.relpath(makefile, repo))
            if index not in groups["test"]["extra_index_urls"]:
                groups["test"]["extra_index_urls"].append(index)
        for command in re.findall(r"pip3? install\s+([^&\\\n]*)", read(repo, makefile)):
            skip = False
            for token in command.split():
                if skip or token.startswith("-r"):
                    skip = token == "-r"  # the name after -r is a requirements file, listed elsewhere
                elif not token.startswith(("-", ".", "$")) and "/" not in token:  # a package name, not a path or a wheel of ours
                    add("test", token, os.path.relpath(makefile, repo))
    for pyproject in sorted(glob.glob(os.path.join(repo, "py", "*", "pyproject.toml"))):
        match = re.search(r"requires\s*=\s*\[([^\]]*)\]", read(repo, pyproject))
        for spec in re.findall(r"[\"']([^\"']+)[\"']", match.group(1) if match else ""):
            add("build", spec, os.path.relpath(pyproject, repo))
    return groups


def npm_roots(repo):
    roots = {}
    for line, packages in dockerfile.npm_packages(read(repo, DOCKERFILE)):
        for package in packages:
            roots[package] = {"categories": categories_for("npm", package), "declared_by": ["%s:%d" % (DOCKERFILE, line)]}
    return roots


def bazel_version(repo):
    return read(repo, "cpp/.bazelversion").strip()


def workflow_actions(repo):
    """{(owner/repo[/path], ref): [workflow files]} for every `uses:` that is not a local file."""
    found = {}
    for path in sorted(glob.glob(os.path.join(repo, ".github", "workflows", "*.y*ml"))):
        for match in re.finditer(r"^\s*(?:-\s*)?uses:\s*(\S+)", read(repo, path), re.M):
            reference = match.group(1)
            if reference.startswith("."):
                continue
            action, _, ref = reference.partition("@")
            workflows = found.setdefault((action, ref), [])
            if os.path.relpath(path, repo) not in workflows:
                workflows.append(os.path.relpath(path, repo))
    return found
