# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find what the collector would miss, without the network or Docker. Three checks:

* a tracked file that downloads or installs something but is not read by the collector;
* a file the collector reads for some commands (a Makefile for `pip install`, a workflow for `uses:`)
  that also holds a command it does not read;
* a build file (the Dockerfile and its scripts) in which a download, `git clone` or install command
  did not become a dependency, or that uses `wget`, which the readers do not support.

All fail loudly so that a new script, or a command in a new shape, is noticed when it is added,
not when a list turns out to be incomplete. Python 3.8 compatible."""

import fnmatch
import re

import dockerfile
import inputs
from util import run

# The files the collector reads, as globs from the repository root (`*` also matches `/`).
READ_BY_THE_COLLECTOR = [
    ".devcontainer/Dockerfile",
    ".devcontainer/requirements.txt",
    ".devcontainer/scripts/*.sh",
    ".devcontainer/devcontainer.json",
    ".github/workflows/*",
    "py/*/setup.py",
    "py/*/pyproject.toml",
    "py/*/requirements*",
    "py/*/Makefile",
]

# The commands that download or install something, by kind. A URL is part of `curl` and `wget`
# because a `curl` that has none (a health check) does not download a dependency.
COMMANDS = {
    "curl": r"\bcurl\b[^|;&\n]*?https?://",
    "wget": r"\bwget\b[^\n#]*?https?://",
    "git clone": r"\bgit clone\b",
    "npm install": r"\bnpm (?:install|i)\b",
    "pip install": r"\bpip3? install\b",
    "apt-get install": r"\bapt(?:-get)? install\b",
}

# What the collector understands outside the build files. Every other file it reads (workflows,
# setup.py, devcontainer.json, ...) is read for something that is not a command, so a command in it is not read.
UNDERSTOOD = {"py/*/Makefile": {"pip install"}}

# What makes a file an input of the build, in a file the collector does not read.
DOWNLOADS_SOMETHING = re.compile("|".join(COMMANDS.values()) + r"|^\s*FROM\s|\bhttp_archive\b|\bgit_repository\b|^[\s-]*uses:", re.M)

MAX_BYTES = 2 * 1024 * 1024


def _matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _text(repo, path):
    """The text of a tracked file, or None for a file that is not text."""
    try:
        with open("%s/%s" % (repo, path), "rb") as f:
            return f.read(MAX_BYTES).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _tracked(repo):
    return sorted(run(["git", "ls-files"], cwd=repo).splitlines())


def unread_files(repo):
    """Tracked files that download or install something and that nothing accounts for."""
    config = inputs.config()["coverage"]
    accounted = READ_BY_THE_COLLECTOR + config["covered_by_bazel"] + config["not_an_input"]
    problems = []
    for path in _tracked(repo):
        if _matches(path, accounted):
            continue
        text = _text(repo, path)
        if text is not None and DOWNLOADS_SOMETHING.search(text):
            problems.append("%s downloads or installs something, but the collector does not read it. Teach the collector "
                            "about it, or add it to `coverage` in categories.json if Bazel covers it or it is not an input "
                            "of the build" % path)
    return problems


def unparsed_commands(repo):
    """Commands in the build files that the parsers did not turn into a dependency."""
    problems = []
    for path in inputs.build_files(repo):
        text = inputs.read(repo, path)
        commands = "\n".join(command for _, command in dockerfile.runs(text, path))
        parsed = {
            "curl": len(dockerfile.downloads(text, path)),
            "git clone": len(dockerfile.git_clones(text, path)),
            "npm install": len(dockerfile.npm_packages(text, path)),
            "pip install": len(dockerfile.pip_packages(text, path)),
            "apt-get install": len(dockerfile.apt_packages(text, path)),
        }
        for kind, count in parsed.items():
            found = len(re.findall(COMMANDS[kind], commands))
            if found != count:
                problems.append("%s has %d `%s` commands and the collector read %d. Teach dockerfile.py the new shape"
                                % (path, found, kind, count))
        if re.search(COMMANDS["wget"], commands):
            problems.append("%s downloads with `wget`, which the collector does not read. Use `curl`, or teach dockerfile.py `wget`" % path)
    return problems


def unread_commands(repo):
    """Commands in a file the collector reads, but not for commands: a Makefile is read for `pip install`
    and `pip download`, a workflow for `uses:`. A command of another kind in it is not in the list."""
    config = inputs.config()["coverage"]["allowed_commands"]
    build = set(inputs.build_files(repo))
    problems = []
    for path in _tracked(repo):
        if path in build or not _matches(path, READ_BY_THE_COLLECTOR):
            continue
        text = _text(repo, path)
        if text is None:
            continue
        understood = set()
        for pattern, kinds in UNDERSTOOD.items():
            if fnmatch.fnmatchcase(path, pattern):
                understood |= kinds
        for pattern, allowed in config.items():
            if not pattern.startswith("_") and fnmatch.fnmatchcase(path, pattern):
                understood |= set(allowed["kinds"])
        for kind, regex in COMMANDS.items():
            if kind not in understood and re.search(regex, text):
                problems.append("%s has a `%s` command, which the collector does not read in this kind of file. Teach the collector "
                                "about it, or allow it under `allowed_commands` in categories.json with the reason" % (path, kind))
    return problems


def problems(repo):
    return unread_files(repo) + unread_commands(repo) + unparsed_commands(repo)
