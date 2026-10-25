#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the environment that the GPU job runs in.

    env_check.py newest-torch

The newest-torch leg is meant to test the newest PyTorch. pip installs the newest release that the image's Python can
use, so when a release stops supporting that Python, the leg silently falls back to an older torch and no longer tests
the newest. This asks PyPI for the newest release, compares it with the installed one, and says so when they differ:
the base container of the leg has to change (a newer Ubuntu or Python). The result is printed, written to
env-check.txt in GPU_RESULTS (the workflow shows it in the job summary), and the exit status is 1 when the leg is limited.

If PyPI cannot be reached the check says it could not verify and exits 0: a network problem is not an environment
problem.

Needs only the standard library.
"""

import importlib.metadata
import json
import os
import re
import sys
import urllib.request

PYPI_URL = "https://pypi.org/pypi/torch/json"
TIMEOUT_SECONDS = 30

_STABLE = re.compile(r"^\d+(\.\d+)*$")
_CLAUSE = re.compile(r"^\s*(>=|<=|==|!=|~=|>|<)\s*([0-9]+(?:\.[0-9]+)*)(\.\*)?\s*$")


def version_tuple(text):
    """(2, 14, 1) for "2.14.1"; None for anything that is not a plain release (2.15.0rc1, 2.14.1+cu128 ...)."""
    if not _STABLE.match(text):
        return None
    return tuple(int(part) for part in text.split("."))


def python_allows(requires_python, python_version):
    """Whether a Requires-Python string like ">=3.10,<3.14" allows python_version, a tuple like (3, 10, 12). A clause
    that is not understood is taken as satisfied, so that a new syntax cannot make the check fail by itself."""
    for clause in (requires_python or "").split(","):
        if not clause.strip():
            continue
        found = _CLAUSE.match(clause)
        if not found:
            continue
        op, number, wildcard = found.group(1), tuple(int(p) for p in found.group(2).split(".")), found.group(3)
        head = tuple(python_version[:len(number)])
        if wildcard:
            ok = (head == number) if op == "==" else (head != number)
        elif op == ">=":
            ok = tuple(python_version) >= number
        elif op == ">":
            ok = tuple(python_version) > number
        elif op == "<=":
            ok = tuple(python_version) <= number
        elif op == "<":
            ok = tuple(python_version) < number
        elif op == "==":
            ok = head == number
        elif op == "!=":
            ok = head != number
        else:  # ~=
            ok = tuple(python_version) >= number and tuple(python_version[:len(number) - 1]) == number[:-1]
        if not ok:
            return False
    return True


def releases(pypi_json):
    """(version tuple, version string, requires_python), oldest first, of the plain releases that have a file that is
    not yanked."""
    found = []
    for version, files in pypi_json["releases"].items():
        parsed = version_tuple(version)
        live = [f for f in files if not f.get("yanked")]
        if parsed is None or not live:
            continue
        found.append((parsed, version, live[0].get("requires_python") or ""))
    return sorted(found)


def assess(installed, pypi_json, python_version):
    """(limited, lines): whether the installed torch is not the newest release, and what to tell the reader."""
    all_releases = releases(pypi_json)
    newest = all_releases[-1]
    usable = [r for r in all_releases if python_allows(r[2], python_version)]
    newest_usable = usable[-1]
    python_text = ".".join(str(p) for p in python_version)
    installed_tuple = version_tuple(installed.split("+")[0])
    lines = ["installed torch: %s" % installed,
             "newest torch on PyPI: %s (requires Python %s)" % (newest[1], newest[2] or "any"),
             "newest torch that Python %s can use: %s" % (python_text, newest_usable[1])]
    if installed_tuple == newest[0]:
        lines.append("OK: the newest torch is installed")
        return False, lines
    if newest_usable[0] < newest[0]:
        lines.append("ERROR: this container cannot run the newest torch. torch %s needs Python %s, and this image has "
                     "Python %s, so pip installed torch %s. Change the base container of the newest-torch leg (a "
                     "newer Ubuntu or Python) so that the leg tests the newest torch."
                     % (newest[1], newest[2], python_text, newest_usable[1]))
    else:
        lines.append("ERROR: torch %s is installed, but torch %s is the newest and this Python can use it. Something "
                     "else in the install (a version limit on a dependency?) held torch back."
                     % (installed, newest[1]))
    return True, lines


def newest_torch():
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=TIMEOUT_SECONDS) as response:
            pypi_json = json.load(response)
    except (OSError, ValueError) as error:
        return False, ["could not ask PyPI for the newest torch (%s): not verified" % error]
    return assess(importlib.metadata.version("torch"), pypi_json, sys.version_info[:3])


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args != ["newest-torch"]:
        print("usage: env_check.py newest-torch", file=sys.stderr)
        return 2
    limited, lines = newest_torch()
    text = "\n".join(lines)
    print(text, flush=True)
    results = os.environ.get("GPU_RESULTS")
    if results:
        os.makedirs(results, exist_ok=True)
        with open(os.path.join(results, "env-check.txt"), "w") as handle:
            handle.write(text + "\n")
    return 1 if limited else 0


if __name__ == "__main__":
    sys.exit(main())
