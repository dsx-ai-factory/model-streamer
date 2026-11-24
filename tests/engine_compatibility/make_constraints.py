#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print pip constraints: the installed versions of a wheel's direct dependencies.

    make_constraints.py WHEEL > constraints.txt

Reads the dependency names from the wheel's METADATA and keeps the matching
`name==version` lines of `pip freeze` (the streamer itself excluded). A direct
dependency that is not installed, or not pinned with `==` (editable, name @ url),
cannot be constrained and is reported on stderr. Exits non-zero on any failure,
so the caller never mistakes a broken run for an empty constraint set.
"""

import re
import subprocess
import sys
import zipfile


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def direct_dependencies(wheel_path: str) -> set:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        lines = wheel.read(metadata).decode(errors="replace").splitlines()
    names = set()
    for line in lines:
        if line.lower().startswith("requires-dist:"):
            match = re.match(r"\s*([A-Za-z0-9._-]+)", line.split(":", 1)[1])
            if match:
                names.add(canonical(match.group(1)))
    names.discard("runai-model-streamer")
    return names


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} WHEEL", file=sys.stderr)
        sys.exit(2)
    names = direct_dependencies(sys.argv[1])
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    constrained = set()
    for line in frozen.splitlines():
        if "==" in line and canonical(line.split("==")[0].strip()) in names:
            constrained.add(canonical(line.split("==")[0].strip()))
            print(line)
    for name in sorted(names - constrained):
        print(f"unconstrained direct dependency: {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
