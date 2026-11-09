# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect framework dependencies while installing candidate Streamer wheels."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys


CANDIDATES = (
    "runai-model-streamer", "runai-model-streamer-s3",
    "runai-model-streamer-gcs", "runai-model-streamer-azure",
)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def snapshot(directory):
    baseline = {
        canonical(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions()
        if canonical(dist.metadata["Name"]) not in CANDIDATES
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "framework-baseline.json").write_text(json.dumps(baseline, indent=2))
    (directory / "framework-constraints.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(baseline.items()))
    )


def verify(directory, package_version):
    baseline = json.loads((directory / "framework-baseline.json").read_text())
    for name, version in baseline.items():
        actual = metadata.version(name)
        if actual != version:
            raise ValueError(f"Framework dependency changed: {name}: {version} -> {actual}")
    for name in CANDIDATES:
        actual = metadata.version(name)
        if actual != package_version:
            raise ValueError(f"Wrong candidate version: {name}=={actual}")


def install(directory, wheel_directory, package_version):
    wheels = sorted(wheel_directory.rglob("*.whl"))
    if len(wheels) != len(CANDIDATES):
        raise ValueError(f"Expected four candidate wheels, found {len(wheels)}")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--constraint",
        str(directory / "framework-constraints.txt"),
        *(str(wheel) for wheel in wheels),
    ], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    verify(directory, package_version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("snapshot", help="Record framework versions and constraints")
    baseline.add_argument("--directory", type=Path, default=Path("."))
    candidate = commands.add_parser("install", help="Install and verify candidate wheels")
    candidate.add_argument("--directory", type=Path, default=Path("."))
    candidate.add_argument("--wheel-directory", type=Path, required=True)
    candidate.add_argument("--package-version", required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.directory)
    else:
        install(args.directory, args.wheel_directory, args.package_version)


if __name__ == "__main__":
    main()
