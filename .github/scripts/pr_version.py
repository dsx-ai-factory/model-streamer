# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a PR development version from stable tags on its target branch."""

import argparse
import re
import subprocess


STABLE_VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"


def candidate_version(tags, branch, pr_number, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Tested commit must be a full 40-character SHA")
    if pr_number < 1:
        raise ValueError("PR number must be positive")
    release = re.fullmatch(r"release/(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", branch)
    if branch != "master" and not release:
        raise ValueError(f"Unsupported PR target branch: {branch}")
    versions = [tuple(map(int, match.groups())) for tag in tags
                if (match := re.fullmatch(STABLE_VERSION, tag))]
    if release:
        line = tuple(map(int, release.groups()))
        versions = [version for version in versions if version[:2] == line]
    if not versions:
        raise ValueError(f"No stable release tags reachable from target branch {branch}")
    major, minor, patch = max(versions)
    if release:
        patch += 1
    else:
        minor += 1
        patch = 0
    return f"{major}.{minor}.{patch}.dev{pr_number}+g{commit[:7]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-branch", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", args.base_sha):
            raise ValueError("Target commit must be a full 40-character SHA")
        tags = subprocess.check_output(
            ["git", "tag", "--merged", args.base_sha], text=True).splitlines()
        version = candidate_version(tags, args.base_branch, args.pr_number, args.commit)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"PR version error: {error}\n")
    print(f"PACKAGE_VERSION={version}")


if __name__ == "__main__":
    main()
