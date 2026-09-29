# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch verified upstream license texts and regenerate THIRD_PARTY_NOTICES.txt."""

import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
INTRO = """Model Streamer third-party notices
=================================

NVIDIA-authored code is licensed under Apache-2.0; see LICENSE.
The third-party components below retain their respective upstream terms.
Original license and NOTICE texts follow, with source revisions and URLs.
Opening source comments are included where needed for copyright attribution.

Scope
-----
Native entries cover dependencies selected by .devcontainer/Dockerfile,
cpp/WORKSPACE, and their dependency manifests, including transitive components.
Not every component is linked into every backend or build configuration.
The glibc entry covers code incorporated into cpp/cc/portability/portability.c.
The GCC runtime entry is a toolchain reference; verify the actual compiler
version for each release.

Python runtime entries provide reference notices for the direct dependencies
declared in setup.py. They are installed separately by pip, are not bundled in
Model Streamer wheels, and these reference revisions do not pin their versions.
Those distributions provide their own licenses and transitive notices. Anyone
redistributing an environment must inventory its exact resolved distributions,
including their transitive dependencies and any bundled native libraries.
Test-only entries are not linked into release libraries.

This inventory is not a binary scan or release approval. Reconcile it with the
actual release artifacts and configuration before distribution. In particular,
LGPL source/relinking obligations are not satisfied by notices alone. See
docs/release-compliance.md for the maintainer checklist.

"""


def opening_comment(text, python_style=False):
    """Keep the contiguous opening comment blocks, without upstream code."""
    lines = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_block:
            lines.append(line)
            in_block = "*/" not in line
        elif not stripped or stripped.startswith("//") or (
            python_style and stripped.startswith("#")
        ):
            lines.append(line)
        elif stripped.startswith("/*"):
            lines.append(line)
            in_block = "*/" not in line
        else:
            break
    result = "\n".join(lines).strip()
    if not result:
        raise ValueError("No opening attribution comment found")
    return result + "\n"


def fetch_notice(record):
    request = urllib.request.Request(
        record["url"], headers={"User-Agent": "model-streamer-license-notices"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if "archive_member" in record:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            data = archive.extractfile(record["archive_member"]).read()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Upstream content changed: " + record["url"])
    text = data.decode("utf-8")
    if record.get("opening_comment_only"):
        try:
            return opening_comment(text, record["path"].endswith(".py"))
        except ValueError as error:
            raise ValueError(record["url"] + ": " + str(error)) from error
    return text


def main():
    manifest = json.loads((ROOT / "licensing/third_party_manifest.json").read_text())
    records = [record for component in manifest["components"] for record in component["files"]]
    # Collect everything before writing so failed downloads cannot truncate notices.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        texts = iter(executor.map(fetch_notice, records))
        sections = [INTRO]
        for component in manifest["components"]:
            sections.append(
                "\n" + "=" * 78 + "\n"
                + component["name"] + "\n"
                + "Source: " + component["repository"] + "\n"
                + "Revision: " + component["revision"] + "\n"
                + "Scope: " + component["scope"] + "\n"
                + "License: " + component["license"] + "\n"
            )
            for record in component["files"]:
                sections.append(
                    "\n--- " + record["path"] + " ---\n"
                    + "Retrieved from: " + record["url"] + "\n"
                    + "SHA-256 (upstream file): " + record["sha256"] + "\n\n"
                    + next(texts).rstrip() + "\n"
                )
    (ROOT / "THIRD_PARTY_NOTICES.txt").write_text("".join(sections), encoding="utf-8")
    print(f"Generated notices for {len(manifest['components'])} components.")


if __name__ == "__main__":
    main()
