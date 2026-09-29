# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check source attribution and license inclusion in Model Streamer wheels."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".cc", ".h", ".py", ".bzl", ".cmake", ".ldscript", ".BUILD"}
SOURCE_NAMES = {"BUILD", "BUILD.tpl", "WORKSPACE", "Makefile", "Dockerfile", "requirements.dev"}
PACKAGES = (
    "runai_model_streamer", "runai_model_streamer_s3",
    "runai_model_streamer_gcs", "runai_model_streamer_azure",
)


def check_sources():
    errors = []
    paths = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    checked = 0
    for name in filter(None, paths):
        path = ROOT / name
        if path.is_symlink() or (
            path.suffix not in SOURCE_SUFFIXES and path.name not in SOURCE_NAMES
        ):
            continue
        content = path.read_text()
        if not content.strip():
            continue
        checked += 1
        header = "\n".join(content.splitlines()[:60])
        if name == "cpp/cc/portability/portability.c":
            required = ("Free Software Foundation, Inc.", "GNU Lesser General Public",
                        "NVIDIA CORPORATION & AFFILIATES.", "Modifications:")
            if not all(item in header for item in required):
                errors.append(f"{name}: missing mixed-origin attribution")
            continue
        if "Copyright" not in header:
            errors.append(f"{name}: missing copyright attribution")
        if "NVIDIA" in header:
            if not re.search(
                r"SPDX-FileCopyrightText: Copyright \(c\) \d{4}(?:-\d{4})? "
                r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.", header
            ) or "SPDX-License-Identifier: Apache-2.0" not in header:
                errors.append(f"{name}: incorrect NVIDIA Apache-2.0 header")
        elif not any(marker in header for marker in (
            "Licensed under", "Permission is hereby granted",
            "Redistribution and use", "free software",
        )):
            errors.append(f"{name}: preserve the upstream license header for third-party code")
    print(f"Checked attribution in {checked} source/build files.")
    return errors


def check_packages(wheels):
    errors = []
    for filename in ("LICENSE", "THIRD_PARTY_NOTICES.txt"):
        expected = (ROOT / filename).read_bytes()
        if not expected.strip():
            errors.append(f"{filename}: empty license material")
        for package in PACKAGES:
            path = ROOT / "py" / package / filename
            if not path.exists() or path.read_bytes() != expected:
                errors.append(f"{package}: {filename} differs from root")
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                matches = [
                    name for name in archive.namelist()
                    if ".dist-info/" in name and name.rsplit("/", 1)[-1] == filename
                ]
                if len(matches) != 1 or archive.read(matches[0]) != expected:
                    errors.append(f"{wheel}: missing or incorrect {filename} in wheel metadata")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", nargs="+", default=[])
    args = parser.parse_args()
    errors = check_sources() + check_packages(args.wheels)
    dco = (ROOT / "DCO").read_text().strip()
    if dco not in (ROOT / "CONTRIBUTING.md").read_text():
        errors.append("CONTRIBUTING.md must contain the full DCO")
    manifest = json.loads((ROOT / "licensing/third_party_manifest.json").read_text())
    notices = (ROOT / "THIRD_PARTY_NOTICES.txt").read_text()
    for component in manifest["components"]:
        if "\n" + component["name"] + "\n" not in notices:
            errors.append(f"Missing third-party component: {component['name']}")
        for record in component["files"]:
            if record["sha256"] not in notices:
                errors.append(f"Missing notice provenance: {record['url']}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"License checks passed; verified {len(args.wheels)} wheel(s).")


if __name__ == "__main__":
    main()
