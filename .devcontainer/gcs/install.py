# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a relocatable Bazel package from the SDK's CcInfo manifest."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("SDK path must stay inside the execution root: " + value)
    return path


def install(manifest, execution_root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    archives = {library["path"] for library in manifest["libraries"]}

    def tree_path(value):
        path = str(PurePosixPath("tree") / relative_path(value))
        # cc_import rejects Bazel's .lo/.pic.lo extension. Keep the archive
        # contents and alwayslink flag, but give the installed file an .a suffix.
        return path + ".a" if value in archives and value.endswith(".lo") else path

    paths = set(manifest["headers"])
    paths.update(archives)
    for value in sorted(paths):
        path = relative_path(value)
        target = destination / tree_path(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Dereference Bazel's external-repository and virtual-include symlinks.
        shutil.copyfile(execution_root / path, target)

    rules = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
        "",
    ]
    deps = []
    for index, library in enumerate(manifest["libraries"]):
        name = "archive_" + str(index)
        deps.append(":" + name)
        rules.append(
            "cc_import(\n"
            "    name = %s,\n"
            "    static_library = %s,\n"
            "    alwayslink = %s,\n"
            ")\n" % (
                json.dumps(name),
                json.dumps(tree_path(library["path"])),
                repr(library["alwayslink"]),
            )
        )
    rules.append(
        "cc_library(\n"
        '    name = "sdk",\n'
        "    hdrs = %s,\n"
        "    includes = %s,\n"
        "    defines = %s,\n"
        "    linkopts = %s,\n"
        "    deps = %s,\n"
        ")\n" % (
            json.dumps([tree_path(path) for path in manifest["headers"]]),
            json.dumps([tree_path(path) for path in manifest["includes"]]),
            json.dumps(manifest["defines"]),
            json.dumps(manifest["linkopts"]),
            json.dumps(deps),
        )
    )
    (destination / "BUILD.bazel").write_text("\n".join(rules))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("execution_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    install(json.loads(args.manifest.read_text()), args.execution_root, args.destination)
