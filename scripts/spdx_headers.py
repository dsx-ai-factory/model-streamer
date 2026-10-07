#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check, or add, the NVIDIA SPDX header on source files.

    scripts/spdx_headers.py               check all tracked files
    scripts/spdx_headers.py FILE...       check only these files
    scripts/spdx_headers.py --fix [FILE...]   add the header where it is missing

FILE can be an absolute path, or a path relative to the folder you run the script from.
Run it inside the git repository. Exit code is 1 when a file is missing the header
(or has an unknown file type).
"""

import argparse
import os
import subprocess
import sys

# The header text. The check looks for these exact lines, so the year must match the one in LICENSE.
COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
LICENSE = "SPDX-License-Identifier: Apache-2.0"

# A header only counts when it is within the first WINDOW lines of the file.
WINDOW = 20

# Files that need no header (the OSRB exempt list): docs, config, data, images and license files.
# A name is exempt if it ends with one of the suffixes or is one of the exact names. A name that
# starts with a dot (.gitignore, .bazelrc) is tool config, so it is exempt too, as in the OSRB scanner.
EXEMPT_SUFFIXES = (".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
                   ".ipynb", ".png", ".jpg", ".svg", ".safetensors", ".so")
EXEMPT_NAMES = ("LICENSE", "NOTICE", "DCO", "CODEOWNERS")

# Files to skip on purpose, for example if OSRB says a file must not get the header. Empty today.
# Write each path relative to the repo root, such as "cpp/s3/s3.cc".
SKIP = []

# Comment syntax per file type. A file type in neither list is reported, never skipped silently.
C_STYLE = (".c", ".cc", ".cpp", ".h", ".hpp", ".ldscript")
HASH_STYLE_EXT = (".py", ".sh", ".bzl", ".cmake", ".dev", ".tpl", ".BUILD")
HASH_STYLE_NAMES = ("Makefile", "WORKSPACE", "BUILD", "Dockerfile")


def style_of(path):
    name = os.path.basename(path)
    if name.endswith(C_STYLE):
        return "c"
    if name.endswith(HASH_STYLE_EXT) or name in HASH_STYLE_NAMES or name.startswith(("BUILD.", "Dockerfile.")):
        return "hash"
    return None


def header_for(style):
    # The exact two lines of the OSRB Apache 2.0 template. C-style is wrapped in /* */.
    if style == "c":
        return ["/*", " * " + COPYRIGHT, " * " + LICENSE, " */"]
    return ["# " + COPYRIGHT, "# " + LICENSE]


def has_header(text):
    # Keep only the first WINDOW lines: split the text into lines, slice, and join them back into
    # one string. A header deep in the file does not count.
    top = "\n".join(text.splitlines()[:WINDOW])
    # "in" is a substring search, so it matches with or without the "# " or " * " comment prefix.
    return COPYRIGHT in top and LICENSE in top


def add_header(text, style):
    # The header is followed by a blank line. If the file already starts with a blank line, that
    # gives two blank lines, which is harmless.
    header = "\n".join(header_for(style)) + "\n\n"
    # A shebang line (#!/usr/bin/env python3) must stay first, so the header goes after it.
    if text.startswith("#!"):
        shebang, rest = text.split("\n", 1)
        return shebang + "\n" + header + rest
    return header + text


def main():
    # RawDescriptionHelpFormatter keeps the line breaks of the docstring in --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    # Find the top folder of the git repo, so the script works from any folder inside it.
    # "git rev-parse --show-toplevel" prints that folder. It fails the script outside a git repo.
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True).strip()
    if args.files:
        # Files may be absolute or relative to the current folder. abspath() makes both absolute, and
        # relpath() makes them relative to the repo root.
        files = [os.path.relpath(os.path.abspath(f), root) for f in args.files]
    else:
        # Tracked files only, so build folders and ignored files are never checked. -z separates the
        # names with NUL, so a name with a space or a newline is read correctly.
        out = subprocess.check_output(["git", "-C", root, "ls-files", "-z"])
        files = [f for f in out.decode().split("\0") if f]

    missing, unknown, fixed = [], [], []
    # From here on, rel is always relative to the repo root, however the file was given.
    for rel in files:
        full = os.path.join(root, rel)
        name = os.path.basename(rel)
        if name.endswith(EXEMPT_SUFFIXES) or name.startswith(".") or name in EXEMPT_NAMES or rel in SKIP:
            continue
        # Symlinks (the *.so links and py/*/LICENSE) and anything that is not a regular file.
        if os.path.islink(full) or not os.path.isfile(full):
            continue
        with open(full, "rb") as f:
            data = f.read()
        # Empty and binary files.
        if not data.strip() or b"\0" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        if has_header(text):
            continue
        style = style_of(rel)
        if style is None:
            unknown.append(rel)
        elif args.fix:
            with open(full, "w") as f:
                f.write(add_header(text, style))
            fixed.append(rel)
        else:
            missing.append(rel)

    for rel in unknown:
        print("unknown file type (edit scripts/spdx_headers.py): " + rel)
    for rel in missing:
        print("missing SPDX header (run scripts/spdx_headers.py --fix): " + rel)
    for rel in fixed:
        print("fixed   " + rel)
    if fixed:
        print("Fixed %d file(s)." % len(fixed))
    return 1 if (missing or unknown) else 0


if __name__ == "__main__":
    sys.exit(main())
