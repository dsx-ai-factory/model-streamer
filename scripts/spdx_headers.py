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
# [explain] The docstring above is the --help text. Exit code 1 is what makes CI and pre-commit fail.

import argparse
import os
import subprocess
import sys

# [explain] The header text lives only here. If OSRB asks for different wording, change these two lines.
# [explain] The check looks for these exact lines, so the year must be 2024, the same as in the LICENSE file.
COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
LICENSE = "SPDX-License-Identifier: Apache-2.0"

# [explain] Files that already have the header must show both lines within this many lines of the top.
WINDOW = 20

# [explain] Files that do not need a header (OSRB exempt list): docs, config, data, images, binaries, license files.
# [explain] A name is exempt if it ends with one of these suffixes, starts with a dot (.gitignore), or is one of these exact names.
EXEMPT_SUFFIXES = (".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
                   ".ipynb", ".png", ".jpg", ".svg", ".safetensors", ".so")
EXEMPT_NAMES = ("LICENSE", "NOTICE", "DCO", "CODEOWNERS")

# [explain] Files to skip on purpose. Empty today. Add a path here if OSRB says a file must not get the header.
# [explain] Write each path relative to the REPO ROOT, for example "cpp/s3/s3.cc". Inside the script every file is held in
# [explain] that form (see "rel" in main), so the skip works the same wherever in the repo the script is run from.
SKIP = []

# [explain] Comment syntax per file type. A tracked file that is in neither list and not exempt is reported as an error,
# [explain] so a new file type is never skipped silently.
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
    # [explain] Exact two-line text from the OSRB Apache 2.0 template. C-style is wrapped in /* */.
    if style == "c":
        return ["/*", " * " + COPYRIGHT, " * " + LICENSE, " */"]
    return ["# " + COPYRIGHT, "# " + LICENSE]


def has_header(text):
    # [explain] The trick on the next line, read from the inside out:
    # [explain]   text.splitlines()    cuts the file text into a list of lines: "a\nb\nc" -> ["a", "b", "c"]
    # [explain]   [:WINDOW]            keeps only the first WINDOW (20) items of that list, or all of them if the file is shorter
    # [explain]   "\n".join(...)       glues those lines back into one string, with a newline between them
    # [explain] The result, top, is "the first 20 lines of the file as one string". We search only that part,
    # [explain] so a header must be at the top, and the word "SPDX" somewhere deep in the code does not count.
    top = "\n".join(text.splitlines()[:WINDOW])
    # [explain] "in" is a plain substring search, so it finds the line with or without the "# " or " * " comment prefix.
    return COPYRIGHT in top and LICENSE in top


def add_header(text, style):
    # [explain] The header as one block of text, followed by a blank line.
    # [explain] If the file already starts with a blank line, this gives two blank lines. That is harmless.
    header = "\n".join(header_for(style)) + "\n\n"
    # [explain] A shebang line (#!/usr/bin/env python3) must stay first, so the header goes after it.
    if text.startswith("#!"):
        shebang, rest = text.split("\n", 1)
        return shebang + "\n" + header + rest
    return header + text


def main():
    # [explain] RawDescriptionHelpFormatter keeps the line breaks of the docstring when --help prints it.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    # [explain] Find the repository root, so the script works when started from any folder. Piece by piece:
    # [explain]   subprocess.check_output([...])   runs the command "git rev-parse --show-toplevel" and returns what it prints.
    # [explain]                                    That git command prints the full path of the repo's top folder.
    # [explain]                                    If the command fails (for example, not inside a git repo), the script stops with an error.
    # [explain]   universal_newlines=True          returns the output as text. Without it, Python returns raw bytes.
    # [explain]   .strip()                         removes the newline git adds at the end of its output.
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True).strip()
    if args.files:
        # [explain] Files may be given as absolute paths or as paths relative to where the script runs.
        # [explain] abspath() makes either kind absolute, and relpath() turns that into a path relative to the repo root.
        files = [os.path.relpath(os.path.abspath(f), root) for f in args.files]
    else:
        # [explain] "git ls-files" lists the files git tracks, so build folders and ignored files are never checked.
        # [explain] "-z" separates the names with a NUL character, so a file name with a space or a newline is still read correctly.
        out = subprocess.check_output(["git", "-C", root, "ls-files", "-z"])
        files = [f for f in out.decode().split("\0") if f]

    # [explain] Three lists, one per outcome. They are printed together at the end.
    missing, unknown, fixed = [], [], []
    # [explain] From here on, "rel" is always the file path relative to the repo root, whichever way the file was given.
    for rel in files:
        full = os.path.join(root, rel)
        name = os.path.basename(rel)
        # [explain] Skip exempt files and files listed in SKIP.
        if name.endswith(EXEMPT_SUFFIXES) or name.startswith(".") or name in EXEMPT_NAMES or rel in SKIP:
            continue
        # [explain] Skip symlinks (the *.so and py/*/LICENSE links) and anything that is not a regular file.
        if os.path.islink(full) or not os.path.isfile(full):
            continue
        with open(full, "rb") as f:
            data = f.read()
        # [explain] Skip empty files and binary files.
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
