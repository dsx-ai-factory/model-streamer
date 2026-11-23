#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check, or add, the SPDX copyright header on source files.

    .github/scripts/spdx_headers/spdx_headers.py               check all tracked files
    .github/scripts/spdx_headers/spdx_headers.py FILE...       check only these files
    .github/scripts/spdx_headers/spdx_headers.py --fix [FILE...]   add the NVIDIA header where a file has none

A file passes when its first lines hold at least one SPDX-FileCopyrightText line and an
SPDX-License-Identifier line. The holder can be anyone: a file written by an outside contributor
carries that contributor's notice, and a file changed by both carries both. Every NVIDIA line must
have the exact form "Copyright (c) YYYY[-YYYY] NVIDIA CORPORATION & AFFILIATES. All rights reserved."

--fix only touches files that have no header at all. It never edits, moves or removes a notice that
exists, and a file with only part of one (one tag, or a tag without a value) is reported, not fixed. The year is the year of the first commit of the file; when the file was changed in later
years the line is a range up to the year of the last change. A file without history (new or not yet
committed) gets the current year. A commit that only adds or rewrites header lines is not a change.
Any other change counts, so there is no judgement of how big a change is.

FILE can be an absolute path, or a path relative to the folder you run the script from.
Run it inside the git repository. Exit code is 1 when a file has no header, has an NVIDIA line in
the wrong form, or has an unknown file type.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys

COPYRIGHT_TAG = "SPDX-FileCopyrightText:"
LICENSE_TAG = "SPDX-License-Identifier:"
# The license of the files this script adds a header to.
LICENSE = LICENSE_TAG + " Apache-2.0"

# The exact NVIDIA line of the Legal copyright guidance, after the SPDX tag.
NVIDIA_LINE = re.compile(r"Copyright \(c\) (\d{4})(?:-(\d{4}))? NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.$")

# A header only counts when it is within the first WINDOW lines of the file.
WINDOW = 20

# Files that need no header (the OSRB exempt list): docs, config, data, images and license files.
# A name is exempt if it ends with one of the suffixes or is one of the exact names. A name that
# starts with a dot (.gitignore, .bazelrc) is tool config, so it is exempt too, as in the OSRB scanner.
EXEMPT_SUFFIXES = (".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
                   ".ipynb", ".png", ".jpg", ".svg", ".safetensors", ".so")
EXEMPT_NAMES = ("LICENSE", "NOTICE", "COPYRIGHT", "DCO", "CODEOWNERS")

# Files to skip on purpose, for example if OSRB says a file must not get the header.
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


def nvidia_copyright(first_year, last_year=None):
    """The NVIDIA copyright text: one year, or a range from the first to the last year."""
    years = str(first_year)
    if last_year is not None and last_year != first_year:
        years += "-%d" % last_year
    return "%s Copyright (c) %s NVIDIA CORPORATION & AFFILIATES. All rights reserved." % (COPYRIGHT_TAG, years)


def header_for(style, copyright_line):
    # The two lines of the OSRB Apache 2.0 template. C-style is wrapped in /* */.
    if style == "c":
        return ["/*", " * " + copyright_line, " * " + LICENSE, " */"]
    return ["# " + copyright_line, "# " + LICENSE]


def top_lines(text):
    # Only the first WINDOW lines count. A header deep in the file does not.
    return text.splitlines()[:WINDOW]


def tag_value(line, tag):
    """The text after `tag` when the line is a comment that starts with it (`# `, `// `, `/* ` or ` * `
    before the tag), without a closing comment delimiter. None for any other line, so a tag inside a
    string or a docstring text is not a notice."""
    match = re.match(r"^\s*(?:#|//|/\*|\*)+\s*" + re.escape(tag) + r"(.*)$", line)
    return match.group(1).replace("*/", "").strip() if match else None


def tag_values(text, tag):
    """The values of every comment line with `tag` in the top of the file."""
    values = (tag_value(line, tag) for line in top_lines(text))
    return [value for value in values if value is not None]


def copyright_lines(text):
    """The text after SPDX-FileCopyrightText: of each such comment line in the top of the file."""
    return tag_values(text, COPYRIGHT_TAG)


def license_values(text):
    """The text after SPDX-License-Identifier: of each such comment line in the top of the file."""
    return tag_values(text, LICENSE_TAG)


def has_header(text):
    # The tag must follow a comment marker (see tag_value).
    # Both tags need a value, and so does every line that has a tag: an empty template is not a header,
    # and an empty tag next to a good one is a notice left unfinished.
    copyrights, licenses = copyright_lines(text), license_values(text)
    return bool(copyrights) and bool(licenses) and all(copyrights) and all(licenses)


def has_partial_header(text):
    """True when the top of the file has an SPDX tag but not a complete header: only one of the two
    tags, or a tag without a value. Such a file belongs to someone else's notice, so it is reported and
    never changed."""
    return bool(copyright_lines(text) or license_values(text)) and not has_header(text)


def bad_nvidia_lines(text):
    """NVIDIA copyright lines that are not in the exact form, or whose years run backwards."""
    bad = []
    for line in copyright_lines(text):
        # A notice is NVIDIA's when its holder (the text after "Copyright", "(c)" and the years) starts
        # with the name, in any spelling, so `Nvidia Corporation` is checked too. A line that only
        # mentions NVIDIA, such as `Foo, derived from NVIDIA code`, belongs to another holder.
        holder = re.sub(r"^Copyright\b\s*(?:\(c\)|©)?[\s\d,-]*", "", line, flags=re.IGNORECASE)
        if not holder.lower().startswith("nvidia"):
            continue
        match = NVIDIA_LINE.match(line)
        if not match or (match.group(2) and int(match.group(2)) < int(match.group(1))):
            bad.append(line)
    return bad


def is_header_line(line):
    """True for a line that belongs to a header: SPDX tags, comment delimiters and blank lines."""
    stripped = line.strip()
    if stripped in ("", "#", "/*", "*/", "*", "//"):
        return True
    return re.match(r"^(#|//|\*|/\*)?\s*SPDX-[A-Za-z-]+:", stripped) is not None


def change_years(root, rel):
    """Years of the commits that changed the content of the file, following renames.

    A commit that only adds or rewrites header lines, or only renames the file, is not a change.
    """
    # --cc also shows a merge, but only what its resolution changed against every parent: the changes
    # the parents brought in are counted once, at the year of the commit that made them.
    proc = subprocess.run(
        ["git", "-C", root, "log", "--no-color", "--follow", "--cc", "--format=%x00%ad", "--date=format:%Y", "-p",
         "-U0", "--", rel],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, errors="replace")
    # git log fails in a repository without a commit yet: the file has no history.
    out = proc.stdout if proc.returncode == 0 else ""
    years = []
    # Every commit starts with a NUL, then its year on the first line, then its diff.
    for chunk in out.split("\0")[1:]:
        year, _, diff = chunk.partition("\n")
        columns = 0  # characters of change marks before the text of a line: 1, or one per parent of a merge
        for line in diff.splitlines():
            if line.startswith("@@"):
                columns = len(line) - len(line.lstrip("@")) - 1
            elif line.startswith("diff "):
                columns = 0  # the file header lines (---, +++, index) are not changes
            elif columns and (set(line[:columns]) & {"+", "-"}) and not is_header_line(line[columns:]):
                # Inside a hunk, a text that starts with ++ or -- is still a changed line.
                years.append(int(year))
                break
    return years


def staged_rename(root, rel):
    """(old path, similarity) when the index holds a rename to rel that is not committed yet, else None.
    `git log --follow` walks commits only, so it cannot see this rename."""
    out = subprocess.check_output(["git", "-C", root, "diff", "--cached", "--name-status", "-M", "--no-color", "-z"],
                                  universal_newlines=True, errors="replace")
    fields = out.split("\0")
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        if status.startswith(("R", "C")):
            old, new = fields[index + 1], fields[index + 2]
            if new == rel and status.startswith("R"):
                return old, int(status[1:] or 100)
            index += 3
        else:
            index += 2
    return None


def is_shallow(root):
    """True for a shallow clone, whose history stops at a boundary commit."""
    out = subprocess.run(["git", "-C", root, "rev-parse", "--is-shallow-repository"], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, universal_newlines=True).stdout
    return out.strip() == "true"


def history_path(root, rel):
    """The path whose commits hold the history of rel: the old path of a rename that is staged but not
    committed yet, because `git log` knows no commit under the new name."""
    renamed = staged_rename(root, rel)
    return renamed[0] if renamed else rel


def has_history(root, rel):
    """True when some commit touches the file."""
    out = subprocess.run(["git", "-C", root, "log", "-1", "--format=%H", "--", rel], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, universal_newlines=True).stdout
    return bool(out.strip())


def file_years(root, rel, current_year):
    """(first, last) year for the NVIDIA line of a file that has no header."""
    renamed = staged_rename(root, rel)
    years = change_years(root, renamed[0] if renamed else rel)
    # Uncommitted edits to a file with history are a change this year. A staged rename alone is not.
    status = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--", rel], universal_newlines=True)
    edited = not renamed or renamed[1] < 100 or status[1:2] not in ("", " ")
    if status.strip() and not status.startswith("??") and edited:
        years.append(current_year)
    if not years:
        return current_year, current_year
    return min(years), max(years)


def add_header(text, style, copyright_line=None):
    # The default is the NVIDIA line for the current year, as for a file created today.
    if copyright_line is None:
        copyright_line = nvidia_copyright(datetime.date.today().year)
    # The header is followed by a blank line. If the file already starts with a blank line, that
    # gives two blank lines, which is harmless.
    header = "\n".join(header_for(style, copyright_line)) + "\n\n"
    # A shebang line (#!/usr/bin/env python3) must stay first, so the header goes after it.
    if text.startswith("#!"):
        # partition() also works when the shebang is the whole file and has no newline.
        shebang, _, rest = text.partition("\n")
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

    current_year = datetime.date.today().year
    missing, unknown, not_utf8, malformed, incomplete, shallow_files, fixed = [], [], [], [], [], [], []
    shallow = args.fix and is_shallow(root)
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
        # Lossy decoding is fine for reading: the header is plain ASCII.
        text = data.decode("utf-8", errors="replace")
        if has_header(text):
            # The header exists and is never edited. Only the form of an NVIDIA line is checked.
            for line in bad_nvidia_lines(text):
                malformed.append((rel, line))
            continue
        if has_partial_header(text):
            # Only one tag, or a tag without a value: a notice someone wrote. --fix would stack an NVIDIA
            # notice on it, which CONTRIBUTING.md forbids for a file NVIDIA did not write.
            incomplete.append(rel)
            continue
        style = style_of(rel)
        if style is None:
            unknown.append(rel)
        elif args.fix:
            # Writing the lossy text back would change the file's bytes, so refuse a non-UTF-8 file.
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                not_utf8.append(rel)
                continue
            if shallow and has_history(root, history_path(root, rel)):
                # A shallow clone has cut off the commits that hold the creation year.
                shallow_files.append(rel)
                continue
            first, last = file_years(root, rel, current_year)
            # newline="" keeps the line endings (CRLF) as they are.
            with open(full, "w", encoding="utf-8", newline="") as f:
                f.write(add_header(text, style, nvidia_copyright(first, last)))
            fixed.append(rel)
        else:
            missing.append(rel)

    for rel in unknown:
        print("unknown file type (edit .github/scripts/spdx_headers/spdx_headers.py): " + rel)
    for rel in not_utf8:
        print("not UTF-8, add the header by hand: " + rel)
    for rel, line in malformed:
        print("NVIDIA copyright line is not 'Copyright (c) YYYY[-YYYY] NVIDIA CORPORATION & AFFILIATES. "
              "All rights reserved.' (%s): %s" % (line, rel))
    for rel in shallow_files:
        print("shallow clone: the years of this file come from history that this clone does not have. Run "
              "`git fetch --unshallow` (a plain fetch is not enough) and run --fix again: " + rel)
    for rel in incomplete:
        print("incomplete SPDX header: it needs both an SPDX-FileCopyrightText and an SPDX-License-Identifier "
              "line, each with a value. Complete it by hand, --fix does not change it: " + rel)
    for rel in missing:
        print("missing SPDX header (run .github/scripts/spdx_headers/spdx_headers.py --fix): " + rel)
    for rel in fixed:
        print("fixed   " + rel)
    if fixed:
        print("Fixed %d file(s)." % len(fixed))
    return 1 if (missing or unknown or not_utf8 or malformed or incomplete or shallow_files) else 0


if __name__ == "__main__":
    sys.exit(main())
