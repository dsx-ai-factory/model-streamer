# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the external inputs of .devcontainer/Dockerfile and of the shell scripts it runs: base images,
apt and pip packages, npm packages, downloaded archives and git clones. Python 3.8 compatible (it
also runs in the image).

Every reader takes the text of the file and its path. A path that ends in `.sh` is read as a script
(every line is a command); any other path is read as a Dockerfile (the `RUN` instructions)."""

import re

DOCKERFILE = ".devcontainer/Dockerfile"
ARCHES = ("x86_64", "aarch64")
# The architecture argument of the scripts (`install-test-services.sh amd64`) and the name the list uses.
SCRIPT_ARCHES = {"amd64": "x86_64", "arm64": "aarch64"}
# `$([ "$(uname -m)" = "aarch64" ] && echo "arm64" || echo "amd64")` -> one value per architecture.
UNAME_CHOICE = re.compile(r'\$\(\s*\[ "\$\(uname -m\)" = "aarch64" \] && echo "(\w+)" \|\| echo "(\w+)"\s*\)')
URL_START = re.compile(r"https?://")


def read_urls(segment):
    """URLs in a shell segment. A `$( ... )` inside a URL may contain spaces and quotes, so the end
    of a URL is the first space, quote or pipe outside any parentheses."""
    urls = []
    for start in URL_START.finditer(segment):
        depth, end = 0, start.start()
        for end in range(start.start(), len(segment)):
            char = segment[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and char in " \t\"'|;&\\":
                break
        else:
            end = len(segment)
        urls.append(segment[start.start():end])
    return urls


def logical_lines(text):
    """Dockerfile instructions as (first line number, text). Continuation lines are joined and
    comment lines inside a continued instruction are dropped, as Docker does."""
    out = []
    current, start = [], 0
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if not current:
            if not stripped:
                continue
            start = number
        if stripped.endswith("\\"):
            current.append(stripped[:-1].strip())
            continue
        current.append(stripped)
        out.append((start, " ".join(part for part in current if part)))
        current = []
    if current:
        out.append((start, " ".join(part for part in current if part)))
    return out


def instructions(text):
    for number, line in logical_lines(text):
        word, _, rest = line.partition(" ")
        yield number, word.upper(), rest.strip()


def is_script(path):
    return path.endswith(".sh")


def runs(text, path):
    """The shell commands of a file as (line, command): the `RUN` instructions of a Dockerfile, or
    every logical line of a script."""
    if is_script(path):
        yield from logical_lines(text)
    else:
        for number, word, rest in instructions(text):
            if word == "RUN":
                yield number, rest


def base_images(text):
    """FROM lines as (line, image, tag, digest, stage name)."""
    found = []
    for number, word, rest in instructions(text):
        if word != "FROM":
            continue
        match = re.match(r"(\S+?)(?::([^@\s/]+))?(?:@(sha256:[0-9a-f]+))?(?:\s+AS\s+(\S+))?$", rest, re.I)
        found.append((number, match.group(1), match.group(2), match.group(3), match.group(4)))
    return found


def _tokens(segment):
    return [token.strip("\"'") for token in segment.split()]


def _segments(run):
    """Split a shell command at `&&`, `||` and `;` that are outside parentheses and quotes."""
    parts, depth, quote, start, index = [], 0, "", 0, 0
    while index < len(run):
        char = run[index]
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and (run.startswith("&&", index) or run.startswith("||", index) or char == ";"):
            parts.append(run[start:index])
            start = index + (1 if char == ";" else 2)
            index = start
            continue
        index += 1
    parts.append(run[start:])
    return [part.strip() for part in parts if part.strip()]


def apt_packages(text, path=DOCKERFILE):
    """Packages named on `apt-get install` lines, as (line, [package])."""
    found = []
    for number, rest in runs(text, path):
        for segment in _segments(rest):
            match = re.search(r"apt-get install\s+(.*)", segment)
            if match:
                packages, skip = [], False
                for token in _tokens(match.group(1)):
                    if skip or token.startswith("-"):
                        skip = token in ("-o", "-t", "-c")  # these options take the next word as their value
                    else:
                        packages.append(token.split("=")[0])  # foo=1.2 names foo
                if packages:
                    found.append((number, packages))
    return found


def pip_packages(text, path=DOCKERFILE):
    """Requirements named on `pip install` lines, as (line, [requirement])."""
    found = []
    for number, rest in runs(text, path):
        for segment in _segments(rest):
            match = re.search(r"pip3? install\s+(.*)", segment)
            if match:
                requirements = [t for t in _tokens(match.group(1)) if not t.startswith("-")]
                if requirements:
                    found.append((number, requirements))
    return found


def npm_packages(text, path=DOCKERFILE):
    found = []
    for number, rest in runs(text, path):
        for segment in _segments(rest):
            match = re.search(r"npm install\s+(.*)", segment)
            if match:
                found.append((number, [t for t in _tokens(match.group(1)) if not t.startswith("-")]))
    return found


def _arch_urls(url, run):
    """Expand a URL that depends on the architecture into {architecture: url}."""
    choice = UNAME_CHOICE.search(url)
    if choice:
        return {"x86_64": UNAME_CHOICE.sub(choice.group(2), url, 1), "aarch64": UNAME_CHOICE.sub(choice.group(1), url, 1)}
    variable = re.search(r"\$\{(\w+)\}", url)
    if variable:
        assignment = re.search(r"%s=\$\(([^)]*\([^)]*\)[^)]*)\)" % variable.group(1), run)
        choice = UNAME_CHOICE.search("$(" + assignment.group(1) + ")") if assignment else None
        if choice:
            return {"x86_64": url.replace(variable.group(0), choice.group(2)),
                    "aarch64": url.replace(variable.group(0), choice.group(1))}
    return {"any": url}


def script_variables(text):
    """What a script sets, as ({name: value} for every architecture, {architecture: {name: value}}).

    A script is not run, so only the shapes the devcontainer scripts use are read: `NAME=value`
    lines, and `NAME=value` lines inside a `case` branch named `amd64)` or `arm64)`. The name
    assigned from `$1` is the architecture argument. Values that run a command are skipped."""
    common, specific, branch = {}, {arch: {} for arch in SCRIPT_ARCHES}, None
    for _, line in logical_lines(text):
        label = re.match(r"(amd64|arm64|\*)\)\s*(.*)$", line)
        if label:
            branch, line = label.group(1), label.group(2)
        closes = line.endswith(";;")  # the end of the branch
        line = re.sub(r"\s*;;$", "", line)
        assignment = re.match(r"([A-Za-z_]\w*)=(.*)$", line)
        if assignment and "$(" not in assignment.group(2) and "`" not in assignment.group(2):
            name, value = assignment.group(1), assignment.group(2).strip().strip("\"'")
            if branch in SCRIPT_ARCHES:
                specific[branch][name] = value
            elif branch is None:
                common[name] = value
        if closes:
            branch = None
    return common, specific


def _expand(text, variables):
    """Replace `${NAME}` by the value of NAME, also inside the values, and the two shell forms the
    scripts use on a version: `${NAME//a/b}` (replace every a) and `${NAME%pattern}` (drop the
    shortest matching suffix, pattern a glob)."""
    import fnmatch

    def value(match):
        name, operator, argument = match.group(1), match.group(2), match.group(3)
        if name not in variables:
            return match.group(0)
        found = variables[name]
        if operator == "//":
            old, _, new = argument.partition("/")
            return found.replace(old, new)
        if operator == "%":
            for cut in range(len(found), -1, -1):
                if fnmatch.fnmatchcase(found[cut:], argument):
                    return found[:cut]
        return found

    for _ in range(5):
        replaced = re.sub(r"\$\{(\w+)(?:(//|%)([^}]*))?\}", value, text)
        if replaced == text:
            break
        text = replaced
    return text


def dockerfile_arguments(path=None):
    """The default of every `ARG NAME=value` in the Dockerfile: the versions it passes to the
    scripts it runs (`RUN NAME="$NAME" bash script.sh`)."""
    import os

    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), DOCKERFILE)
    found = {}
    try:
        with open(path) as handle:
            for line in handle:
                match = re.match(r"\s*ARG\s+([A-Za-z_]\w*)=(\S*)", line)
                if match and match.group(1) not in found:
                    found[match.group(1)] = match.group(2).strip("\"'")
    except OSError:
        pass
    return found


def _script_environment(text, arch):
    """The variables of a script when it runs for `arch` (a key of SCRIPT_ARCHES)."""
    common, specific = script_variables(text)
    variables = dockerfile_arguments()
    variables.update(common)
    variables.update(specific[arch])
    for name, value in common.items():
        if value == "$1":  # `target_arch=$1`: the architecture argument
            variables[name] = arch
    return variables


def _script_downloads(text, path):
    found = []
    environments = {arch: _script_environment(text, arch) for arch in SCRIPT_ARCHES}
    for number, line in logical_lines(text):
        for segment in _segments(line):
            if not re.match(r"curl\b", segment):
                continue
            urls = read_urls(segment)
            if not urls:
                continue
            output = re.search(r"-o\s+(\S+)", segment)
            filename = output.group(1) if output else None
            per_arch = {SCRIPT_ARCHES[arch]: _expand(urls[0], environments[arch]) for arch in SCRIPT_ARCHES}
            for url in per_arch.values():
                if "$" in url:
                    raise ValueError("%s line %d: cannot work out the URL %s; add the variable it uses to a `case amd64)` "
                                     "branch or an assignment, or teach script_variables about the new shape" % (path, number, url))
            pins = {}
            if filename:
                pinned = re.search(r'echo\s+"(\S+)\s+%s"\s*\|\s*sha256sum' % re.escape(filename), text)
                if pinned:
                    for arch, name in SCRIPT_ARCHES.items():
                        value = _expand(pinned.group(1), environments[arch])
                        if re.fullmatch(r"[0-9a-f]{64}", value):
                            pins[name] = value
            if len(set(per_arch.values())) == 1:
                urls_by_arch = {"any": next(iter(per_arch.values()))}
                hashes = {"any": next(iter(pins.values()))} if pins and len(set(pins.values())) == 1 else {}
            else:
                urls_by_arch, hashes = per_arch, pins
            found.append({"line": number, "filename": filename, "urls": urls_by_arch, "sha256": hashes})
    return found


def downloads(text, path=DOCKERFILE):
    """Archives and scripts fetched with curl or wget, as dicts with line, filename, urls ({arch: url})
    and sha256 ({arch: hash}) where the file pins one. In a script the pin is an `echo "<hash>  <file>" |
    sha256sum -c -` line anywhere in the script; the architecture comes from `case amd64)` branches."""
    if is_script(path):
        return _script_downloads(text, path)
    found = []
    for number, word, rest in instructions(text):
        if word != "RUN" or "curl" not in rest:
            continue
        for segment in _segments(rest):
            if not segment.startswith(("curl", "WEED_ARCH=")) and "curl " not in segment:
                continue
            urls = read_urls(segment)
            if not urls:
                continue
            output = re.search(r"-o\s+(\S+)", segment)
            filename = output.group(1) if output else None
            record = {"line": number, "filename": filename, "urls": _arch_urls(urls[0], rest), "sha256": {}}
            found.append(record)
        for record in (r for r in found if r["line"] == number):
            filename = record["filename"]
            pinned = re.search(r'echo\s+"([0-9a-f]{64})\s+%s"\s*\|\s*sha256sum' % re.escape(filename or "-"), rest)
            if pinned:
                record["sha256"] = {"any": pinned.group(1)}
                continue
            # `WEED_SHA256=<amd64 hash>; ... else WEED_SHA256=<arm64 hash>`: the first is amd64.
            variable = re.findall(r"\b(\w+_SHA256)=([0-9a-f]{64})", rest)
            if len(variable) == 2 and set(record["urls"]) == {"x86_64", "aarch64"}:
                record["sha256"] = {"x86_64": variable[0][1], "aarch64": variable[1][1]}
    return found


def git_clones(text, path=DOCKERFILE):
    """`git clone` lines as (line, url, ref) where ref is --branch or a later `git checkout`."""
    found = []
    current = None  # the clone that a later `git checkout` applies to
    for number, rest in runs(text, path):
        if "git clone" not in rest and "git checkout" not in rest:
            continue
        if not is_script(path):
            current = None  # in a Dockerfile a checkout applies to a clone of the same RUN only
        for segment in _segments(rest):
            tokens = _tokens(segment)
            if tokens[:2] == ["git", "clone"]:
                branch = None
                for index, token in enumerate(tokens):
                    if token == "--branch" and index + 1 < len(tokens):
                        branch = tokens[index + 1]
                urls = [t for t in tokens if re.match(r"(https?://|ssh://|git://|git@)", t)]
                if not urls:
                    raise ValueError("%s line %d: no repository URL in: %s" % (path, number, segment))
                current = {"line": number, "url": urls[0], "ref": branch, "recursive": "--recurse-submodules" in tokens}
                found.append(current)
            elif tokens[:2] == ["git", "checkout"] and current and not current["ref"] and len(tokens) > 2:
                current["ref"] = tokens[2]
    return found
