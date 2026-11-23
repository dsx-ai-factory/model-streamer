# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The license of each dependency, and the table that lists them. Python 3.8 compatible: the
detection functions run in the devcontainer (`collect.py resolve`).

Where a license comes from, in the order the collector trusts it:

1. The rows of `cpp/NOTICE`, for the dependencies linked into a native library we distribute (the
   `ship` category). That file is the shipped notice; a row is found by the last part of its
   package name.
2. `categories.json`, section `licenses`: a license written down by hand, for what the sources
   below cannot tell. An entry can carry a `url` (the license text) and the reason.
3. What the package says about itself: the wheel metadata of a pip package, the `license` field
   of an npm lock file, the Debian copyright file of an apt package, the LICENSE file of a Bazel
   repository.

A dependency none of these covers has the license `UNKNOWN`. The collector never guesses.
"""

import re

UNKNOWN = "UNKNOWN"
NOTICE = "cpp/NOTICE"
NOTICE_SOURCE = "cpp/NOTICE"
# A row of cpp/NOTICE: | `package` | version | license | license source |
NOTICE_ROW = re.compile(r"^\| `([^`]+)` \| ([^|]+?) \| ([^|]+?) \| ([^|]*?) \|$")

# SPDX identifiers, and the names Debian copyright files use for them, that need no link to the text.
# A license outside this list is "nonstandard" for the review: it must come with a link.
STANDARD = frozenset(name.lower() for name in (
    "MIT", "Expat", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSD-2-clause", "BSD-3-clause", "BSD",
    "ISC", "Zlib", "MPL-2.0", "GPL-2", "GPL-3", "LGPL-2", "LGPL-2.1", "LGPL-3", "GPL-2.0", "GPL-3.0", "LGPL-2.0",
    "LGPL-3.0", "Unlicense", "BSL-1.0", "X11",
    "0BSD", "CC0-1.0", "PSF-2.0", "Python-2.0", "OpenSSL", "public-domain", "GCC-exception-3.1",
    "LLVM-exception", "HPND", "MIT-CMU", "Apache",
))

CLASSIFIERS = {  # trove classifiers that name one license without ambiguity
    "MIT License": "MIT",
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
}


# A parenthesis that starts an explanatory note, not an operand of the expression: one that follows
# an operator (`MIT AND (LicenseRef-X)`) belongs to the expression.
_NOTE = re.compile(r"(?<!\bAND)(?<!\bOR)(?<!\bWITH)(?<!\band)(?<!\bor)(?<!\bwith) \(")


def license_expression(license_text):
    """The license expression of a text that may end in a parenthesized note: the expression of
    `MIT (dual-licensed LGPL-2.1 OR MIT; the MIT option is used)` is `MIT`."""
    return _NOTE.split(license_text, 1)[0]


def is_standard(license_text):
    """True when every license named in the expression is a common one. The expression is what comes
    before a parenthesized note: `MIT (dual-licensed LGPL-2.1 OR MIT; the MIT option is used)`."""
    expression = license_expression(license_text)
    names = [token for token in re.findall(r"[A-Za-z0-9.+-]+", expression) if token not in ("AND", "OR", "WITH", "and", "or", "with")]
    if not names:
        return False
    for name in names:
        base = re.sub(r"(-only|-or-later|\+)$", "", name).lower()
        if base not in STANDARD:
            return False
    return True


def _squash(text):
    return re.sub(r"\s+", " ", text).lower()


def detect_text(text):
    """The SPDX identifier of a license text, or None. Only a text that matches exactly one common
    license is named: a file that matches two (a project license with a bundled notice for another)
    is left to the person who reads it. GPL and LGPL are not named: their texts say nothing of
    which version a project grants."""
    flat = _squash(text[:200000])
    found = []
    if "apache license" in flat and "version 2.0, january 2004" in flat:
        found.append("Apache-2.0 WITH LLVM-exception" if "llvm exceptions to the apache 2.0 license" in flat else "Apache-2.0")
    if "permission is hereby granted, free of charge, to any person obtaining a copy" in flat and "without warranty of any kind" in flat:
        # The JSON license is the MIT text plus "shall be used for Good, not Evil": not MIT.
        found.append(None if "good, not evil" in flat else "MIT")
    if "redistribution and use in source and binary forms, with or without modification, are permitted provided that" in flat:
        if "advertising materials" in flat or "end-user documentation" in flat or "acknowledgment" in flat:
            # A BSD license with an advertising or acknowledgment clause (Apache-1.1 has the same
            # first sentence and a "must not be used to endorse" clause): not one of the common ones.
            found.append(None)
        elif "neither the name" in flat or "endorse or promote products" in flat:
            found.append("BSD-3-Clause")
        else:
            found.append("BSD-2-Clause")
    if "permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted" in flat:
        found.append("ISC")
    if "mozilla public license" in flat and "version 2.0" in flat:
        found.append("MPL-2.0")
    if "this software is provided 'as-is'" in flat and "altered source versions must be plainly marked as such" in flat:
        found.append("Zlib")
    if "this is free and unencumbered software released into the public domain" in flat:
        found.append("Unlicense")
    if "boost software license - version 1.0" in flat:
        found.append("BSL-1.0")
    return found[0] if len(found) == 1 and found[0] else None


def pip_license(metadata):
    """(license, source) from the core metadata of a wheel, as pip's resolver report carries it, or None.
    The license expression (PEP 639) wins; then a short `License` field; then a long one that is the
    text of a common license; then the trove classifiers."""
    expression = (metadata.get("license_expression") or "").strip()
    if expression:
        return expression, "wheel metadata (License-Expression)"
    field = (metadata.get("license") or "").strip()
    if field and field.upper() != "UNKNOWN" and len(field) <= 80 and "\n" not in field:
        return field, "wheel metadata (License)"
    if field:
        named = detect_text(field)
        if named:
            return named, "wheel metadata (License text)"
    classified = []
    for classifier in metadata.get("classifier") or metadata.get("classifiers") or []:
        if classifier.startswith("License :: "):
            name = classifier.split(" :: ")[-1]
            if name.startswith(("OSI Approved", "Other/Proprietary")) or name == "License":
                continue
            classified.append(CLASSIFIERS.get(name, name))
    if classified:
        return " AND ".join(sorted(set(classified))), "wheel metadata (classifiers)"
    return None


def npm_license(details):
    """The `license` of an entry of an npm lock file (a string, `{type}`, or a list), or None."""
    value = details.get("license")
    if isinstance(value, dict):
        value = value.get("type")
    if isinstance(value, list):
        value = " OR ".join(item.get("type", "") if isinstance(item, dict) else str(item) for item in value)
    return value.strip() if isinstance(value, str) and value.strip() else None


def debian_copyright(text):
    """(license, source) from the copyright file of a Debian or Ubuntu package, or None. A machine-readable
    file names its licenses in `License:` fields; an older one points to /usr/share/common-licenses; a
    file that has neither is searched for the text of a common license."""
    if re.search(r"^Format:", text, re.M):
        names = []
        for match in re.finditer(r"^License:[ \t]*(\S[^\n]*)$", text, re.M):
            name = re.sub(r"\s+", " ", match.group(1)).strip()
            if name and name not in names:
                names.append(name)
        if names:
            return ", ".join(sorted(names)), "Debian copyright file (License fields)"
    referenced = sorted(set(re.findall(r"/usr/share/common-licenses/([A-Za-z0-9](?:[A-Za-z0-9.+-]*[A-Za-z0-9+])?)", text)))
    if referenced:
        return ", ".join(referenced), "Debian copyright file (common-licenses references)"
    named = detect_text(text)
    if named:
        return named, "Debian copyright file (License text)"
    return None


LICENSE_FILE = re.compile(r"^(licen[sc]e|copying|unlicense)([._-].*)?$", re.I)


def repository_license(files):
    """The license of a fetched repository, from its top-level license files: `files` is {file name:
    text}. Every license file must be recognized, and the result lists each license once. None
    when there is no license file or one of them is not a common license."""
    found = []
    for name in sorted(files):
        if not LICENSE_FILE.match(name):
            continue
        named = detect_text(files[name])
        if not named:
            return None
        if named not in found:
            found.append(named)
    return " AND ".join(found) if found else None


def component_id(package):
    """The id of a notice row: the last part of its package name, lower case, `_` as `-`."""
    return package.rsplit("/", 1)[-1].lower().replace("_", "-")


def notice_components(root):
    """The rows of the repository's cpp/NOTICE as a list of {id, version, spdx, url}: the license of each
    component the notice lists and the link to its text. An empty list when the file is not there."""
    import os

    try:
        with open(os.path.join(root, NOTICE), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    found = []
    for line in lines:
        match = NOTICE_ROW.match(line)
        if match:
            found.append({"id": component_id(match.group(1)), "version": match.group(2),
                          "spdx": match.group(3), "url": match.group(4).strip()})
    return found


def ship_without_notice(entries, components, config):
    """The `ship` entries with a component id that has no row in cpp/NOTICE: they are linked into a
    native library but the notice does not list them (they are not declared for the notice pipeline)."""
    ids = {component["id"] for component in components}
    ship = {key: values for key, values in config["ship"].items() if not key.startswith("_")}
    found = []
    for item in entries:
        key = "%s:%s" % (item["ecosystem"], item["name"])
        if "ship" in item["categories"] and any(cid not in ids for cid in ship.get(key, [])):
            missing = [cid for cid in ship[key] if cid not in ids]
            found.append("%s %s: no row in %s for %s" % (item["ecosystem"], item["name"], NOTICE, ", ".join(missing)))
    return found


def _row_for(rows, version):
    """The row of a component id for a dependency of this version. A name can have two rows (curl is built
    from source for the SDKs and comes again as a Bazel module): take the one whose version is the
    dependency's (`_` and `.` alike), then one that starts with it, else the first."""
    wanted = str(version).replace("_", ".")
    for test in (lambda have: have == wanted, lambda have: have.startswith(wanted) or wanted.startswith(have)):
        for row in rows:
            if test(str(row.get("version", "")).replace("_", ".")):
                return row
    return rows[0]


def assign(entries, components, config):
    """Set `license`, `license_source` and, for a nonstandard license, `license_url` on every entry,
    from the sources described at the top of this file. `components` is the list `notice_components`
    returns; `config` is categories.json. An entry that arrives with a detected `license` keeps it unless
    a better source covers it. Returns the entries."""
    by_id = {}
    for component in components:
        by_id.setdefault(component["id"], []).append(component)
    ship = {key: ids for key, ids in config["ship"].items() if not key.startswith("_")}
    manual = {key: value for key, value in config.get("licenses", {}).items() if not key.startswith("_")}
    urls = {key: value for key, value in config.get("license_urls", {}).items() if not key.startswith("_")}
    for item in entries:
        key = "%s:%s" % (item["ecosystem"], item["name"])
        found = ([_row_for(by_id[cid], item["version"]) for cid in ship.get(key, []) if cid in by_id]
                 if "ship" in item["categories"] else [])
        if found:
            names = []
            for component in found:
                if component["spdx"] not in names:
                    names.append(component["spdx"])
            item["license"] = " AND ".join(names)
            item["license_source"] = NOTICE_SOURCE
            if not is_standard(item["license"]):
                texts = [component["url"] for component in found if component.get("url")]
                if texts:
                    item["license_url"] = texts[0]
        elif key in manual:
            value = manual[key]
            value = {"license": value} if isinstance(value, str) else value
            item["license"] = value["license"]
            item["license_source"] = "categories.json licenses"
            if value.get("url"):
                item["license_url"] = value["url"]
        elif item.get("license") in (None, "", UNKNOWN):
            item["license"] = UNKNOWN
            item["license_source"] = "no source names a license"
        if item["license"] != UNKNOWN and not item.get("license_url") and not is_standard(item["license"]):
            link = _license_link(item["license"], urls)
            if link:
                item["license_url"] = link
    return entries


def _license_link(license_text, urls):
    """The link `license_urls` of categories.json gives for a license a package names by an
    identifier that is not a common one (`LicenseRef-NVIDIA-Proprietary`), or None."""
    for name in re.findall(r"[A-Za-z0-9.+-]+", license_expression(license_text)):
        if name in urls:
            return urls[name]
    return None


def problems(data):
    """What `collect.py check` reports about the licenses of a list: an entry without one, and a
    nonstandard license, written down by hand, without a link to its text."""
    found = []
    for item in data["dependencies"]:
        label = "%s %s %s" % (item["ecosystem"], item["name"], item["version"])
        if not item.get("license"):
            found.append(label + ": no license field")
        elif (item.get("license_source") in (NOTICE_SOURCE, "categories.json licenses")
              and item["license"] != UNKNOWN and not is_standard(item["license"]) and not item.get("license_url")):
            found.append(label + ": nonstandard license %r has no license_url" % item["license"])
    return found


def unknown(data):
    """The entries whose license no source names."""
    return ["%s %s %s" % (item["ecosystem"], item["name"], item["version"])
            for item in data["dependencies"] if item.get("license") == UNKNOWN]


def summary(data):
    """{license: number of entries} for the list."""
    counts = {}
    for item in data["dependencies"]:
        counts[item.get("license", UNKNOWN)] = counts.get(item.get("license", UNKNOWN), 0) + 1
    return counts


def table(data, form="md"):
    """The list as a table of name, license and category, one row per dependency, for a bug comment or
    a review. `form` is "md" (a pipe table, readable as plain text too) or "csv"."""
    rows = []
    for item in sorted(data["dependencies"], key=lambda e: (e["name"].lower(), e["version"], e["ecosystem"])):
        license_text = item.get("license", UNKNOWN)
        if item.get("license_url"):
            license_text += " (%s)" % item["license_url"]
        rows.append(("%s %s" % (item["name"], item["version"]), license_text, ", ".join(item["categories"]), item["ecosystem"]))
    if form == "csv":
        import csv
        import io
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["dependency", "license", "category", "ecosystem"])
        writer.writerows(rows)
        return out.getvalue()
    lines = ["| Dependency | License | Category | Ecosystem |", "|---|---|---|---|"]
    lines += ["| %s |" % " | ".join(cell.replace("|", "\\|") for cell in row) for row in rows]
    return "\n".join(lines) + "\n"
