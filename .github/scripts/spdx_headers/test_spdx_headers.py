# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for spdx_headers.py. Run with: python3 .github/scripts/spdx_headers/test_spdx_headers.py"""
# unittest is in the Python standard library, so the tests need no install and no pytest.

import datetime
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(SCRIPTS_DIR, "spdx_headers.py")
sys.path.insert(0, SCRIPTS_DIR)
import spdx_headers  # noqa: E402

THIS_YEAR = datetime.date.today().year

# The header as the script writes it for a file created in 2024, for use in test files.
COPYRIGHT = spdx_headers.nvidia_copyright(2024)
HASH_HEADER = "# %s\n# %s\n" % (COPYRIGHT, spdx_headers.LICENSE)
C_HEADER = "/*\n * %s\n * %s\n */\n" % (COPYRIGHT, spdx_headers.LICENSE)
# What --fix writes for a file with no history: the current year.
FIXED_HASH_HEADER = "# %s\n# %s\n" % (spdx_headers.nvidia_copyright(THIS_YEAR), spdx_headers.LICENSE)

# A notice of an outside contributor, in the same SPDX form.
EXTERNAL_COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2023 Example Corp"
EXTERNAL_HEADER = "# %s\n# %s\n" % (EXTERNAL_COPYRIGHT, spdx_headers.LICENSE)
STACKED_HEADER = "# %s\n# %s\n# %s\n" % (EXTERNAL_COPYRIGHT, COPYRIGHT, spdx_headers.LICENSE)


class HelperTests(unittest.TestCase):
    """Tests of the small functions. No files are involved."""

    def test_has_header_hash_style(self):
        self.assertTrue(spdx_headers.has_header(HASH_HEADER + "import os\n"))

    def test_has_header_c_style(self):
        self.assertTrue(spdx_headers.has_header(C_HEADER + "int x;\n"))

    def test_has_header_missing(self):
        self.assertFalse(spdx_headers.has_header("import os\n"))

    def test_has_header_needs_both_lines(self):
        self.assertFalse(spdx_headers.has_header("# %s\n" % COPYRIGHT))
        self.assertFalse(spdx_headers.has_header("# %s\n" % spdx_headers.LICENSE))

    def test_has_header_needs_a_value_after_each_tag(self):
        empty = "# SPDX-FileCopyrightText:\n# SPDX-License-Identifier:\n"
        self.assertFalse(spdx_headers.has_header(empty + "import os\n"))
        self.assertFalse(spdx_headers.has_header("# %s\n# SPDX-License-Identifier:\n" % COPYRIGHT))
        self.assertFalse(spdx_headers.has_header("# SPDX-FileCopyrightText:\n# %s\n" % spdx_headers.LICENSE))
        self.assertFalse(spdx_headers.has_header("/*\n * SPDX-FileCopyrightText:\n * SPDX-License-Identifier: */\n"))
        self.assertTrue(spdx_headers.has_header(C_HEADER + "int x;\n"))

    def test_a_one_line_c_notice_loses_its_closing_delimiter(self):
        one_line = "/* %s */\n/* %s */\n" % (COPYRIGHT, spdx_headers.LICENSE)
        self.assertEqual(spdx_headers.copyright_lines(one_line), [COPYRIGHT.split(":", 1)[1].strip()])
        self.assertTrue(spdx_headers.has_header(one_line))
        self.assertEqual(spdx_headers.bad_nvidia_lines(one_line), [])
        # An empty one-line notice has no value, so it is not a header.
        self.assertEqual(spdx_headers.copyright_lines("/* SPDX-FileCopyrightText: */\n"), [""])
        self.assertFalse(spdx_headers.has_header("/* SPDX-FileCopyrightText: */\n/* %s */\n" % spdx_headers.LICENSE))

    def test_an_nvidia_notice_is_checked_whatever_the_case_of_the_name(self):
        text = "# SPDX-FileCopyrightText: Copyright (c) 2024 Nvidia Corporation\n# %s\n" % spdx_headers.LICENSE
        self.assertTrue(spdx_headers.has_header(text))
        self.assertEqual(spdx_headers.bad_nvidia_lines(text), ["Copyright (c) 2024 Nvidia Corporation"])
        self.assertEqual(spdx_headers.bad_nvidia_lines(HASH_HEADER), [])
        self.assertEqual(spdx_headers.bad_nvidia_lines(EXTERNAL_HEADER), [])

    def test_an_empty_tag_next_to_a_good_one_is_not_a_complete_header(self):
        text = "# SPDX-FileCopyrightText:\n# %s\n# %s\n" % (EXTERNAL_COPYRIGHT, spdx_headers.LICENSE)
        self.assertFalse(spdx_headers.has_header(text))
        self.assertTrue(spdx_headers.has_partial_header(text))
        text = "# %s\n# SPDX-License-Identifier:\n# %s\n" % (EXTERNAL_COPYRIGHT, spdx_headers.LICENSE)
        self.assertFalse(spdx_headers.has_header(text))
        self.assertTrue(spdx_headers.has_header(STACKED_HEADER))
        self.assertTrue(spdx_headers.has_header(EXTERNAL_HEADER))

    def test_partial_header_is_one_tag_or_an_empty_tag(self):
        self.assertTrue(spdx_headers.has_partial_header("# SPDX-License-Identifier: MIT\nimport os\n"))
        self.assertTrue(spdx_headers.has_partial_header("# %s\nimport os\n" % EXTERNAL_COPYRIGHT))
        self.assertTrue(spdx_headers.has_partial_header("# SPDX-FileCopyrightText:\n# %s\n" % spdx_headers.LICENSE))
        self.assertFalse(spdx_headers.has_partial_header(HASH_HEADER + "import os\n"))
        self.assertFalse(spdx_headers.has_partial_header("import os\n"))

    def test_has_header_accepts_any_year_and_a_range(self):
        for first, last in ((2019, None), (2021, 2026), (2024, 2024)):
            header = "# %s\n# %s\n" % (spdx_headers.nvidia_copyright(first, last), spdx_headers.LICENSE)
            self.assertTrue(spdx_headers.has_header(header), (first, last))

    def test_has_header_accepts_an_outside_holder(self):
        self.assertTrue(spdx_headers.has_header(EXTERNAL_HEADER + "import os\n"))

    def test_has_header_accepts_a_stacked_header(self):
        self.assertTrue(spdx_headers.has_header(STACKED_HEADER + "import os\n"))

    def test_has_header_accepts_another_license_id(self):
        text = "# %s\n# SPDX-License-Identifier: MIT AND Apache-2.0\n" % EXTERNAL_COPYRIGHT
        self.assertTrue(spdx_headers.has_header(text))

    def test_nvidia_copyright_one_year_and_range(self):
        self.assertEqual(spdx_headers.nvidia_copyright(2025),
                         "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.")
        self.assertIn("(c) 2024-2026 NVIDIA", spdx_headers.nvidia_copyright(2024, 2026))
        # The same first and last year is one year, not a range.
        self.assertIn("(c) 2026 NVIDIA", spdx_headers.nvidia_copyright(2026, 2026))

    def test_bad_nvidia_lines(self):
        self.assertEqual(spdx_headers.bad_nvidia_lines(STACKED_HEADER), [])
        self.assertEqual(spdx_headers.bad_nvidia_lines(EXTERNAL_HEADER), [])
        for line in ("Copyright 2024 NVIDIA Corporation", "Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.",
                     "Copyright (c) 2026-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
                     "Copyright (c) 2024, 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."):
            text = "# SPDX-FileCopyrightText: %s\n# %s\n" % (line, spdx_headers.LICENSE)
            self.assertEqual(spdx_headers.bad_nvidia_lines(text), [line], line)

    def test_a_line_that_only_mentions_nvidia_is_another_holder(self):
        # The holder is the text after the years; mentioning NVIDIA later does not make it NVIDIA's notice.
        for line in ("Copyright (c) 2025 Foo, derived from NVIDIA code", "Copyright 2024 Acme (not NVIDIA)"):
            text = "# SPDX-FileCopyrightText: %s\n# %s\n" % (line, spdx_headers.LICENSE)
            self.assertEqual(spdx_headers.bad_nvidia_lines(text), [], line)
        # A holder that starts with the name is NVIDIA's, whatever the spelling, and needs the exact form.
        text = "# SPDX-FileCopyrightText: Copyright (c) 2025 Nvidia Corporation\n# %s\n" % spdx_headers.LICENSE
        self.assertEqual(spdx_headers.bad_nvidia_lines(text), ["Copyright (c) 2025 Nvidia Corporation"])

    def test_a_tag_in_code_or_a_string_is_not_a_header(self):
        for line in ('HELP = "add SPDX-FileCopyrightText: Copyright (c) 2024 Foo to files"',
                     'x = 1  # SPDX-FileCopyrightText: Copyright (c) 2024 Foo',
                     'print("SPDX-License-Identifier: MIT")'):
            self.assertEqual(spdx_headers.copyright_lines(line + "\n"), [], line)
            self.assertEqual(spdx_headers.license_values(line + "\n"), [], line)
        text = ('HELP = "SPDX-FileCopyrightText: Copyright (c) 2024 Foo"\n'
                'LICENSE = "SPDX-License-Identifier: MIT"\n')
        self.assertFalse(spdx_headers.has_header(text))
        self.assertFalse(spdx_headers.has_partial_header(text))

    def test_every_comment_style_of_a_tag_counts(self):
        for prefix in ("# ", "#", "// ", "/* ", " * ", "  #  "):
            text = "%sSPDX-FileCopyrightText: Copyright (c) 2024 Foo\n%sSPDX-License-Identifier: MIT\n" % (prefix, prefix)
            self.assertTrue(spdx_headers.has_header(text), prefix)

    def test_is_header_line(self):
        for line in ("", "#", "/*", " */", " * SPDX-License-Identifier: Apache-2.0", "# " + COPYRIGHT):
            self.assertTrue(spdx_headers.is_header_line(line), line)
        for line in ("import os", "int x;", "# a comment"):
            self.assertFalse(spdx_headers.is_header_line(line), line)

    def test_has_header_must_be_near_the_top(self):
        # A header that starts after WINDOW lines of code does not count.
        late = "x = 1\n" * spdx_headers.WINDOW + HASH_HEADER
        self.assertFalse(spdx_headers.has_header(late))

    def test_style_of(self):
        for path in ("a.cc", "dir/a.h", "a.c", "streamer.ldscript"):
            self.assertEqual(spdx_headers.style_of(path), "c", path)
        for path in ("a.py", "BUILD", "cpp/BUILD.tpl", "Makefile", "WORKSPACE", ".devcontainer/Dockerfile",
                     "a.bzl", "x.cmake", "requirements.dev", "cpp/third_party/aws.BUILD"):
            self.assertEqual(spdx_headers.style_of(path), "hash", path)
        self.assertIsNone(spdx_headers.style_of("a.unknown"))

    def test_add_header_hash(self):
        result = spdx_headers.add_header("import os\n", "hash", COPYRIGHT)
        self.assertEqual(result, HASH_HEADER + "\nimport os\n")

    def test_add_header_c(self):
        result = spdx_headers.add_header("int x;\n", "c", COPYRIGHT)
        self.assertEqual(result, C_HEADER + "\nint x;\n")

    def test_add_header_defaults_to_the_current_year(self):
        self.assertEqual(spdx_headers.add_header("import os\n", "hash"), FIXED_HASH_HEADER + "\nimport os\n")

    def test_add_header_keeps_shebang_first(self):
        result = spdx_headers.add_header("#!/usr/bin/env python3\nimport os\n", "hash", COPYRIGHT)
        self.assertEqual(result, "#!/usr/bin/env python3\n" + HASH_HEADER + "\nimport os\n")

    def test_added_header_passes_the_check(self):
        for style in ("hash", "c"):
            self.assertTrue(spdx_headers.has_header(spdx_headers.add_header("code\n", style)))


class EndToEndTests(unittest.TestCase):
    """Runs the script itself, inside a temporary git repository created for each test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = os.path.realpath(self.tmp.name)
        subprocess.check_call(["git", "-C", self.repo, "init", "-q"])
        # The fixture commits are unsigned, whatever the developer's own configuration says.
        subprocess.check_call(["git", "-C", self.repo, "config", "commit.gpgsign", "false"])

    def tearDown(self):
        self.tmp.cleanup()

    def add_file(self, relative_path, content):
        """Create a file in the test repo and tell git about it. Returns its absolute path."""
        path = os.path.join(self.repo, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        subprocess.check_call(["git", "-C", self.repo, "add", relative_path])
        return path

    def run_script(self, *args, from_folder=None):
        """Run spdx_headers.py like a user would. Returns (exit code, printed text)."""
        proc = subprocess.run([sys.executable, SCRIPT] + list(args), cwd=from_folder or self.repo,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        return proc.returncode, proc.stdout

    def test_missing_header_is_reported(self):
        self.add_file("src/a.py", "import os\n")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("src/a.py", out)

    def test_fix_never_changes_a_file_with_only_part_of_a_header(self):
        license_only = "# SPDX-License-Identifier: MIT\nimport os\n"
        copyright_only = "# %s\nimport os\n" % EXTERNAL_COPYRIGHT
        paths = {"src/license_only.py": self.add_file("src/license_only.py", license_only),
                 "src/copyright_only.py": self.add_file("src/copyright_only.py", copyright_only)}
        for arguments in ((), ("--fix",)):
            code, out = self.run_script(*arguments)
            self.assertEqual(code, 1, arguments)
            for rel in paths:
                self.assertIn("incomplete SPDX header", out)
                self.assertIn(rel, out)
        for path, expected in ((paths["src/license_only.py"], license_only), (paths["src/copyright_only.py"], copyright_only)):
            with open(path) as f:
                self.assertEqual(f.read(), expected)

    def test_file_with_header_passes(self):
        self.add_file("src/a.py", HASH_HEADER + "import os\n")
        self.assertEqual(self.run_script(), (0, ""))

    def test_exempt_files_are_ignored(self):
        for name in ("README.md", "config.yaml", "data.json", "LICENSE", ".gitignore", "image.png"):
            self.add_file(name, "no header here\n")
        self.assertEqual(self.run_script(), (0, ""))

    def test_file_in_a_folder_that_starts_with_a_dot_is_still_checked(self):
        # Only the file name decides if a file is a dotfile, and the name "Dockerfile" has no leading dot.
        self.add_file(".devcontainer/Dockerfile", "FROM ubuntu\n")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn(".devcontainer/Dockerfile", out)

    def test_unknown_file_type_is_an_error(self):
        self.add_file("src/a.weird", "text\n")
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("unknown file type", out)

    def test_empty_files_are_ignored(self):
        self.add_file("src/empty.py", "")
        self.add_file("src/only_blank_lines.py", "\n\n")
        self.assertEqual(self.run_script(), (0, ""))

    def test_binary_files_are_ignored(self):
        path = os.path.join(self.repo, "data.bin")
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02")
        subprocess.check_call(["git", "-C", self.repo, "add", "data.bin"])
        self.assertEqual(self.run_script(), (0, ""))

    def test_symlink_is_ignored(self):
        self.add_file("real.py", HASH_HEADER + "x = 1\n")
        os.symlink("real.py", os.path.join(self.repo, "link.py"))
        subprocess.check_call(["git", "-C", self.repo, "add", "link.py"])
        self.assertEqual(self.run_script(), (0, ""))

    def test_file_not_tracked_by_git_is_not_checked(self):
        self.add_file("tracked.py", HASH_HEADER)
        with open(os.path.join(self.repo, "untracked.py"), "w") as f:
            f.write("import os\n")
        self.assertEqual(self.run_script(), (0, ""))

    def test_fix_adds_headers_and_lists_the_fixed_files(self):
        self.add_file("src/a.py", "import os\n")
        self.add_file("src/b.cc", "int x;\n")
        self.add_file("BUILD", "load()\n")
        code, out = self.run_script("--fix")
        self.assertEqual(code, 0)
        for fixed_file in ("src/a.py", "src/b.cc", "BUILD"):
            self.assertIn("fixed   " + fixed_file, out)
        self.assertIn("Fixed 3 file(s).", out)

    def test_after_fix_the_check_passes_and_a_second_fix_changes_nothing(self):
        self.add_file("src/a.py", "import os\n")
        self.run_script("--fix")
        self.assertEqual(self.run_script(), (0, ""))
        self.assertEqual(self.run_script("--fix"), (0, ""))

    def test_fix_keeps_shebang_first_and_keeps_the_rest_of_the_file(self):
        path = self.add_file("tool.py", "#!/usr/bin/env python3\nprint('hi')\n")
        self.run_script("--fix")
        with open(path) as f:
            self.assertEqual(f.read(), "#!/usr/bin/env python3\n" + FIXED_HASH_HEADER + "\nprint('hi')\n")

    def test_fix_handles_a_shebang_without_a_newline(self):
        path = self.add_file("tool.py", "#!/usr/bin/env python3")
        code, _ = self.run_script("--fix")
        self.assertEqual(code, 0)
        with open(path) as f:
            self.assertEqual(f.read(), "#!/usr/bin/env python3\n" + FIXED_HASH_HEADER + "\n")

    def test_fix_refuses_a_file_that_is_not_utf8_and_leaves_it_unchanged(self):
        original = b"// caf\xe9\nint x;\n"
        path = os.path.join(self.repo, "a.cc")
        with open(path, "wb") as f:
            f.write(original)
        subprocess.check_call(["git", "-C", self.repo, "add", "a.cc"])
        code, out = self.run_script("--fix")
        self.assertEqual(code, 1)
        self.assertIn("not UTF-8", out)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), original)

    def test_non_utf8_file_that_already_has_the_header_passes(self):
        path = os.path.join(self.repo, "a.cc")
        with open(path, "wb") as f:
            f.write(C_HEADER.encode() + b"// caf\xe9\n")
        subprocess.check_call(["git", "-C", self.repo, "add", "a.cc"])
        self.assertEqual(self.run_script(), (0, ""))

    def test_fix_keeps_crlf_line_endings(self):
        path = os.path.join(self.repo, "a.py")
        with open(path, "wb") as f:
            f.write(b"import os\r\n")
        subprocess.check_call(["git", "-C", self.repo, "add", "a.py"])
        self.run_script("--fix")
        with open(path, "rb") as f:
            self.assertTrue(f.read().endswith(b"\n\nimport os\r\n"))

    def test_outside_holder_and_stacked_headers_pass(self):
        self.add_file("src/external.py", EXTERNAL_HEADER + "import os\n")
        self.add_file("src/stacked.py", STACKED_HEADER + "import os\n")
        self.assertEqual(self.run_script(), (0, ""))

    def test_nvidia_line_in_the_wrong_form_is_reported(self):
        text = "# SPDX-FileCopyrightText: Copyright 2024 NVIDIA Corporation\n# %s\n" % spdx_headers.LICENSE
        self.add_file("src/a.py", text)
        code, out = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("src/a.py", out)
        self.assertIn("NVIDIA copyright line", out)

    def test_fix_never_edits_a_file_that_has_a_header(self):
        for name, header in (("external.py", EXTERNAL_HEADER), ("stacked.py", STACKED_HEADER),
                             ("nvidia.py", HASH_HEADER)):
            self.add_file(name, header + "import os\n")
        self.assertEqual(self.run_script("--fix"), (0, ""))
        for name, header in (("external.py", EXTERNAL_HEADER), ("stacked.py", STACKED_HEADER),
                             ("nvidia.py", HASH_HEADER)):
            with open(os.path.join(self.repo, name)) as f:
                self.assertEqual(f.read(), header + "import os\n", name)

    def commit(self, message, year):
        """Commit what is staged, with author and committer dated in the given year."""
        date = "%d-06-15T12:00:00" % year
        env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
        subprocess.check_call(["git", "-C", self.repo, "-c", "user.name=T", "-c", "user.email=t@example.com",
                               "commit", "-q", "-m", message], env=env)

    def write_and_commit(self, relative_path, content, year):
        path = os.path.join(self.repo, relative_path)
        os.makedirs(os.path.dirname(path) or self.repo, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        subprocess.check_call(["git", "-C", self.repo, "add", relative_path])
        self.commit("change " + relative_path, year)

    def read(self, relative_path):
        with open(os.path.join(self.repo, relative_path)) as f:
            return f.read()

    def test_fix_uses_the_year_of_the_first_commit(self):
        self.write_and_commit("a.py", "import os\n", 2021)
        self.assertEqual(self.run_script("--fix", "a.py")[0], 0)
        self.assertTrue(self.read("a.py").startswith("# " + spdx_headers.nvidia_copyright(2021) + "\n"))

    def test_fix_writes_a_range_up_to_the_last_change(self):
        self.write_and_commit("a.py", "import os\n", 2021)
        self.write_and_commit("a.py", "import os\nimport sys\n", 2023)
        self.assertEqual(self.run_script("--fix", "a.py")[0], 0)
        self.assertTrue(self.read("a.py").startswith("# " + spdx_headers.nvidia_copyright(2021, 2023) + "\n"))

    def test_any_change_counts_but_a_header_only_commit_does_not(self):
        self.write_and_commit("a.py", "import os\n", 2021)
        # Adding the header, then rewriting its year, are not changes to the file.
        self.write_and_commit("a.py", HASH_HEADER + "import os\n", 2022)
        self.write_and_commit("a.py", HASH_HEADER.replace("2024", "2025") + "import os\n", 2025)
        self.assertEqual(spdx_headers.change_years(self.repo, "a.py"), [2021])
        # A one-character edit of the code is a change.
        self.write_and_commit("a.py", HASH_HEADER + "import oss\n", 2026)
        self.assertEqual(sorted(spdx_headers.change_years(self.repo, "a.py")), [2021, 2026])

    def test_an_added_or_removed_line_that_starts_with_two_marks_is_a_change(self):
        # The statements start in the first column, so a patch shows the added `++counter;` as
        # `+++counter;` and the removed `--counter;` as `---counter;`, like the file header lines.
        self.write_and_commit("added.cc", "int x;\n", 2021)
        self.write_and_commit("added.cc", "int x;\n++counter;\n", 2023)
        self.assertEqual(sorted(spdx_headers.change_years(self.repo, "added.cc")), [2021, 2023])
        self.assertEqual(spdx_headers.file_years(self.repo, "added.cc", THIS_YEAR), (2021, 2023))
        self.write_and_commit("removed.cc", "int x;\n--counter;\n", 2021)
        self.write_and_commit("removed.cc", "int x;\n", 2023)
        self.assertEqual(sorted(spdx_headers.change_years(self.repo, "removed.cc")), [2021, 2023])
        self.assertEqual(spdx_headers.file_years(self.repo, "removed.cc", THIS_YEAR), (2021, 2023))

    def test_a_conflict_resolution_in_a_merge_is_a_change_in_the_year_of_the_merge(self):
        self.write_and_commit("a.py", "base\n", 2021)
        main = subprocess.check_output(["git", "-C", self.repo, "rev-parse", "--abbrev-ref", "HEAD"],
                                       universal_newlines=True).strip()
        subprocess.check_call(["git", "-C", self.repo, "checkout", "-q", "-b", "side"])
        self.write_and_commit("a.py", "side\n", 2022)
        subprocess.check_call(["git", "-C", self.repo, "checkout", "-q", main])
        self.write_and_commit("a.py", "main\n", 2023)
        # A git identity is needed to merge. The two changes of the same line must conflict, and the merge
        # must stop there: MERGE_HEAD exists, so the next commit has two parents.
        merge = subprocess.run(["git", "-C", self.repo, "-c", "user.name=T", "-c", "user.email=t@example.com",
                                "merge", "side"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(merge.returncode, 1, merge.stdout)
        self.assertIn("CONFLICT", merge.stdout)
        subprocess.check_call(["git", "-C", self.repo, "rev-parse", "-q", "--verify", "MERGE_HEAD"],
                              stdout=subprocess.DEVNULL)
        with open(os.path.join(self.repo, "a.py"), "w") as f:
            f.write("resolved by hand\n")
        subprocess.check_call(["git", "-C", self.repo, "add", "a.py"])
        self.commit("merge side", 2024)
        self.assertEqual(sorted(spdx_headers.change_years(self.repo, "a.py")), [2021, 2022, 2023, 2024])
        self.assertEqual(self.run_script("--fix", "a.py")[0], 0)
        self.assertTrue(self.read("a.py").startswith("# " + spdx_headers.nvidia_copyright(2021, 2024) + "\n"))

    def test_a_clean_merge_is_not_a_change(self):
        self.write_and_commit("a.py", "one\ntwo\nthree\n", 2021)
        main = subprocess.check_output(["git", "-C", self.repo, "rev-parse", "--abbrev-ref", "HEAD"],
                                       universal_newlines=True).strip()
        subprocess.check_call(["git", "-C", self.repo, "checkout", "-q", "-b", "side"])
        self.write_and_commit("a.py", "ONE\ntwo\nthree\n", 2022)
        subprocess.check_call(["git", "-C", self.repo, "checkout", "-q", main])
        self.write_and_commit("a.py", "one\ntwo\nTHREE\n", 2023)
        date = "2024-06-15T12:00:00"
        subprocess.check_call(["git", "-C", self.repo, "-c", "user.name=T", "-c", "user.email=t@example.com",
                               "merge", "-q", "--no-ff", "-m", "merge side", "side"],
                              env=dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date))
        self.assertEqual(sorted(spdx_headers.change_years(self.repo, "a.py")), [2021, 2022, 2023])

    def test_fix_refuses_to_write_years_from_a_shallow_clone(self):
        self.write_and_commit("a.py", "import os\n", 2021)
        self.write_and_commit("a.py", "import os\nimport sys\n", 2023)
        with tempfile.TemporaryDirectory() as parent:
            clone = os.path.join(os.path.realpath(parent), "clone")
            subprocess.check_call(["git", "clone", "-q", "--depth", "1", "file://" + self.repo, clone],
                                  stderr=subprocess.DEVNULL)
            code, out = self.run_script("--fix", "a.py", from_folder=clone)
            self.assertEqual(code, 1)
            self.assertIn("git fetch --unshallow", out)
            with open(os.path.join(clone, "a.py")) as f:
                self.assertEqual(f.read(), "import os\nimport sys\n")
            # Checking, without --fix, still works in a shallow clone.
            self.assertEqual(self.run_script(from_folder=clone)[0], 1)
            # A file that has no history yet gets its header: its year is this year.
            with open(os.path.join(clone, "new.py"), "w") as f:
                f.write("import re\n")
            subprocess.check_call(["git", "-C", clone, "add", "new.py"])
            self.assertEqual(self.run_script("--fix", "new.py", from_folder=clone)[0], 0)
            with open(os.path.join(clone, "new.py")) as f:
                self.assertTrue(f.read().startswith(FIXED_HASH_HEADER))
        # A rename that is staged but not committed has no history under the new name: the old one counts.
        with tempfile.TemporaryDirectory() as parent:
            clone = os.path.join(os.path.realpath(parent), "clone")
            subprocess.check_call(["git", "clone", "-q", "--depth", "1", "file://" + self.repo, clone],
                                  stderr=subprocess.DEVNULL)
            subprocess.check_call(["git", "-C", clone, "mv", "a.py", "b.py"])
            code, out = self.run_script("--fix", "b.py", from_folder=clone)
            self.assertEqual(code, 1)
            self.assertIn("git fetch --unshallow", out)
            with open(os.path.join(clone, "b.py")) as f:
                self.assertEqual(f.read(), "import os\nimport sys\n")
        # The same files in a full clone are fixed with the range of their history.
        self.assertEqual(self.run_script("--fix", "a.py")[0], 0)
        self.assertTrue(self.read("a.py").startswith("# " + spdx_headers.nvidia_copyright(2021, 2023) + "\n"))

    def test_fix_follows_a_rename(self):
        self.write_and_commit("old.py", "import os\n", 2020)
        subprocess.check_call(["git", "-C", self.repo, "mv", "old.py", "new.py"])
        self.commit("rename", 2022)
        self.assertEqual(self.run_script("--fix", "new.py")[0], 0)
        # The rename is not a change, so the range ends with the creation year.
        self.assertTrue(self.read("new.py").startswith("# " + spdx_headers.nvidia_copyright(2020) + "\n"))

    def test_fix_follows_a_rename_that_is_staged_but_not_committed(self):
        self.write_and_commit("old.py", "import os\nimport sys\nimport re\nimport json\n", 2020)
        subprocess.check_call(["git", "-C", self.repo, "mv", "old.py", "new.py"])
        self.assertEqual(self.run_script("--fix", "new.py")[0], 0)
        # The rename is not a change: the year of the creation, not the current year.
        self.assertTrue(self.read("new.py").startswith("# " + spdx_headers.nvidia_copyright(2020) + "\n"))

    def test_fix_counts_an_edit_made_with_a_staged_rename_as_a_change_this_year(self):
        self.write_and_commit("old.py", "import os\nimport sys\nimport re\nimport json\n", 2020)
        subprocess.check_call(["git", "-C", self.repo, "mv", "old.py", "new.py"])
        with open(os.path.join(self.repo, "new.py"), "a") as f:
            f.write("import time\n")
        self.assertEqual(self.run_script("--fix", "new.py")[0], 0)
        self.assertTrue(self.read("new.py").startswith("# " + spdx_headers.nvidia_copyright(2020, THIS_YEAR) + "\n"))

    def test_fix_reads_history_when_git_colors_its_diffs(self):
        self.write_and_commit("a.py", "import os\n", 2021)
        subprocess.check_call(["git", "-C", self.repo, "config", "color.diff", "always"])
        self.assertEqual(spdx_headers.change_years(self.repo, "a.py"), [2021])
        self.assertEqual(self.run_script("--fix", "a.py")[0], 0)
        self.assertTrue(self.read("a.py").startswith("# " + spdx_headers.nvidia_copyright(2021) + "\n"))

    def test_fix_counts_an_uncommitted_edit_as_a_change_this_year(self):
        self.write_and_commit("a.py", "import os\n", 2021)
        with open(os.path.join(self.repo, "a.py"), "a") as f:
            f.write("import sys\n")
        self.assertEqual(self.run_script("--fix", "a.py")[0], 0)
        self.assertTrue(self.read("a.py").startswith("# " + spdx_headers.nvidia_copyright(2021, THIS_YEAR) + "\n"))

    def test_fix_uses_the_current_year_for_a_new_file(self):
        self.add_file("a.py", "import os\n")
        self.assertEqual(self.run_script("--fix")[0], 0)
        self.assertTrue(self.read("a.py").startswith(FIXED_HASH_HEADER))

    def test_relative_file_path_is_resolved_from_the_current_folder(self):
        # Two files miss the header: sub/a.py and other.py.
        self.add_file("sub/a.py", "import os\n")
        self.add_file("other.py", "import os\n")
        # We stand inside the folder "sub" and name the file "a.py", which is relative to "sub".
        code, out = self.run_script("a.py", from_folder=os.path.join(self.repo, "sub"))
        # Only a.py is checked. It is reported with its path from the repo root.
        self.assertEqual(code, 1)
        self.assertIn("sub/a.py", out)
        self.assertNotIn("other.py", out)

    def test_absolute_file_path_works_from_any_folder(self):
        absolute_path = self.add_file("sub/a.py", "import os\n")
        # We stand inside "sub" and give the full path of the file.
        code, out = self.run_script(absolute_path, from_folder=os.path.join(self.repo, "sub"))
        self.assertEqual(code, 1)
        self.assertIn("sub/a.py", out)


if __name__ == "__main__":
    unittest.main()
