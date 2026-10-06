# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for spdx_headers.py. Run with: python3 scripts/test_spdx_headers.py"""
# [explain] unittest is part of the Python standard library, so the tests need no install and no pytest.

import os
import subprocess
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(SCRIPTS_DIR, "spdx_headers.py")
sys.path.insert(0, SCRIPTS_DIR)
import spdx_headers  # noqa: E402

# [explain] The header as the script writes it, for use in test files.
HASH_HEADER = "# %s\n# %s\n" % (spdx_headers.COPYRIGHT, spdx_headers.LICENSE)
C_HEADER = "/*\n * %s\n * %s\n */\n" % (spdx_headers.COPYRIGHT, spdx_headers.LICENSE)


class HelperTests(unittest.TestCase):
    """Tests of the small functions. No files are involved."""

    def test_has_header_hash_style(self):
        self.assertTrue(spdx_headers.has_header(HASH_HEADER + "import os\n"))

    def test_has_header_c_style(self):
        self.assertTrue(spdx_headers.has_header(C_HEADER + "int x;\n"))

    def test_has_header_missing(self):
        self.assertFalse(spdx_headers.has_header("import os\n"))

    def test_has_header_needs_both_lines(self):
        self.assertFalse(spdx_headers.has_header("# %s\n" % spdx_headers.COPYRIGHT))
        self.assertFalse(spdx_headers.has_header("# %s\n" % spdx_headers.LICENSE))

    def test_has_header_wrong_year(self):
        wrong_year = HASH_HEADER.replace("2024", "2019")
        self.assertFalse(spdx_headers.has_header(wrong_year))

    def test_has_header_must_be_near_the_top(self):
        # [explain] A header that starts after WINDOW lines of code does not count.
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
        result = spdx_headers.add_header("import os\n", "hash")
        self.assertEqual(result, HASH_HEADER + "\nimport os\n")

    def test_add_header_c(self):
        result = spdx_headers.add_header("int x;\n", "c")
        self.assertEqual(result, C_HEADER + "\nint x;\n")

    def test_add_header_keeps_shebang_first(self):
        result = spdx_headers.add_header("#!/usr/bin/env python3\nimport os\n", "hash")
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

    def test_file_with_header_passes(self):
        self.add_file("src/a.py", HASH_HEADER + "import os\n")
        self.assertEqual(self.run_script(), (0, ""))

    def test_exempt_files_are_ignored(self):
        for name in ("README.md", "config.yaml", "data.json", "LICENSE", ".gitignore", "image.png"):
            self.add_file(name, "no header here\n")
        self.assertEqual(self.run_script(), (0, ""))

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
            self.assertEqual(f.read(), "#!/usr/bin/env python3\n" + HASH_HEADER + "\nprint('hi')\n")

    def test_relative_file_path_is_resolved_from_the_current_folder(self):
        # [explain] Two files miss the header: sub/a.py and other.py.
        self.add_file("sub/a.py", "import os\n")
        self.add_file("other.py", "import os\n")
        # [explain] We stand inside the folder "sub" and name the file "a.py", which is relative to "sub".
        code, out = self.run_script("a.py", from_folder=os.path.join(self.repo, "sub"))
        # [explain] Only a.py is checked. It is reported with its path from the repo root.
        self.assertEqual(code, 1)
        self.assertIn("sub/a.py", out)
        self.assertNotIn("other.py", out)

    def test_absolute_file_path_works_from_any_folder(self):
        absolute_path = self.add_file("sub/a.py", "import os\n")
        # [explain] We stand inside "sub" and give the full path of the file.
        code, out = self.run_script(absolute_path, from_folder=os.path.join(self.repo, "sub"))
        self.assertEqual(code, 1)
        self.assertIn("sub/a.py", out)


if __name__ == "__main__":
    unittest.main()
