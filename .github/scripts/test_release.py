# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise release asset packaging, verification, and tag validation locally."""

import hashlib
import io
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest

VERSION = "0.17.2"
BUNDLE_SCRIPT = Path(__file__).with_name("release_bundle.py").resolve()


class ReleaseBundleTests(unittest.TestCase):
    def setUp(self):
        """Create wheel, ELF library, header, license, and notice fixtures for both architectures."""
        self.root = Path(tempfile.mkdtemp(prefix="release-bundle-test-"))
        self.bundle = self.root / "release-bundle"
        self.paths = []
        for package in ("runai_model_streamer", "runai_model_streamer_s3",
                        "runai_model_streamer_gcs", "runai_model_streamer_azure"):
            directory = self.root / "packages" / package / "dist"
            directory.mkdir(parents=True)
            for arch in ("x86_64", "aarch64"):
                path = directory / f"{package}-{VERSION}-py3-none-manylinux_2_28_{arch}.whl"
                path.write_bytes(b"fixture wheel content")
                self.paths.append(path)
        self.notices = self.root / "third_party_notices" / "sdk.txt"
        self.notices.parent.mkdir()
        self.notices.write_text("fixture third-party notices\n")
        self.cpp = self.root / "cpp-artifacts"
        self.libraries = []
        for arch, machine in (("x86_64", 62), ("aarch64", 183)):
            for component, library in (("streamer", "libstreamer.so"), ("s3", "libstreamers3.so"),
                                       ("gcs", "libstreamergcs.so"), ("azure", "libstreamerazure.so")):
                directory = self.cpp / f"ci-cpp-{component}-{arch}"
                (directory / "include" / "streamer").mkdir(parents=True)
                (directory / "lib").mkdir()
                (directory / "LICENSE").write_text("fixture license\n")
                for header in ("streamer.h", "device.h", "response_code.h", "submission_id.h"):
                    (directory / "include" / "streamer" / header).write_text(f"// {header}\n")
                # Only the ELF header is needed for packaging validation; not executable code.
                elf = bytearray(64)
                elf[:6] = b"\x7fELF\x02\x01"
                struct.pack_into("<HH", elf, 16, 3, machine)
                path = directory / "lib" / library
                path.write_bytes(elf + f"{component}-{arch}".encode())
                self.libraries.append(path)

    def collect(self, bundle="release-bundle", version=VERSION, extra_args=()):
        """Run bundle creation with the fixture inputs and return the captured process result."""
        return subprocess.run([sys.executable, str(BUNDLE_SCRIPT), "create", "--version", version,
                               "--bundle", bundle, *extra_args],
                              cwd=self.root, capture_output=True, text=True)

    def verify(self, bundle="release-bundle", version=VERSION, extra_args=()):
        """Run bundle verification and return the captured process result."""
        return subprocess.run([sys.executable, str(BUNDLE_SCRIPT), "verify", "--version", version,
                               "--bundle", bundle, *extra_args],
                              cwd=self.root, capture_output=True, text=True)

    def collect_ok(self):
        """Create the default release bundle and assert that packaging succeeds."""
        result = self.collect()
        self.assertEqual(result.returncode, 0, result.stderr)

    def archive_path(self, arch="x86_64"):
        return self.bundle / "cpp" / f"runai-model-streamer-cpp-sdk-{VERSION}-linux-{arch}.tar.gz"

    def test_complete_bundle_and_checksums(self):
        self.collect_ok()
        self.assertEqual(len(list((self.bundle / "wheels").glob("*.whl"))), 8)
        self.assertEqual(len(list((self.bundle / "cpp").glob("*.tar.gz"))), 2)
        self.assertEqual(len((self.bundle / "SHA256SUMS").read_text().splitlines()), 10)
        verified = self.verify()
        self.assertEqual(verified.returncode, 0, verified.stderr)
        for path in self.paths:
            self.assertEqual(path.read_bytes(),
                             (self.bundle / "wheels" / path.name).read_bytes())

    def test_archive_layout_contents_and_metadata(self):
        """Check SDK archive paths, file contents, and normalized metadata for both architectures."""
        self.collect_ok()
        for arch in ("x86_64", "aarch64"):
            root = f"runai-model-streamer-cpp-sdk-{VERSION}-linux-{arch}"
            expected = {}
            for directory in self.cpp.glob(f"ci-cpp-*-{arch}"):
                expected.update({f"{root}/{path.relative_to(directory).as_posix()}": path.read_bytes()
                                 for path in directory.rglob("*") if path.is_file()})
            expected[f"{root}/THIRD_PARTY_NOTICES.txt"] = self.notices.read_bytes()
            with tarfile.open(self.archive_path(arch)) as archive:
                self.assertEqual({member.name: archive.extractfile(member).read()
                                  for member in archive.getmembers() if member.isfile()}, expected)
                for member in archive.getmembers():
                    self.assertTrue(member.isdir() or member.isfile())
                    self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
                    self.assertEqual((member.uname, member.gname), ("", ""))
                    self.assertEqual(member.mode, 0o755 if member.isdir() else 0o644)

    def test_development_bundle_matches_release_layout(self):
        """Check PR and release asset parity, successful PR verification, and tampering detection."""
        self.collect_ok()
        version = "0.0.0+dev.abc1234"
        for path in self.paths:
            path.rename(path.with_name(path.name.replace(VERSION, version)))
        options = {"bundle": "pr-bundle", "version": version,
                   "extra_args": ("--allow-development-version",)}
        result = self.collect(**options)
        self.assertEqual(result.returncode, 0, result.stderr)
        pr_bundle = self.root / "pr-bundle"
        release_files = {path.relative_to(self.bundle).as_posix(): path
                         for path in self.bundle.rglob("*") if path.is_file()}
        pr_files = {path.relative_to(pr_bundle).as_posix().replace(version, VERSION): path
                    for path in pr_bundle.rglob("*") if path.is_file()}
        self.assertEqual(release_files.keys(), pr_files.keys())
        for name, release_path in release_files.items():
            if name.startswith("wheels/"):
                self.assertEqual(release_path.read_bytes(), pr_files[name].read_bytes())
            elif name.startswith("cpp/"):
                with tarfile.open(release_path) as archive:
                    release_members = {member.name: archive.extractfile(member).read()
                                       for member in archive.getmembers() if member.isfile()}
                with tarfile.open(pr_files[name]) as archive:
                    pr_members = {member.name.replace(version, VERSION): archive.extractfile(member).read()
                                  for member in archive.getmembers() if member.isfile()}
                self.assertEqual(release_members, pr_members)
        result = self.verify(**options)
        self.assertEqual(result.returncode, 0, result.stderr)
        next((pr_bundle / "wheels").glob("*.whl")).write_bytes(b"changed")
        result = self.verify(**options)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum mismatch", result.stderr)

    def test_development_version_requires_opt_in(self):
        """Reject development versions during creation and verification unless explicitly allowed."""
        for operation in (self.collect, self.verify):
            with self.subTest(operation=operation.__name__):
                result = operation(version="0.0.0+dev.abc1234")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("form X.Y.Z", result.stderr)

    def test_invalid_development_versions_rejected(self):
        """Reject malformed development versions even when development versions are enabled."""
        for version in ("0.0.0+dev.abc123", "0.0.0+dev.abc12345", "0.0.0+dev.xyz1234",
                        "0.17.2+dev.abc1234", "0.0.0+dev.abc1234/../../output", "0.0.0+dev.abc1234\n"):
            for operation in (self.collect, self.verify):
                with self.subTest(version=version, operation=operation.__name__):
                    result = operation(version=version, extra_args=("--allow-development-version",))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("form X.Y.Z", result.stderr)

    def test_checksum_manifest_works_for_flat_downloads(self):
        """Check that SHA256SUMS verifies assets downloaded into a single directory."""
        self.collect_ok()
        downloads = self.root / "downloads"
        downloads.mkdir()
        for path in self.bundle.rglob("*"):
            if path.is_file():
                shutil.copyfile(path, downloads / path.name)
        result = subprocess.run(["sha256sum", "--check", "SHA256SUMS"], cwd=downloads,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repackaging_ignores_source_metadata(self):
        self.collect_ok()
        for path in self.cpp.rglob("*"):
            if path.is_file():
                path.chmod(0o755)
                os.utime(path, (1700000000, 1700000000))
        result = self.collect("second-bundle")
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in self.bundle.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(),
                                 (self.root / "second-bundle" / path.relative_to(self.bundle)).read_bytes())

    def test_missing_architecture_rejected(self):
        # Keep the fixture file, but make it unavailable to the wheel glob.
        self.paths[0].rename(self.paths[0].with_suffix(".unavailable"))
        self.assertNotEqual(self.collect().returncode, 0)

    def test_wrong_version_rejected(self):
        path = self.paths[0]
        path.rename(path.with_name(path.name.replace(VERSION, "0.17.1")))
        self.assertNotEqual(self.collect().returncode, 0)

    def test_duplicate_platform_wheel_rejected(self):
        path = self.paths[0]
        path.with_name(path.name.replace("manylinux_2_28", "manylinux2014")).write_bytes(b"duplicate")
        self.assertNotEqual(self.collect().returncode, 0)

    def test_checksum_detects_modified_wheel(self):
        """Reject a bundle when a wheel no longer matches its recorded checksum."""
        self.collect_ok()
        (self.bundle / "wheels" / self.paths[0].name).write_bytes(b"changed")
        self.assertNotEqual(self.verify().returncode, 0)

    def test_missing_component_rejected(self):
        """Reject packaging when a required component artifact directory is missing."""
        path = self.cpp / "ci-cpp-azure-aarch64"
        path.rename(path.with_name("unavailable"))
        self.assertNotEqual(self.collect().returncode, 0)

    def test_missing_library_rejected(self):
        """Reject packaging when a component artifact lacks its expected shared library."""
        self.libraries[0].rename(self.libraries[0].with_suffix(".unavailable"))
        self.assertNotEqual(self.collect().returncode, 0)

    def test_missing_header_rejected(self):
        """Reject packaging when an artifact lacks a required public header."""
        path = self.cpp / "ci-cpp-streamer-x86_64" / "include" / "streamer" / "device.h"
        path.rename(path.with_suffix(".unavailable"))
        self.assertNotEqual(self.collect().returncode, 0)

    def test_different_headers_rejected(self):
        """Reject packaging when public header contents differ between component artifacts."""
        path = self.cpp / "ci-cpp-azure-aarch64" / "include" / "streamer" / "device.h"
        path.write_text("different header")
        self.assertNotEqual(self.collect().returncode, 0)

    def test_different_license_rejected(self):
        """Reject packaging when license contents differ between component artifacts."""
        (self.cpp / "ci-cpp-gcs-aarch64" / "LICENSE").write_text("different license")
        self.assertNotEqual(self.collect().returncode, 0)

    def test_missing_third_party_notices_rejected(self):
        """Reject packaging when the third-party notice input is missing."""
        self.notices.unlink()
        self.assertNotEqual(self.collect().returncode, 0)

    def test_sdk_without_third_party_notices_rejected(self):
        self.collect_ok()
        path = self.archive_path()
        with tarfile.open(path) as archive:
            members = [(member, archive.extractfile(member).read() if member.isfile() else None)
                       for member in archive.getmembers() if not member.name.endswith("THIRD_PARTY_NOTICES.txt")]
        with tarfile.open(path, "w:gz") as archive:
            for member, data in members:
                archive.addfile(member, None if data is None else io.BytesIO(data))
        manifest = self.bundle / "SHA256SUMS"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text("".join(f"{digest}  {path.name}\n" if path.name in line else line
                                    for line in manifest.read_text().splitlines(keepends=True)))
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("THIRD_PARTY_NOTICES.txt", result.stderr)

    def test_sdk_with_a_link_as_third_party_notices_rejected(self):
        self.collect_ok()
        path = self.archive_path()
        with tarfile.open(path) as archive:
            members = [(member, archive.extractfile(member).read() if member.isfile() else None)
                       for member in archive.getmembers() if not member.name.endswith("THIRD_PARTY_NOTICES.txt")]
        with tarfile.open(path, "w:gz") as archive:
            for member, data in members:
                archive.addfile(member, None if data is None else io.BytesIO(data))
            link = tarfile.TarInfo(members[0][0].name.split("/")[0] + "/THIRD_PARTY_NOTICES.txt")
            link.type, link.linkname = tarfile.SYMTYPE, "LICENSE"
            archive.addfile(link)
        manifest = self.bundle / "SHA256SUMS"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text("".join(f"{digest}  {path.name}\n" if path.name in line else line
                                    for line in manifest.read_text().splitlines(keepends=True)))
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("regular, non-empty THIRD_PARTY_NOTICES.txt", result.stderr)

    def test_wrong_architecture_rejected(self):
        self.libraries[0].write_bytes(self.libraries[4].read_bytes())
        self.assertNotEqual(self.collect().returncode, 0)

    def test_non_elf_library_rejected(self):
        self.libraries[0].write_bytes(b"not an ELF shared library")
        self.assertNotEqual(self.collect().returncode, 0)

    def test_symlink_library_rejected(self):
        path = self.libraries[0]
        real = path.with_suffix(".real")
        path.rename(real)
        path.symlink_to(real)
        self.assertNotEqual(self.collect().returncode, 0)

    def test_checksum_detects_modified_sdk(self):
        self.collect_ok()
        with self.archive_path().open("ab") as stream:
            stream.write(b"changed")
        self.assertNotEqual(self.verify().returncode, 0)

    def test_missing_sdk_rejected_even_without_checksum_entry(self):
        self.collect_ok()
        path = self.archive_path()
        path.rename(self.root / path.name)
        manifest = self.bundle / "SHA256SUMS"
        manifest.write_text("".join(line for line in manifest.read_text().splitlines(keepends=True)
                                    if path.name not in line))
        self.assertNotEqual(self.verify().returncode, 0)

    def test_missing_bundled_wheel_rejected(self):
        self.collect_ok()
        path = self.bundle / "wheels" / self.paths[0].name
        path.rename(self.root / path.name)
        self.assertNotEqual(self.verify().returncode, 0)

    def test_incomplete_checksum_manifest_rejected(self):
        self.collect_ok()
        path = self.bundle / "SHA256SUMS"
        path.write_text("".join(path.read_text().splitlines(keepends=True)[1:]))
        self.assertNotEqual(self.verify().returncode, 0)

    def test_duplicate_checksum_entry_rejected(self):
        self.collect_ok()
        path = self.bundle / "SHA256SUMS"
        lines = path.read_text().splitlines(keepends=True)
        lines[-1] = lines[0]
        path.write_text("".join(lines))
        self.assertNotEqual(self.verify().returncode, 0)

    def test_unexpected_asset_rejected_even_with_valid_checksum(self):
        self.collect_ok()
        path = self.bundle / "wheels" / "unexpected.whl"
        path.write_bytes(b"unexpected")
        with (self.bundle / "SHA256SUMS").open("a") as stream:
            stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
        self.assertNotEqual(self.verify().returncode, 0)


class ReleaseTagTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="release-tag-test-"))
        self.git("init", "-b", "master")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release@example.com")
        self.git("commit", "--allow-empty", "-m", "Initial")
        self.git("branch", "release/0.17")
        self.git("remote", "add", "origin", str(self.root))
        self.git("tag", "-a", VERSION, "-m", "Release")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True)

    def validate(self, version=VERSION):
        output = self.root / "output"
        return subprocess.run(["bash", str(Path(__file__).with_name("validate-release.sh").resolve())],
                              cwd=self.root, capture_output=True, text=True,
                              env={**os.environ, "GITHUB_REF_NAME": version, "GITHUB_REF_TYPE": "tag",
                                   "GITHUB_OUTPUT": str(output)})

    def test_annotated_tag_on_release_branch(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("branch=release/0.17", (self.root / "output").read_text())

    def test_commit_outside_release_branch_rejected(self):
        self.git("commit", "--allow-empty", "-m", "Master only")
        self.git("tag", "0.17.3")
        result = self.validate("0.17.3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must point", result.stdout)

    def test_missing_release_branch_rejected(self):
        self.git("tag", "0.18.0")
        self.assertNotEqual(self.validate("0.18.0").returncode, 0)

    def test_lightweight_tag_on_release_branch(self):
        self.git("tag", "0.17.3")
        self.assertEqual(self.validate("0.17.3").returncode, 0)

    def test_tag_different_from_checked_out_commit_rejected(self):
        self.git("commit", "--allow-empty", "-m", "Later commit")
        self.assertNotEqual(self.validate().returncode, 0)

    def test_invalid_tag_rejected_before_fetch(self):
        for version in ("v0.17.2", "0.17.2rc1", "0.17.02", "0.17.2; false"):
            with self.subTest(version=version):
                result = self.validate(version)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("form X.Y.Z", result.stdout)


if __name__ == "__main__":
    unittest.main()
