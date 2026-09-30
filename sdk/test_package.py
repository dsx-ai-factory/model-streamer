import hashlib
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import package


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bazel = self.root / "bazel-bin"
        self.output = self.root / "dist"

    def libraries(self, arch):
        for library, target in package.LIBRARIES.items():
            path = self.bazel / target / library
            path.parent.mkdir(parents=True, exist_ok=True)
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            struct.pack_into("<HH", header, 16, 3, package.MACHINES[arch])
            path.write_bytes(header)

    @patch.object(package.subprocess, "check_output", return_value="(RUNPATH) Library runpath: [$ORIGIN]")
    def test_both_archives_have_portable_checksums_and_complete_contents(self, _readelf):
        for arch in package.MACHINES:
            self.libraries(arch)
            archive = package.package("v0.17.0", arch, self.bazel, self.output)
            checksum = archive.with_name(archive.name + ".sha256").read_text()
            self.assertEqual(checksum, "{}  {}\n".format(
                hashlib.sha256(archive.read_bytes()).hexdigest(), archive.name))
            with tarfile.open(archive) as tar:
                root = archive.name[:-7]
                names = tar.getnames()
                self.assertTrue(all(n.startswith(root + "/") for n in names))
                self.assertTrue(all(member.isfile() for member in tar.getmembers()))
                for header in ("streamer.h", "device.h", "response_code.h", "submission_id.h"):
                    self.assertIn(root + "/include/streamer/" + header, names)
                for library in package.LIBRARIES:
                    self.assertIn(root + "/lib/" + library, names)
                self.assertEqual(tar.extractfile(root + "/VERSION").read(), b"0.17.0\n")
                for name in names:
                    if name.endswith((".pc", ".cmake")):
                        data = tar.extractfile(name).read()
                        self.assertNotIn(str(package.ROOT).encode(), data)
                        self.assertNotIn(b"@VERSION@", data)

    def test_rejects_wrong_architecture_before_writing_an_archive(self):
        self.libraries("aarch64")
        with self.assertRaisesRegex(ValueError, "x86_64"):
            package.package("0.17.0", "x86_64", self.bazel, self.output)
        self.assertFalse(self.output.exists())

    @patch.object(package.subprocess, "check_output", return_value="(RUNPATH) Library runpath: [/build/lib]")
    def test_rejects_missing_origin(self, _readelf):
        self.libraries("x86_64")
        with self.assertRaisesRegex(ValueError, r"\$ORIGIN"):
            package.package("0.17.0", "x86_64", self.bazel, self.output)

    @patch.object(package.subprocess, "check_output", return_value="(RUNPATH) Library runpath: [$ORIGIN]")
    def test_rejects_missing_plugin(self, _readelf):
        self.libraries("x86_64")
        # An empty plugin is an invalid build output, not a valid SDK component.
        (self.bazel / "azure/libstreamerazure.so").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "libstreamerazure"):
            package.package("0.17.0", "x86_64", self.bazel, self.output)

    def test_rejects_unsafe_version(self):
        for version in ("../0.17.0", "", "$(touch bad)"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                package.package(version, "x86_64", self.bazel, self.output)


if __name__ == "__main__":
    unittest.main()
