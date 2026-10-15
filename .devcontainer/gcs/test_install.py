# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that SDK installation survives removal of the Bazel execution root."""

import json
from pathlib import Path
import tempfile
import unittest

from install import install, relative_path


class InstallTest(unittest.TestCase):
    def test_relocates_generated_headers_and_preserves_link_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution_root = root / "execroot"
            headers = [
                "external/google_cloud_cpp/google/cloud/storage/client.h",
                "bazel-out/opt/bin/external/protobuf/_virtual_includes/protos/message.pb.h",
            ]
            archive = "bazel-out/opt/bin/external/grpc/libgrpc.pic.a"
            always_archive = "bazel-out/opt/bin/external/cares/libares.lo"
            for path in headers + [archive, always_archive]:
                source = execution_root / path
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(path)
            actual_header = root / "generated-header"
            actual_header.write_text("generated protobuf header")
            # Model the virtual include tree as Bazel creates it.
            virtual_header = "bazel-out/opt/bin/_virtual_includes/protos/virtual.pb.h"
            (execution_root / virtual_header).parent.mkdir(parents=True, exist_ok=True)
            (execution_root / virtual_header).symlink_to(actual_header)
            headers.append(virtual_header)
            manifest = {
                "source_checksums": {"google_cloud_cpp": "sdk-sha", "protobuf": "protobuf-sha"},
                "headers": headers,
                "libraries": [
                    {"path": archive, "alwayslink": False},
                    {"path": always_archive, "alwayslink": True},
                ],
                "includes": ["external/google_cloud_cpp", "bazel-out/opt/bin/_virtual_includes/protos"],
                "defines": ["GOOGLE_CLOUD_CPP_HAVE_GRPC=1"],
                "linkopts": ["-pthread", "-ldl"],
            }
            destination = root / "installed" / "aarch64"
            install(manifest, execution_root, destination)
            execution_root.rename(root / "old-execroot")
            actual_header.rename(root / "old-generated-header")
            self.assertEqual((destination / "tree" / virtual_header).read_text(), "generated protobuf header")
            self.assertFalse((destination / "tree" / virtual_header).is_symlink())
            self.assertEqual((destination / "tree" / archive).read_text(), archive)
            self.assertEqual((destination / "tree" / (always_archive + ".a")).read_text(), always_archive)
            build = (destination / "BUILD.bazel").read_text()
            self.assertIn("alwayslink = True", build)
            self.assertIn("alwayslink = False", build)
            self.assertIn("libares.lo.a", build)
            self.assertIn('"tree/bazel-out/opt/bin/_virtual_includes/protos"', build)
            self.assertIn('defines = ["GOOGLE_CLOUD_CPP_HAVE_GRPC=1"]', build)
            self.assertIn('linkopts = ["-pthread", "-ldl"]', build)
            self.assertNotIn(str(execution_root), build)
            self.assertEqual(json.loads((destination / "manifest.json").read_text()), manifest)

    def test_rejects_paths_outside_execution_root(self):
        for path in ["/tmp/header.h", "../header.h", "external/../../header.h"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(path)

    def test_missing_archive_fails_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"headers": [], "libraries": [{"path": "missing.a", "alwayslink": False}]}
            with self.assertRaises(FileNotFoundError):
                install(manifest, root, root / "installed")


if __name__ == "__main__":
    unittest.main()
