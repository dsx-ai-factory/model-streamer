# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that SDK installation survives removal of the Bazel execution root."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from install import build_metadata, install, relative_path


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

    def test_records_sdk_inputs_without_application_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gcs"
            contents = {
                ".bazelrc": "build --compilation_mode=opt\n",
                ".bazelversion": "7.6.1\n",
                "MODULE.bazel": "SDK modules\n",
                "BUILD.bazel": "SDK targets\n",
                "abi_check.cc": "ABI check\n",
                "../scripts/build-gcs-sdk.sh": "SDK build command\n",
                "export.bzl": "SDK exporter\n",
                "install.py": "SDK installer\n",
                "config/MODULE.bazel": "SDK configuration module\n",
                "config/model_streamer_gcs_sdk/compatibility.h": "ABI contract\n",
            }
            for name, content in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            # Application edits must not become part of the SDK identity.
            application = root.parent / "cpp"
            application.mkdir()
            (application / ".bazelrc").write_text("build --copt=-g\n")
            (application / ".bazelversion").write_text("different application Bazel\n")
            with patch("install.subprocess.check_output", return_value="GCC version\n") as run:
                metadata = build_metadata(root, "/usr/bin/g++")
            self.assertEqual(metadata["build_inputs"], contents)
            self.assertEqual(metadata["compiler"], "/usr/bin/g++")
            self.assertEqual(metadata["compiler_version"], "GCC version\n")
            run.assert_called_once_with(["/usr/bin/g++", "--version"], text=True)
            with patch("install.subprocess.check_output", return_value="GCC version\n"):
                (application / ".bazelrc").write_text("build --copt=-O1\n")
                self.assertEqual(build_metadata(root, "/usr/bin/g++"), metadata)
                script = root.parent / "scripts/build-gcs-sdk.sh"
                script.write_text("updated SDK build command\n")
                self.assertNotEqual(build_metadata(root, "/usr/bin/g++"), metadata)
                script.write_text(contents["../scripts/build-gcs-sdk.sh"])
                (root / "config/MODULE.bazel").write_text("updated SDK configuration\n")
                self.assertNotEqual(build_metadata(root, "/usr/bin/g++"), metadata)
                (root / "config/MODULE.bazel").write_text(contents["config/MODULE.bazel"])
                # A google-cloud-cpp bump is a MODULE.bazel change: the SDK must be rebuilt.
                (root / "MODULE.bazel").write_text("updated SDK modules\n")
                self.assertNotEqual(build_metadata(root, "/usr/bin/g++"), metadata)

    def test_missing_archive_fails_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"headers": [], "libraries": [{"path": "missing.a", "alwayslink": False}]}
            with self.assertRaises(FileNotFoundError):
                install(manifest, root, root / "installed")


if __name__ == "__main__":
    unittest.main()
