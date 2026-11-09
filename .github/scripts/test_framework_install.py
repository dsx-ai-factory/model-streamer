# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check dependency protection and candidate installation failure handling."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import framework_install as check


class FrameworkInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.baseline = self.directory / "framework-baseline.json"
        self.baseline.write_text(json.dumps({"torch": "2.14.0"}))
        self.version = "0.0.0+dev.abc1234"

    def test_snapshot_normalizes_names_and_excludes_candidates(self):
        distributions = [
            SimpleNamespace(metadata={"Name": "Torch"}, version="2.14.0"),
            SimpleNamespace(metadata={"Name": "some_package"}, version="1.2"),
            SimpleNamespace(metadata={"Name": "runai_model_streamer_s3"}, version="0.17.1"),
        ]
        with patch.object(check.metadata, "distributions", return_value=distributions):
            check.snapshot(self.directory)
        self.assertEqual(json.loads(self.baseline.read_text()),
                         {"torch": "2.14.0", "some-package": "1.2"})
        self.assertEqual((self.directory / "framework-constraints.txt").read_text(),
                         "some-package==1.2\ntorch==2.14.0\n")

    def test_verify_accepts_candidate_versions_and_unchanged_framework(self):
        versions = {name: self.version for name in check.CANDIDATES}
        versions["torch"] = "2.14.0"
        with patch.object(check.metadata, "version", side_effect=versions.__getitem__):
            check.verify(self.directory, self.version)

    def test_verify_rejects_framework_dependency_change(self):
        with patch.object(check.metadata, "version", return_value="2.14.1"):
            with self.assertRaisesRegex(ValueError, "Framework dependency changed: torch"):
                check.verify(self.directory, self.version)

    def test_verify_rejects_wrong_candidate_version(self):
        versions = {name: "0.17.1" for name in check.CANDIDATES}
        versions["torch"] = "2.14.0"
        with patch.object(check.metadata, "version", side_effect=versions.__getitem__):
            with self.assertRaisesRegex(ValueError, "Wrong candidate version"):
                check.verify(self.directory, self.version)

    def test_incomplete_wheel_set_is_rejected_before_installation(self):
        with patch.object(check.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Expected four candidate wheels"):
                check.install(self.directory, self.directory, self.version)
            run.assert_not_called()

    def test_pip_failure_stops_verification(self):
        for name in check.CANDIDATES:
            (self.directory / f"{name}.whl").touch()
        failure = subprocess.CalledProcessError(1, ["pip", "install"])
        with patch.object(check.subprocess, "run", side_effect=failure) as run:
            with patch.object(check, "verify") as verify:
                with self.assertRaises(subprocess.CalledProcessError):
                    check.install(self.directory, self.directory, self.version)
                self.assertIn("--constraint", run.call_args.args[0])
                self.assertTrue(run.call_args.kwargs["check"])
                verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
