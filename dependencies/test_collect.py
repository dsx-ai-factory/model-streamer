# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the parts of the dependency collector that need no network and no container.

    python3 dependencies/test_collect.py
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import collect  # noqa: E402
import container_sources  # noqa: E402
import dockerfile  # noqa: E402
import inputs  # noqa: E402
import util  # noqa: E402

SAMPLE = r"""
FROM nvidia/cuda:12.8.0-devel-ubuntu20.04@sha256:%s AS cuda-headers
FROM ubuntu:20.04

# a comment
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    # a comment inside a continued RUN
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade "pip==25.0.1" \
    && pip install boto3 cpplint==1.5.4

RUN WEED_ARCH=$([ "$(uname -m)" = "aarch64" ] && echo "arm64" || echo "amd64") \
    && if [ "$WEED_ARCH" = "amd64" ]; then WEED_SHA256=%s; \
       else WEED_SHA256=%s; fi \
    && curl -fsSL -o weed.tar.gz https://github.com/seaweedfs/seaweedfs/releases/download/4.47/linux_${WEED_ARCH}.tar.gz \
    && echo "${WEED_SHA256}  weed.tar.gz" | sha256sum -c -

RUN curl -fsSL -o libaio-0.3.113.tar.gz https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz && \
    echo "%s  libaio-0.3.113.tar.gz" | sha256sum -c - && tar zxf libaio-0.3.113.tar.gz

RUN curl -fsSL -o fake.tar.gz https://example.org/fake_1.0_Linux_$([ "$(uname -m)" = "aarch64" ] && echo "arm64" || echo "amd64").tar.gz

RUN git clone --recurse-submodules --depth 1 --branch "1.11.584" https://github.com/aws/aws-sdk-cpp /opt/aws-sdk-cpp
RUN git clone --recurse-submodules https://github.com/Azure/azure-sdk-for-cpp /opt/azure && cd /opt/azure && git checkout azure-storage-blobs_12.15.0
RUN npm install -g azurite
""" % ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


class DockerfileTest(unittest.TestCase):
    def test_base_images(self):
        images = dockerfile.base_images(SAMPLE)
        self.assertEqual(images[0][1:], ("nvidia/cuda", "12.8.0-devel-ubuntu20.04", "sha256:" + "a" * 64, "cuda-headers"))
        self.assertEqual(images[1][1:], ("ubuntu", "20.04", None, None))

    def test_apt_packages_skip_flags_and_comments(self):
        self.assertEqual([p for _, p in dockerfile.apt_packages(SAMPLE)], [["python3", "wget"]])

    def test_apt_pins_keep_the_package_and_option_values_are_skipped(self):
        text = "RUN apt-get install -y -o Dpkg::Options=x foo=1.2 bar\n"
        self.assertEqual([p for _, p in dockerfile.apt_packages(text)], [["foo", "bar"]])

    def test_registry_port_is_not_a_tag(self):
        images = dockerfile.base_images("FROM localhost:5000/team/app:1 AS x\nFROM localhost:5000/team/app\n")
        self.assertEqual([i[1:] for i in images], [("localhost:5000/team/app", "1", None, "x"),
                                                   ("localhost:5000/team/app", None, None, None)])

    def test_pip_and_npm_packages(self):
        self.assertEqual([p for _, p in dockerfile.pip_packages(SAMPLE)], [["pip==25.0.1"], ["boto3", "cpplint==1.5.4"]])
        self.assertEqual([p for _, p in dockerfile.npm_packages(SAMPLE)], [["azurite"]])

    def test_architecture_dependent_download_with_one_hash_each(self):
        weed = dockerfile.downloads(SAMPLE)[0]
        self.assertEqual(weed["urls"]["x86_64"], "https://github.com/seaweedfs/seaweedfs/releases/download/4.47/linux_amd64.tar.gz")
        self.assertEqual(weed["urls"]["aarch64"], "https://github.com/seaweedfs/seaweedfs/releases/download/4.47/linux_arm64.tar.gz")
        self.assertEqual(weed["sha256"], {"x86_64": "b" * 64, "aarch64": "c" * 64})

    def test_pinned_hash_follows_its_file_name(self):
        libaio = dockerfile.downloads(SAMPLE)[1]
        self.assertEqual(libaio["sha256"], {"any": "d" * 64})

    def test_command_substitution_inside_a_url_is_not_split(self):
        fake = dockerfile.downloads(SAMPLE)[2]
        self.assertEqual(fake["urls"], {"x86_64": "https://example.org/fake_1.0_Linux_amd64.tar.gz",
                                        "aarch64": "https://example.org/fake_1.0_Linux_arm64.tar.gz"})
        self.assertEqual(fake["sha256"], {})

    def test_git_clones_take_the_branch_or_the_checkout(self):
        aws, azure = dockerfile.git_clones(SAMPLE)
        self.assertEqual((aws["url"], aws["ref"], aws["recursive"]), ("https://github.com/aws/aws-sdk-cpp", "1.11.584", True))
        self.assertEqual(azure["ref"], "azure-storage-blobs_12.15.0")

    def test_two_clones_in_one_run_keep_their_own_checkout(self):
        text = ("RUN git clone https://h/a /opt/a && cd /opt/a && git checkout v1 && \\\n"
                "    git clone ssh://h/b /opt/b && git clone git@h:org/c.git --branch v3 /opt/c\n")
        first, second, third = dockerfile.git_clones(text)
        self.assertEqual([(c["url"], c["ref"]) for c in (first, second, third)],
                         [("https://h/a", "v1"), ("ssh://h/b", None), ("git@h:org/c.git", "v3")])

    def test_clone_without_a_url_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no repository URL"):
            dockerfile.git_clones("RUN git clone --depth 1 somewhere\n")


class InputsTest(unittest.TestCase):
    """The readers run against this repository's own files."""

    def test_apt_roots_include_dockerfile_and_feature_packages(self):
        roots = inputs.apt_roots(util.REPO)
        self.assertIn("python3", roots)
        self.assertIn("vim", roots)
        self.assertEqual(roots["nodejs"]["categories"], ["test"])

    def test_makefile_installs_are_test_packages(self):
        test = inputs.pip_groups(util.REPO)["test"]
        self.assertEqual(test["declared_by"]["azure-identity"], ["py/runai_model_streamer_azure/Makefile"])
        self.assertIn("azure-storage-blob", test["specs"])
        self.assertFalse([s for s in test["specs"] if "/" in s or s.startswith(("-", "."))])  # no paths or wheels of ours

    def test_pip_groups(self):
        groups = inputs.pip_groups(util.REPO)
        self.assertIn("boto3", groups["run"]["specs"])
        self.assertIn("numpy==1.24.4", groups["test"]["specs"])
        self.assertTrue(groups["test"]["extra_index_urls"])
        self.assertEqual(groups["run"]["declared_by"]["azure-identity"], ["py/runai_model_streamer_azure/setup.py"])
        self.assertTrue(any(spec.startswith("setuptools") for spec in groups["build"]["specs"]))

    def test_every_workflow_action_has_a_reference(self):
        for (action, ref), workflows in inputs.workflow_actions(util.REPO).items():
            self.assertTrue(ref, action)
            self.assertTrue(workflows)


class NamesTest(unittest.TestCase):
    def test_archive_names(self):
        self.assertEqual(util.archive_name_version("openssl-3.5.0.tar.gz", ""), ("openssl", "3.5.0"))
        self.assertEqual(util.archive_name_version("fake-gcs-server_1.52.2.tar.gz", ""), ("fake-gcs-server", "1.52.2"))
        self.assertEqual(util.archive_name_version("weed.tar.gz", "https://github.com/seaweedfs/seaweedfs/releases/download/4.47/linux_amd64.tar.gz"),
                         ("seaweedfs", "4.47"))

    def test_bazel_versions(self):
        self.assertEqual(container_sources._version_from("https://github.com/abseil/abseil-cpp/archive/20250127.1.tar.gz", "abseil-cpp-20250127.1"), "20250127.1")
        self.assertEqual(container_sources._version_from("https://github.com/google/boringssl/archive/82a53d8c902f940eb1310f76a0b96c40c67f632f.tar.gz",
                                                         "boringssl-82a53d8c902f940eb1310f76a0b96c40c67f632f"), "82a53d8c902f")

    def test_bazel_date_version(self):
        self.assertEqual(container_sources._version_from("https://github.com/google/re2/archive/2022-04-01.tar.gz", "re2-2022-04-01"), "2022-04-01")

    def test_bazel_rule_attributes(self):
        body = '  name = "x",\n  urls = ["https://a", "https://b"],\n  sha256 = "ff",\n  go_env = {},\n'
        self.assertEqual(container_sources._attributes(body), {"name": "x", "urls": ["https://a", "https://b"], "sha256": "ff"})


APT_HISTORY = """Start-Date: 2025-04-04  02:08:11
Commandline: apt-get install --reinstall mawk
Reinstall: mawk:amd64 (1.3.4)

Start-Date: 2026-10-06  09:25:47
Commandline: apt-get install -y cmake git curl libssl-dev
Install: libmpc3:amd64 (1.1.0-1, automatic), cmake:amd64 (3.16.3), git:amd64 (1:2.25.1)

Start-Date: 2026-10-06  09:26:32
Commandline: apt install -y apt-transport-https ca-certificates curl gnupg
Install: gnupg:amd64 (2.2.19, automatic), apt-transport-https:amd64 (2.0.11)

Start-Date: 2026-10-12  14:41:44
Commandline: apt-get -y install --no-install-recommends less ssh vim
Install: vim:amd64 (2:8.1)
"""


class AptHistoryTest(unittest.TestCase):
    def test_first_transaction_and_automatic_flag(self):
        history = container_sources.parse_apt_history(APT_HISTORY)
        self.assertEqual(history["libmpc3"], ("apt-get install -y cmake git curl libssl-dev", True))
        self.assertEqual(history["cmake"][1], False)
        self.assertNotIn("mawk", history)  # a reinstall of a base package is not an install

    def test_an_upgrade_is_attributed_to_the_command_that_pulled_it_in(self):
        text = "Commandline: apt-get install -y libc6-dev\nInstall: libc6-dev:amd64 (2.31)\nUpgrade: libc6:amd64 (2.31-17, 2.31-18)\n"
        self.assertEqual(container_sources.parse_apt_history(text)["libc6"], ("apt-get install -y libc6-dev", False))

    def test_attribution(self):
        history = container_sources.parse_apt_history(APT_HISTORY)
        roots = {"cmake": {"categories": ["build"], "declared_by": ["Dockerfile:59"]},
                 "curl": {"categories": ["build"], "declared_by": ["Dockerfile:59"]},
                 "vim": {"categories": ["build"], "declared_by": ["devcontainer.json"]}}
        commands = {"apt-transport-https ca-certificates curl gnupg": ["test"]}
        added = ["libmpc3", "gnupg", "vim", "unknown"]
        result = container_sources.attribute_apt(added, history, roots, commands)
        categories, declared, automatic = result["libmpc3"]
        self.assertEqual((categories, automatic), (["build"], True))
        self.assertIn("Dockerfile:59", declared)
        self.assertIn("apt history: apt-get install -y cmake git curl libssl-dev", declared)
        # the nodesource command is overridden: test, and curl's Dockerfile line is not attached
        self.assertEqual(result["gnupg"][0], ["test"])
        self.assertNotIn("Dockerfile:59", result["gnupg"][1])
        self.assertEqual(result["vim"][1], ["apt history: apt-get -y install --no-install-recommends less ssh vim", "devcontainer.json"])
        self.assertIn("not in the apt history", result["unknown"][1][0])


class PipTagsTest(unittest.TestCase):
    def test_aarch64_resolution_follows_the_interpreter_version(self):
        for version, tag in (((3, 8), "cp38"), ((3, 10), "cp310")):
            tags = container_sources.aarch64_pip_tags(version)
            self.assertEqual(tags[tags.index("--python-version") + 1], "%d.%d" % version)
            self.assertEqual(tags[tags.index("--abi") + 1], tag)


class ImageNameTest(unittest.TestCase):
    def test_registry_and_repository(self):
        self.assertEqual(util.image_parts("ubuntu"), ("docker.io", "library/ubuntu"))
        self.assertEqual(util.image_parts("nvidia/cuda"), ("docker.io", "nvidia/cuda"))
        self.assertEqual(util.image_parts("ghcr.io/org/name"), ("ghcr.io", "org/name"))
        self.assertEqual(util.image_parts("nvcr.io/nvidia/cuda"), ("nvcr.io", "nvidia/cuda"))
        self.assertEqual(util.image_parts("localhost:5000/team/app"), ("localhost:5000", "team/app"))

    def test_unknown_registry_has_no_anonymous_lookup(self):
        self.assertIsNone(util.oci_digest("nvcr.io", "nvidia/cuda", "12.8.0"))

    def test_the_last_from_is_the_base_and_any_image_is_listed(self):
        text = "FROM quay.io/org/builder:1 AS builder\nFROM debian\nRUN true\n"
        self.assertEqual([i[1:3] for i in dockerfile.base_images(text)], [("quay.io/org/builder", "1"), ("debian", None)])


class MergeTest(unittest.TestCase):
    def entry(self, categories, platform, url, declared):
        return util.entry("numpy", "1.24.4", "pip", categories, [util.file_record(url, platform=platform)], declared)

    def test_merge_unites_categories_and_files(self):
        merged = collect.merge([self.entry(["run"], "x86_64", "https://x/a", ["a"]),
                                self.entry(["test"], "aarch64", "https://x/b", ["b"]),
                                self.entry(["test"], "aarch64", "https://x/b", ["b"])])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["categories"], ["run", "test"])
        self.assertEqual(merged[0]["declared_by"], ["a", "b"])
        self.assertEqual([f["platform"] for f in merged[0]["files"]], ["x86_64", "aarch64"])

    def test_ship_tag_and_unmatched_keys(self):
        item = util.entry("openssl", "3.5.0", "url", ["build"], [], [])
        other = util.entry("openssl", "1.1.1", "apt", ["build"], [], [])
        unmatched = collect.tag_ship([item, other])
        self.assertEqual(item["categories"], ["build", "ship"])
        self.assertEqual(other["categories"], ["build"])
        self.assertIn("url:liburing", unmatched)  # the map names it, the list above does not
        self.assertNotIn("url:openssl", unmatched)

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(AssertionError):
            util.entry("x", "1", "pip", ["dev"], [], [])


if __name__ == "__main__":
    unittest.main()
