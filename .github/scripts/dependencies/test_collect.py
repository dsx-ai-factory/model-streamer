# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the parts of the dependency collector that need no network and no container.

    python3 .github/scripts/dependencies/test_collect.py
"""

import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import collect  # noqa: E402
import completeness  # noqa: E402
import container_sources  # noqa: E402
import dockerfile  # noqa: E402
import inputs  # noqa: E402
import static_sources  # noqa: E402
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


SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail

target_arch=$1
case "$target_arch" in
    amd64) tool_sha256=%s ;;
    arm64) tool_sha256=%s ;;
    *)
        echo "Unsupported: $target_arch" >&2
        exit 2
        ;;
esac

case "$1" in
    amd64)
        PLATFORM=linux-x86_64
        SUM=%s
        ;;
    arm64)
        PLATFORM=linux-sbsa
        SUM=%s
        ;;
esac
ARCHIVE="cuda_cudart-${PLATFORM}-12.8.57-archive"

curl -fsSL -o tool.tar.gz "https://example.org/tool/1.0/linux_${target_arch}.tar.gz"
echo "${tool_sha256}  tool.tar.gz" | sha256sum -c -
curl -fsSL -o /tmp/cuda.tar.xz \
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/${PLATFORM}/${ARCHIVE}.tar.xz"
echo "${SUM}  /tmp/cuda.tar.xz" | sha256sum -c -
build_zlib() {
    curl -fsSL -o zlib-1.3.1.tar.gz https://example.org/zlib-1.3.1.tar.gz
    echo "%s  zlib-1.3.1.tar.gz" | sha256sum -c -
}
git clone --recurse-submodules https://example.org/org/sdk /opt/sdk
cd /opt/sdk
git checkout sdk_1.2
apt-get install -y nodejs
npm install -g azurite
""" % ("1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)


class ScriptTest(unittest.TestCase):
    """The shell scripts the Dockerfile runs are read like its RUN lines."""
    PATH = ".devcontainer/scripts/x.sh"

    def downloads(self):
        return dockerfile.downloads(SCRIPT, self.PATH)

    def test_architecture_argument_and_case_branches_give_a_url_and_a_hash_per_architecture(self):
        tool = self.downloads()[0]
        self.assertEqual(tool["urls"], {"x86_64": "https://example.org/tool/1.0/linux_amd64.tar.gz",
                                        "aarch64": "https://example.org/tool/1.0/linux_arm64.tar.gz"})
        self.assertEqual(tool["sha256"], {"x86_64": "1" * 64, "aarch64": "2" * 64})

    def test_variables_that_name_each_other_are_expanded(self):
        cuda = self.downloads()[1]
        self.assertEqual(cuda["filename"], "/tmp/cuda.tar.xz")  # the hash is pinned under this name
        self.assertEqual(cuda["urls"]["aarch64"], "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/"
                                                  "linux-sbsa/cuda_cudart-linux-sbsa-12.8.57-archive.tar.xz")
        self.assertEqual(cuda["sha256"], {"x86_64": "3" * 64, "aarch64": "4" * 64})

    def test_a_url_without_variables_is_the_same_for_both_architectures(self):
        zlib = self.downloads()[2]
        self.assertEqual(zlib["urls"], {"any": "https://example.org/zlib-1.3.1.tar.gz"})
        self.assertEqual(zlib["sha256"], {"any": "5" * 64})  # the pin is a line below the curl, inside a function

    def test_an_unknown_variable_in_a_url_is_an_error(self):
        with self.assertRaisesRegex(ValueError, r"x\.sh line 1: cannot work out the URL"):
            dockerfile.downloads('curl -fsSL -o a.tar.gz "https://h/${UNKNOWN}/a.tar.gz"\n', self.PATH)

    def test_the_redistributable_archive_is_named_after_its_component(self):
        self.assertEqual(static_sources._download_name(self.downloads()[1]), ("cuda_cudart", "12.8.57"))

    def test_a_checkout_on_a_later_line_applies_to_the_clone(self):
        (clone,) = dockerfile.git_clones(SCRIPT, self.PATH)
        self.assertEqual((clone["url"], clone["ref"], clone["line"], clone["recursive"]), ("https://example.org/org/sdk", "sdk_1.2", 35, True))

    def test_apt_and_npm_packages_of_a_script(self):
        self.assertEqual(dockerfile.apt_packages(SCRIPT, self.PATH), [(38, ["nodejs"])])
        self.assertEqual(dockerfile.npm_packages(SCRIPT, self.PATH), [(39, ["azurite"])])

    def test_the_scripts_of_this_repository_are_read(self):
        files = inputs.build_files(util.REPO)
        self.assertEqual(files[0], ".devcontainer/Dockerfile")
        self.assertIn(".devcontainer/scripts/install-cuda-headers.sh", files)
        found = {(e["name"], e["version"]) for e in static_sources.downloads(util.REPO, compute_hashes=False)}
        self.assertIn(("cuda_cudart", "12.8.57"), found)
        self.assertIn(("libaio", "0.3.113"), found)
        self.assertIn("azurite", inputs.npm_roots(util.REPO))

    def test_every_url_ship_key_names_a_download_of_this_repository(self):
        names = {"url:" + e["name"] for e in static_sources.downloads(util.REPO, compute_hashes=False)}
        keys = {key for key in inputs.config()["ship"] if key.startswith("url:")}
        self.assertEqual(keys - names, set())


class CompletenessTest(unittest.TestCase):
    def repo_with(self, files):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for path, text in files.items():
            full = os.path.join(folder.name, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(text)
        subprocess.check_call(["git", "init", "-q", folder.name])
        subprocess.check_call(["git", "-C", folder.name, "add", "."])
        return folder.name

    def test_this_repository_is_covered(self):
        self.assertEqual(completeness.problems(util.REPO), [])

    def test_a_new_file_that_downloads_something_is_reported(self):
        repo = self.repo_with({"tools/get.sh": "curl -fsSL -o x.tgz https://example.org/x.tgz\n",
                               ".devcontainer/Dockerfile": "FROM ubuntu:20.04\n",
                               "docs/notes.md": "run `pip install x`\n"})
        problems = completeness.unread_files(repo)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("tools/get.sh "))

    def test_a_command_in_a_shape_the_parser_does_not_know_is_reported(self):
        repo = self.repo_with({".devcontainer/Dockerfile": "FROM ubuntu:20.04\n",
                               ".devcontainer/scripts/a.sh": 'x=$(curl -fsSL https://example.org/a.tgz -o a.tgz)\n'
                                                              "curl -fsSL -o b.tgz https://example.org/b.tgz\n"})
        problems = completeness.unparsed_commands(repo)
        self.assertEqual(problems, ["%s has 2 `curl` commands and the collector read 1. Teach dockerfile.py the new shape"
                                    % ".devcontainer/scripts/a.sh"])

    def test_wget_in_a_build_file_is_reported(self):
        repo = self.repo_with({".devcontainer/Dockerfile": "FROM ubuntu:20.04\n",
                               ".devcontainer/scripts/a.sh": "wget -O a.tgz https://example.org/a.tgz\n"})
        problems = completeness.unparsed_commands(repo)
        self.assertEqual(len(problems), 1)
        self.assertIn("a.sh downloads with `wget`", problems[0])

    def test_a_command_in_a_file_that_is_read_for_something_else_is_reported(self):
        repo = self.repo_with({".devcontainer/Dockerfile": "FROM ubuntu:20.04\n",
                               "py/x/Makefile": "install:\n\tpip3 install a\n\tcurl -fsSL -o b.tgz https://example.org/b.tgz\n",
                               ".github/workflows/w.yml": "steps:\n- run: sudo apt-get install -y jq\n- uses: actions/checkout@v5\n"})
        problems = completeness.unread_commands(repo)
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith(".github/workflows/w.yml has a `apt-get install` command"))
        self.assertTrue(problems[1].startswith("py/x/Makefile has a `curl` command"))  # its `pip3 install` is read

    def test_a_listed_command_is_allowed(self):
        repo = self.repo_with({".devcontainer/Dockerfile": "FROM ubuntu:20.04\n",
                               ".github/workflows/w.yml": "steps:\n- run: sudo apt-get install -y jq\n"})
        config = {"coverage": dict(inputs.config()["coverage"],
                                   allowed_commands={".github/workflows/w.yml": {"kinds": ["apt-get install"], "reason": "a CI tool"}})}
        with unittest.mock.patch.object(inputs, "config", return_value=config):
            self.assertEqual(completeness.unread_commands(repo), [])


class InputsTest(unittest.TestCase):
    """The readers run against this repository's own files."""

    def test_apt_roots_include_dockerfile_and_feature_packages(self):
        roots = inputs.apt_roots(util.REPO)
        self.assertIn("python3", roots)
        self.assertIn("vim", roots)
        self.assertEqual(roots["nodejs"]["categories"], ["test"])

    def test_makefile_installs_are_test_packages(self):
        test = inputs.pip_groups(util.REPO)["test"]
        self.assertEqual(test["declared_by"]["azure-identity"], ["py/runai_model_streamer_azure/requirements.txt"])  # `pip install -r requirements.txt`
        self.assertIn("azure-storage-blob", [inputs.spec_name(s) for s in test["specs"]])
        self.assertFalse([s for s in test["specs"] if "/" in s or s.startswith(("-", "."))])  # no paths or wheels of ours

    def test_pip_groups(self):
        groups = inputs.pip_groups(util.REPO)
        self.assertIn("boto3", groups["run"]["specs"])
        self.assertIn("numpy==1.24.4", groups["test"]["specs"])
        self.assertIn("torch>=2.6", groups["test"]["specs"])  # quoted in the Makefile; the quotes must not survive
        self.assertTrue(groups["test"]["extra_index_urls"])
        self.assertEqual(groups["run"]["declared_by"]["azure-identity"], ["py/runai_model_streamer_azure/requirements.txt"])
        self.assertTrue(any(spec.startswith("setuptools") for spec in groups["build"]["specs"]))

    def test_every_workflow_action_has_a_reference(self):
        for (action, ref), workflows in inputs.workflow_actions(util.REPO).items():
            self.assertTrue(ref, action)
            self.assertTrue(workflows)


# `bazel mod show_repo <module>...`, as Bazel 7.6 prints it.
BAZEL_SHOW_REPO = """\
## google_cloud_cpp@3.7.0:
# <builtin>
http_archive(
  name = "google_cloud_cpp~",
  urls = ["https://github.com/googleapis/google-cloud-cpp/archive/refs/tags/v3.7.0.tar.gz"],
  integrity = "sha256-ghNn8h7FNkMb9PtdxwmkJ5RGVQbX0/i+QNUtSTvP3QY=",
  strip_prefix = "google-cloud-cpp-3.7.0",
  remote_file_urls = {},
  remote_file_integrity = {},
  remote_patches = {"https://bcr.bazel.build/modules/google_cloud_cpp/3.7.0/patches/copyright.patch": "sha256-qITPA+QfNNAlErCIuUMoQeZ4e9sCEEAWmeTehRfLO+I="},
  remote_patch_strip = 0,
)
# Rule google_cloud_cpp~ instantiated at (most recent call last):
#   <builtin> in <toplevel>
# Rule http_archive defined at (most recent call last):
#   /root/.cache/bazel/external/bazel_tools/tools/build_defs/repo/http.bzl:387:31 in <toplevel>

## gcs_sdk_config@_:
# <builtin>
local_repository(
  name = "gcs_sdk_config~",
  path = "config",
)

## curl@8.8.0.bcr.3:
# <builtin>
http_archive(
  name = "curl~",
  urls = ["https://github.com/curl/curl/releases/download/curl-8_8_0/curl-8.8.0.tar.gz"],
  integrity = "sha256-d8DhzTWrW0W2WWRak7RtZgIk0AJPEYXoqVzbJ649eH0=",
  strip_prefix = "curl-8.8.0",
)
"""

# `bazel mod graph --output json`, cut down: the first occurrence of a module is expanded,
# repeats are marked `unexpanded`.
BAZEL_MOD_GRAPH = """\
{
  "key": "<root>",
  "name": "model_streamer_gcs_sdk",
  "version": "",
  "dependencies": [
    {
      "key": "gcs_sdk_config@_",
      "name": "gcs_sdk_config",
      "version": "",
      "dependencies": [
        {"key": "rules_cc@0.2.17", "name": "rules_cc", "version": "0.2.17", "unexpanded": true}
      ]
    },
    {
      "key": "google_cloud_cpp@3.7.0",
      "name": "google_cloud_cpp",
      "version": "3.7.0",
      "dependencies": [
        {"key": "rules_cc@0.2.17", "name": "rules_cc", "version": "0.2.17", "unexpanded": true},
        {
          "key": "curl@8.8.0.bcr.3",
          "name": "curl",
          "version": "8.8.0.bcr.3",
          "dependencies": []
        }
      ]
    },
    {"key": "rules_cc@0.2.17", "name": "rules_cc", "version": "0.2.17", "dependencies": []}
  ]
}
"""


class BazelTest(unittest.TestCase):
    def test_module_sections_carry_each_rule(self):
        sections = dict(container_sources._module_sections(BAZEL_SHOW_REPO))
        self.assertEqual(set(sections), {"google_cloud_cpp", "gcs_sdk_config", "curl"})
        rule = container_sources.RULE.search(sections["google_cloud_cpp"])
        self.assertEqual(rule.group(1), "http_archive")
        attributes = container_sources._attributes(rule.group(2))
        self.assertEqual(attributes["name"], "google_cloud_cpp~")
        self.assertEqual(attributes["urls"], ["https://github.com/googleapis/google-cloud-cpp/archive/refs/tags/v3.7.0.tar.gz"])
        self.assertEqual(container_sources.RULE.search(sections["gcs_sdk_config"]).group(1), "local_repository")

    def test_registry_integrity_becomes_a_sha256(self):
        rule = container_sources.RULE.search(dict(container_sources._module_sections(BAZEL_SHOW_REPO))["curl"])
        digest = container_sources._module_digest(container_sources._attributes(rule.group(2)))
        self.assertEqual(digest, ("sha256", "77c0e1cd35ab5b45b659645a93b46d660224d0024f1185e8a95cdb27ae3d787d",
                                  "Bazel module registry (integrity)"))
        self.assertEqual(container_sources._module_digest({"sha256": "ff"}), ("sha256", "ff", "Bazel module registry"))
        self.assertIsNone(container_sources._module_digest({}))

    def test_module_graph_takes_the_first_requiring_module(self):
        original = container_sources._bazel
        container_sources._bazel = lambda directory, *args: BAZEL_MOD_GRAPH
        try:
            modules = container_sources._module_graph("/work")
        finally:
            container_sources._bazel = original
        # Local override modules (no version) are not dependencies, and their children answer to their parent.
        self.assertEqual(modules, {"rules_cc": ("0.2.17", None),
                                   "google_cloud_cpp": ("3.7.0", None),
                                   "curl": ("8.8.0.bcr.3", "google_cloud_cpp")})

    def test_module_declarations_name_the_line(self):
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "MODULE.bazel")
            with open(path, "w") as f:
                f.write('module(name = "example")\n'
                        'bazel_dep(name = "google_cloud_cpp", version = "3.7.0")\n'
                        'bazel_dep(name = "rules_cc", version = "0.2.17")\n'
                        'single_version_override(module_name = "envoy_api", version = "0.0.0-20260130-84e8436")\n')
            self.assertEqual(container_sources._module_declarations(path),
                             {"google_cloud_cpp": 2, "rules_cc": 3, "envoy_api": 4})


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
