# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Google SDK sources for image builds and prebuilt SDKs for consumers."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

GCP_VERSION = "2.37.0"
GCP_SHA256 = "10867580483cb338e7d50920c2383698f3572cc6b4c7d072e38d5f43755cbd80"
PROTOBUF_VERSION = "27.0"
PROTOBUF_SHA256 = "da288bf1daa6c04d03a9051781caa52aceb9163586bff9aa6cfb12f69b9395aa"
GCP_SOURCE_CHECKSUMS = {"google_cloud_cpp": GCP_SHA256, "protobuf": PROTOBUF_SHA256}

def load_protobuf_repo():
    """Provide protobuf's rule definitions required by rules_cc, without building it."""
    http_archive(
        name = "com_google_protobuf",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = "protobuf-" + PROTOBUF_VERSION,
        url = "https://github.com/protocolbuffers/protobuf/releases/download/v%s/protobuf-%s.tar.gz" % (PROTOBUF_VERSION, PROTOBUF_VERSION),
    )

def load_gcp_repo():
    """Fetch SDK sources only in the devcontainer's SDK builder workspace."""
    http_archive(
        name = "google_cloud_cpp",
        sha256 = GCP_SHA256,
        strip_prefix = "google-cloud-cpp-" + GCP_VERSION,
        url = "https://github.com/googleapis/google-cloud-cpp/archive/v%s.tar.gz" % GCP_VERSION,
    )

def _prebuilt_gcp_impl(ctx):
    root = ctx.path(ctx.attr.sdk_root)
    for arch in ["x86_64", "aarch64"]:
        package = root.get_child(arch)
        manifest = package.get_child("manifest.json")
        if not manifest.exists:
            fail("Prebuilt GCS SDK missing at %s. Rebuild the devcontainer for this checkout." % package)
        if json.decode(ctx.read(manifest)).get("source_checksums") != ctx.attr.source_checksums:
            fail("Prebuilt GCS SDK at %s does not match this checkout. Rebuild the devcontainer." % package)
        ctx.symlink(package, arch)
    ctx.file("WORKSPACE", "")
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

_prebuilt_gcp = repository_rule(
    implementation = _prebuilt_gcp_impl,
    local = True,
    attrs = {
        "sdk_root": attr.string(default = "/opt/gcs-sdk"),
        "source_checksums": attr.string_dict(mandatory = True),
        "build_file": attr.label(default = "//third_party:gcp.BUILD"),
    },
)

def load_prebuilt_gcp_repo():
    """Use the SDK installed by the devcontainer, rejecting stale SDK versions."""
    _prebuilt_gcp(name = "google_cloud_cpp", source_checksums = GCP_SOURCE_CHECKSUMS)
