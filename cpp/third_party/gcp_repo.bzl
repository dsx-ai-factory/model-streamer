# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume the SDK installed by the devcontainer, validating SDK-owned inputs."""

def sdk_build_input_matches(actual, recorded):
    """Compare checkout text with install.py's universal-newline metadata."""
    return actual.replace("\r\n", "\n").replace("\r", "\n") == recorded

def _prebuilt_gcp_impl(ctx):
    root = ctx.path(ctx.attr.sdk_root)
    source = ctx.path(Label("@gcs_sdk_config//:BUILD.bazel")).realpath.dirname.dirname
    targets = []
    for arch in ["x86_64", "aarch64"]:
        package = root.get_child(arch)
        manifest = package.get_child("manifest.json")
        # Native CI images intentionally install only one architecture.
        if not manifest.exists:
            continue
        metadata = json.decode(ctx.read(manifest))
        inputs = metadata.get("build_inputs", {})
        # MODULE.bazel fixes the SDK's google-cloud-cpp version: a bump there is a mismatch below.
        if not inputs or ".bazelrc" not in inputs or ".bazelversion" not in inputs or "MODULE.bazel" not in inputs:
            fail("Prebuilt GCS SDK at %s lacks SDK build metadata. Rebuild the devcontainer." % package)
        for name, content in inputs.items():
            path = source.get_child(name)
            if not path.exists or not sdk_build_input_matches(ctx.read(path), content):
                fail("Prebuilt GCS SDK build input %s changed. Rebuild the devcontainer." % name)
        compiler = "/usr/bin/g++"
        result = ctx.execute([compiler, "--version"])
        if result.return_code or result.stdout != metadata.get("compiler_version") or metadata.get("compiler") != compiler:
            fail("Prebuilt GCS SDK compiler for %s changed. Rebuild the devcontainer." % arch)
        ctx.symlink(package, arch)
        targets.append('        ":%s": "//%s:sdk",' % (arch, arch))
    if not targets:
        fail("Prebuilt GCS SDK missing at %s. Rebuild the devcontainer for this checkout." % root)
    ctx.file("WORKSPACE", "")
    ctx.template("BUILD.bazel", ctx.attr.build_file, {"%{SDK_TARGETS}": "\n".join(targets)})

prebuilt_gcp = repository_rule(
    implementation = _prebuilt_gcp_impl,
    local = True,
    attrs = {
        "sdk_root": attr.string(default = "/opt/gcs-sdk"),
        "build_file": attr.label(default = "//third_party:gcp.BUILD.tpl"),
    },
)
