# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configures the C++ toolchain."""

load("//toolchain:rules.bzl", "native_tools")

OS = "linux-gnu"

def _get_gcc_version(repository_ctx):
    result = repository_ctx.execute([native_tools()["gcc"], "-dumpversion"])
    if result.return_code != 0 or not result.stdout.strip():
        fail("Cannot determine the native GCC version: " + result.stderr)
    return result.stdout.strip()

def _cc_autoconf_toolchain_impl(repository_ctx):
    result = repository_ctx.execute(["/usr/bin/uname", "-m"])
    arch = result.stdout.strip()
    if result.return_code != 0 or arch not in ["x86_64", "aarch64"]:
        fail("Unsupported native architecture: " + arch)
    gcc_version = _get_gcc_version(repository_ctx)
    define_statement = 'define_toolchain(name = "%s", os = "%s", arch = "%s", gcc_version = "%s")' % (arch, OS, arch, gcc_version)

    repository_ctx.template(
        "BUILD",
        repository_ctx.path(Label("//toolchain/template:BUILD.tpl")),
        {
            "%{DEFINE_STATEMENTS}": define_statement,
        },
    )
    repository_ctx.file("toolchain.bzl", repository_ctx.read(Label("//toolchain/template:toolchain.bzl")))
    repository_ctx.file("rules.bzl", repository_ctx.read(Label("//toolchain:rules.bzl")))

_cc_autoconf_toolchain = repository_rule(
    implementation = _cc_autoconf_toolchain_impl,
    # Indicates that the repository inspects the system for configuration purpose
    configure = True,
)

# Used from MODULE.bazel with use_repo_rule; the toolchain is registered there.
cc_autoconf_toolchain = _cc_autoconf_toolchain
