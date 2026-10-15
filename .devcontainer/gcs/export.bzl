# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the complete C++ SDK interface, without maintaining a library list."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _export_sdk_impl(ctx):
    info = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    compilation = info.compilation_context
    headers = compilation.headers.to_list()
    libraries = {}
    files = list(headers)
    linkopts = []
    for linker_input in info.linking_context.linker_inputs.to_list():
        linkopts.extend(linker_input.user_link_flags)
        for library in linker_input.libraries:
            archive = library.pic_static_library or library.static_library
            if not archive:
                fail("GCS prebuild requires static libraries: %s" % linker_input.owner)
            previous = libraries.get(archive.path)
            libraries[archive.path] = struct(
                path = archive.path,
                alwayslink = library.alwayslink or (previous != None and previous.alwayslink),
            )
            files.append(archive)

    manifest = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(manifest, json.encode(struct(
        source_checksums = ctx.attr.source_checksums,
        headers = [header.path for header in headers],
        includes = depset(transitive = [
            compilation.includes,
            compilation.quote_includes,
            compilation.system_includes,
        ]).to_list(),
        defines = compilation.defines.to_list(),
        libraries = libraries.values(),
        linkopts = linkopts,
    )))

    # Requesting this target builds every archive and generated header, including
    # protobuf/gRPC outputs. The installer never needs the Bazel cache at runtime.
    return [DefaultInfo(files = depset([manifest] + files))]

export_sdk = rule(
    implementation = _export_sdk_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "source_checksums": attr.string_dict(mandatory = True),
    },
)
