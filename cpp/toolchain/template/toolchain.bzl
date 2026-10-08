"""Rules used to setup the C++ toolchain."""

load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "tool_path")  # buildifier: disable=deprecated-function
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/toolchains:cc_toolchain.bzl", "cc_toolchain")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
)

def _toolchain_identifier(name):
    return "%s-toolchain" % name

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

all_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.lto_backend,
]

# Statically link c++ standard library
# https://bazel.build/tutorials/ccp-toolchain-config
features = [
    # Compilation modes. Bazel enables the feature whose NAME matches --compilation_mode, so these
    # three are what make `-c opt` and `-c dbg` mean anything here. None of them is `enabled = True`:
    # Bazel turns exactly one on per build, and forcing one would apply it in every mode.
    #
    # With no no such features the effect is silent. A toolchain that defines none still ACCEPTS `-c opt`
    #  - it just emits no -O flag, so gcc fell back to its own default of -O0 and every build was unoptimised:
    #  `bazel build -c opt` and a plain `bazel build`
    #
    # Worth carrying: the flag and the mode are two switches, and BOTH have to be right. A mode that
    # silently does nothing looks exactly like a mode that works.
    #
    # Flags are spelled for gcc because this toolchain only ever drives gcc, for linux-x86_64 and
    # linux-aarch64. A compiler that spells optimisation differently would need its own feature; it
    # would not silently get the wrong flags.
    feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        # -DNDEBUG goes with -O2, as it does in Bazel's own toolchains: it is what
                        # drops assert() from release builds.
                        flags = [
                            "-O2",
                            "-DNDEBUG",
                        ],
                    ),
                ]),
            ),
        ],
    ),
    feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        # -O0 is stated rather than left to gcc's default. It is already gcc's
                        # default, so this changes nothing today - but relying on a compiler default
                        # to carry a build decision is the exact thing that hid the missing -O2.
                        flags = [
                            "-O0",
                            "-g",
                        ],
                    ),
                ]),
            ),
        ],
    ),
    feature(
        # No flags, and that is deliberate: fastbuild is meant to be the quick unoptimised build, and
        # gcc gives that with no -O flag at all. Declared anyway so all three modes are visible in one
        # place - an undeclared mode is what made this hard to see the first time.
        name = "fastbuild",
    ),
    feature(
        name = "default_linker_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = ([
                    flag_group(
                        flags = [
                            "-static-libstdc++",
                            "-l:libstdc++.a"
                        ],
                    ),
                ]),
            ),
        ],
    ),
]


def _impl(ctx):
    tool_paths = [tool_path(name = k, path = v) for k, v in ctx.attr.tools.items()]

    # Documented at
    # https://docs.bazel.build/versions/main/skylark/lib/cc_common.html#create_cc_toolchain_config_info.
    #
    # create_cc_toolchain_config_info is the public interface for registering
    # C++ toolchain behavior.
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = _toolchain_identifier(ctx.attr.name),
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = ctx.attr.arch,
        target_libc = "unknown",
        compiler = "gcc",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
        cxx_builtin_include_directories = ctx.attr.builtin_includes,
        features = features
    )


_toolchain_config = rule(
    implementation = _impl,
    # You can alternatively define attributes here that make it possible to
    # instantiate different cc_toolchain_config targets with different behavior.
    attrs = {
        "tools": attr.string_dict(mandatory = True),
        "builtin_includes": attr.string_list(mandatory = True),
        "arch": attr.string(
            mandatory = True,
            doc = "The architecture (eg: x86_64 / aarch64)",
        ),
    },
    provides = [CcToolchainConfigInfo],
)


def define_toolchain(name, arch, tools, builtin_includes):
    """Define a native Linux GCC toolchain discovered from the build environment.

    Args:
      name: Repository-local toolchain name.
      arch: Native host and target architecture.
      tools: Absolute paths to compiler and binutils executables.
      builtin_includes: GCC's resolved system include search paths.
    """

    native.platform(
        name = name,
        constraint_values = [
            "@platforms//cpu:%s" % arch,
            "@platforms//os:linux",
        ],
    )

    toolchain_config_name = "%s_toolchain_config" % name
    _toolchain_config(name = toolchain_config_name, arch = arch, tools = tools, builtin_includes = builtin_includes)

    empty_target_name = "%s_empty" % name
    empty_target_label = ":%s" % empty_target_name
    native.filegroup(name = empty_target_name)

    cc_toolchain(
        name = "%s_toolchain" % name,
        toolchain_identifier = _toolchain_identifier(name),
        toolchain_config = ":%s" % toolchain_config_name,
        all_files = empty_target_label,
        compiler_files = empty_target_label,
        dwp_files = empty_target_label,
        linker_files = empty_target_label,
        objcopy_files = empty_target_label,
        strip_files = empty_target_label,
        supports_param_files = 0,
    )

    native.toolchain(
        name = "%s_linux_toolchain" % name,
        exec_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:%s" % arch,
        ],
        target_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:%s" % arch,
        ],
        toolchain = ":%s_toolchain" % name,
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    )
