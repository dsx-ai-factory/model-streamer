"""Configures the C++ toolchain."""

def _cc_autoconf_toolchain_impl(repository_ctx):
    # manylinux supplies a native GCC toolset in PATH. Discover its real paths
    # and include directories instead of assuming Ubuntu's cross-toolchain layout.
    arch = repository_ctx.execute(["uname", "-m"]).stdout.strip()
    if arch not in ["x86_64", "aarch64"]:
        fail("Unsupported native build architecture: " + arch)
    tools = {}
    for tool in ["ld", "gcc", "g++", "ar", "cpp", "gcov", "nm", "objdump", "strip"]:
        path = repository_ctx.which(tool)
        if path == None:
            fail("Missing native build tool: " + tool)
        tools[tool] = str(path)
    result = repository_ctx.execute([tools["gcc"], "-E", "-x", "c++", "/dev/null", "-v"])
    if result.return_code != 0:
        fail(result.stderr)
    includes = []
    in_search = False
    for line in result.stderr.splitlines():
        if line == "#include <...> search starts here:":
            in_search = True
        elif line == "End of search list.":
            in_search = False
        elif in_search:
            includes.append(str(repository_ctx.path(line.strip()).realpath))
    if not includes:
        fail("Could not discover GCC system includes")
    statement = "define_toolchain(name = %r, arch = %r, tools = %r, builtin_includes = %r)" % (arch, arch, tools, includes)

    repository_ctx.template(
        "BUILD",
        repository_ctx.path(Label("//toolchain/template:BUILD.tpl")),
        {
            "%{DEFINE_STATEMENTS}": statement,
        },
    )
    repository_ctx.file("toolchain.bzl", repository_ctx.read(Label("//toolchain/template:toolchain.bzl")))

_cc_autoconf_toolchain = repository_rule(
    implementation = _cc_autoconf_toolchain_impl,
    # Indicates that the repository inspects the system for configuration purpose
    configure = True,
    environ = ["PATH"],
)

def configure_toolchain(name):
    """configure_toolchain creates and registers toolchain targets for ARCHITECTURES in a repository.

    Toolchains can be referenced by "@<name>//:<arch>"

    Args:
      name: The name of the toolchain repository.
    """
    _cc_autoconf_toolchain(name = name)
    native.register_toolchains(
        "@%s//:all" % name,
    )
