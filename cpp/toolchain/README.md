# Toolchain

This directory contains configuration for generating toolchain targets
from a standard GCC/G++ toolchain.

The toolchain is installed at the repository the `cc_autoconf_toolchain` repository rule creates.
It registers only the running container's architecture (`aarch64` or `x86_64`),
using native GCC/G++ and binutils from `/usr/bin`. Cross-compilation is not supported.

See .devcontainer/Dockerfile for toolchain Ubuntu package installation.

## Using the toolchain

Place the following in your `MODULE.bazel` file:

```
cc_autoconf_toolchain = use_repo_rule("//toolchain:configure.bzl", "cc_autoconf_toolchain")
cc_autoconf_toolchain(name = "my_awesome_toolchain")
register_toolchains("@my_awesome_toolchain//:all")
```

This will allow bazel to reference generated targets using the name of the repository:

```
bazel build --platforms=@my_awesome_toolchain//:aarch64 //mytarget
```

The example targets an ARM64 container. Use `x86_64` in an AMD64 container.
In this repository, `make build` builds and packages only the native architecture.
Build the other architecture in its matching devcontainer on a native runner.
