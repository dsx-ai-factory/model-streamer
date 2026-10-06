# Toolchain

`configure_toolchain()` discovers the native GCC and binutils executables from
`PATH` and asks GCC for its system include search paths. This supports the
versioned GCC toolset in the pinned manylinux development images without
assuming Ubuntu cross-compiler paths.

Each image provides one native target, `x86_64` or `aarch64`. CI uses native
runners for both. Use `bazel build --config=x86_64 //...` on x86_64 or
`bazel build --config=aarch64 //...` on ARM64. A foreign target must be built
inside a matching container or on a matching host.
