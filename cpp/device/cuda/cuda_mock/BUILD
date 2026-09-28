load("//:rules.bzl", "runai_portable_so")

# A fake CUDA driver, for testing the loader with no GPU.
#
# Built as libcuda.so.1 - the name dlopen searches for - so a test binary's DT_RPATH finds it ahead of
# any real driver. That is what makes the loader tests behave identically on a GPU machine and on
# CPU-only CI, rather than quietly testing the real driver on one and nothing on the other.
#
# TEST ONLY. Nothing shipped may depend on this: a fake driver inside the library would resolve every
# symbol and then do nothing.
runai_portable_so(
    name = "libcuda.so.1",
    srcs = ["libcuda_mock.cc"],
    ldscript = ":libcuda.ldscript",
    testonly = True,
    visibility = ["//device/cuda:__subpackages__"],
)
