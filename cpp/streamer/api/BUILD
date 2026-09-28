load("//:rules.bzl", "runai_cc_auto_library")

# The headers that ship in the SDK tarball, laid out exactly as they are installed: this directory is
# the tarball's include/ tree, so packaging copies it rather than rewriting it.
#
# strip_include_prefix makes streamer/api/streamer/x.h includable as streamer/x.h - the spelling an
# external consumer writes - and propagates to every dependent.
runai_cc_auto_library(
    name = "api",
    strip_include_prefix = "/streamer/api",
)

# A C compiler, deliberately. Not runai_cc_test: that one adds //cc/testing, which is gtest and C++,
# and would defeat the point - the test has to be built the way a consumer builds, or it proves
# nothing about them.
cc_test(
    name = "c_api_test",
    srcs = ["c_api_test.c"],
    copts = ["-std=c99", "-Wall", "-Wextra", "-pedantic"],
    # The same libraries libstreamer.so links (streamer/BUILD). The static library does not carry
    # them, and a C++ test only gets away without them because //cc/testing brings them in.
    linkopts = ["-lpthread", "-ldl", "-lrt", "-lm"],
    deps = ["//streamer:streamer"],
)

# The same source against the SHARED object, which is what a consumer actually links.
#
# Not a duplicate of the target above. That one links the static library, so it proves C linkage and
# nothing else. Only this one passes through streamer.ldscript, where `local: *` hides everything not
# listed under `global:`. A function can have correct C linkage and still be missing from that list -
# the static test passes, the consumer cannot link. Adding an entry point means editing the ldscript,
# and nothing else checks that it was done.
cc_test(
    name = "c_api_so_test",
    srcs = ["c_api_test.c", "//streamer:libstreamer.so"],
    copts = ["-std=c99", "-Wall", "-Wextra", "-pedantic"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../streamer"],
    deps = ["//streamer/api"],
)
