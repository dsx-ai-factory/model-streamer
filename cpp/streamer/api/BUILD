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
