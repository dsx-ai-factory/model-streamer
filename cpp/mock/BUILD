load("//:rules.bzl", "runai_portable_so")

runai_portable_so(
    name = "libstreamer-mock.so",
    srcs = ["streamer-mock.cc"],
    deps = [
        "//utils/fd",
        "//streamer/api",
        "//common/response_code",
        "//common/submission",
        # For runai_direct_block_size only. Checked: this pulls in no liburing and no libaio - the
        # mock stays free of the engines it is standing in for.
        "//posix_io/alignment",
    ],
    ldscript = "//streamer:streamer.ldscript",
)
