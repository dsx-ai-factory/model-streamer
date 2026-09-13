load("//:rules.bzl", "runai_portable_so", "runai_cc_test", "runai_cc_auto_library")

runai_portable_so(
    name = "libstreamer.so",
    srcs = ["streamer.cc"],
    deps = ["//streamer/impl", "//streamer/api", "//common/submission", "//posix_io/alignment"],
    ldscript = ":streamer.ldscript",
    linkopts = ["-lpthread", "-Wl,-rpath,$$ORIGIN"]
)

runai_cc_auto_library(
    name = "streamer",
    deps = ["//streamer/impl", "//streamer/api", "//common/submission", "//posix_io/alignment"],
)

runai_cc_test(
    name = "streamer_test",
    srcs = ["streamer_test.cc"],
    deps = [":streamer",
            "//utils/random",
            "//utils/temp/file",
            "//utils/temp/env",
            "//utils/fd",
            "//utils/logging",
            "//utils/thread",
    ],
    linkstatic=False,
)

runai_cc_test(
    name = "streamer_s3_test",
    srcs = ["streamer_s3_test.cc"],
    deps = [":streamer",
            "//common/backend_api/object_storage",
           "//utils/random",
            "//utils/temp/env",
            "//utils/temp/file",
            "//utils/dylib",
            "//utils/logging",
            "//utils/fdlimit",
            "//s3/s3_mock:libstreamers3.so",
    ],
    linkstatic=False,
    data = ["//s3/s3_mock:libstreamers3.so"],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../s3/s3_mock",
    ],
)

exports_files(["streamer.ldscript"])
