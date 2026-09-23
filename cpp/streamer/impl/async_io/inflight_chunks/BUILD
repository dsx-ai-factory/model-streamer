load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "inflight_chunks",
    deps = [
        "//posix_io/io_engine",
        "//streamer/impl/chunk_splitter",
        "//streamer/impl/device_io/staging_pool",
        "//utils/logging",
    ],
)

runai_cc_test(
    name = "inflight_chunks_test",
    srcs = ["inflight_chunks_test.cc"],
    deps = [
        ":inflight_chunks",
        "//common/exception",
        "//device/mock",
    ],
)
