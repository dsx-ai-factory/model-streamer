load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "filesystem_worker",
    deps = [
        "//streamer/impl/device_io/device_issuer",
        "//streamer/impl/workload",
        "//utils/logging",
        "//utils/threadpool",
    ],
    visibility = ["//streamer:__subpackages__"],
)

runai_cc_test(
    name = "filesystem_worker_test",
    srcs = ["filesystem_worker_test.cc"],
    deps = [
        "//device/mock",
        "//utils/temp/file",
        "//utils/random",
        ":filesystem_worker",
        "//streamer/impl/batches",
    ],
)
